import logging
import math
from typing import Any, Mapping, Optional, Union, Sequence, Tuple, Dict

import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F
from pytorch3d.ops import knn_points

import utils
from utils import ops_3d, FurthestSampling
from fast_2d_gs._C import get_C_function
from lietorch import SE3, SO3
from fast_2d_gs.gaussian_splatting import GaussianSplatting, BasicPointCloud, get_expon_lr_func, RGB2SH
from fast_2d_gs.losses.SC_GS_arap_loss import cal_connectivity_from_points, cal_arap_error
from datasets import NERF_Base_Dataset
from fast_2d_gs.losses.gs_flow_loss import GS_flow, get_gs_flow_matrix
from fast_2d_gs.renderer.gs_2d_render import GS_2D_compute_trans_mat, GS_2D_topk_weights
from .freq_encoder import FreqEncoder


class FastSuperpoint2DGaussianSplatting(GaussianSplatting):
    train_db_times: Tensor
    train_db_Tv2c: Tensor
    train_db_Tw2v: Tensor
    gs_knn_index: Tensor
    gs_knn_dist: Tensor
    # superpoints
    sp_is_init: Tensor
    p2sp: Optional[Tensor]
    sp_cache: Tensor
    """cache the results from deform_net, shape: [T, M, 14]"""
    sp_weights: Tensor
    sp_knn: Tensor
    _culling: Tensor
    factor_culling: Tensor
    mask_blur: Tensor

    def __init__(
            self,
            train_schedule: dict = None,
            num_knn=3,
            gs_knn_num=20,
            gs_knn_update_interval=(1000, 3000),
            canonical_time_id=-1,
            use_canonical_net=False,
            canonical_replace_steps=(),
            lr_feature_scale=2.5,
            lr_spatial_scale_fix=5.0,  # 与camera extent相关
            # deformable
            deform_net_cfg=None,
            deform_net_name='sc_gs',
            lr_deform_scale=1.0,
            lr_deform_max_steps=40000,
            # sp stage
            num_superpoints=512,
            hyper_dim=2,
            sp_prune_threshold=1e-3,
            sp_split_threshold=0.0002,
            sp_merge_threshold=0.01,
            warp_method='LBS',
            LBS_method='weighted_kernel',
            apply_rotation='add',
            # init stage
            node_max_num_ratio_during_init=16,
            init_num_times=16,
            init_sp_step=7500,
            init_sp_from='inputs',
            init_sp_same_points=True,
            init_same_scale='none',
            init_only_dxyz=True,
            smooth_time_init=0.1,
            smooth_time_steps=-1,  # 在t上添加噪声的步数
            # other
            test_time_interpolate=False,
            loss_arap_start_step=0,
            loss_flow_decompose_motion=True,
            which_rotation='quaternion',
            render_method='3d_gs',
            imp_metric='outdoor',
            # densify
            adaptive_control_cfg=None,
            simplify_steps=(3000, 8000, 13000),
            sampling_factor=0.6,
            depth_reinit_steps=(),
            num_depth_factor=1.,
            **kwargs
    ):
        adaptive_control_cfg = utils.merge_dict(
            adaptive_control_cfg,
            # node_densify_interval=[5000, 5000, 25_000],
            # node_force_densify_prune_step=10_000,
            # node_enable_densify_prune=False,
            sp_adjust_interval=[-5000, 5000, 25000],
            sp_merge_interval=[-1, 10_000, 20_000],
            blur_split=True,
            aggressive_clone_interval=[-250, 500, -1],
        )
        assert len(simplify_steps) == 3
        self.simplify_steps = simplify_steps
        super().__init__(**kwargs, adaptive_control_cfg=adaptive_control_cfg)
        # render
        self.render_method = render_method.lower()
        assert self.render_method in ['3d_gs', '3d_gs_my', '2d_gs', '2d_gs_my', '2d_gs_fast']
        if self.render_method == '3d_gs':
            from fast_2d_gs.renderer.gaussian_render_origin import render_gs_official
            self.gs_rasterizer = render_gs_official
        elif self.render_method == '3d_gs_my':
            from fast_2d_gs.renderer.gaussian_render import render
            self.gs_rasterizer = render
        elif self.render_method == '2d_gs':
            from fast_2d_gs.renderer.gs_2d_render_origin import render_2d_gs_offical
            self.gs_rasterizer = render_2d_gs_offical
        elif self.render_method == '2d_gs_my':
            from fast_2d_gs.renderer.gs_2d_render import GS_2D_render
            self.gs_rasterizer = GS_2D_render
        elif self.render_method == '2d_gs_fast':
            from fast_2d_gs.renderer.gs_2d_fast_render import GS_2D_fast_render
            self.gs_rasterizer = GS_2D_fast_render
        else:
            raise ValueError(f"{self.render_method = } is not supported.")
        # train schedule
        self.stages = {}
        self.train_schedule = []
        step = 0
        for stage in ['static', 'init_fix', 'init', 'sp_fix', 'sp', 'sk_init', 'sk_fix', 'sk']:
            steps = 0 if train_schedule is None else train_schedule.get(stage, 0)
            self.stages[stage] = (step, step + steps, steps)  # start, end, len
            self.train_schedule.append((step, stage))
            step += steps
        self._R_dim = {'lie': 3, 'quaternion': 4}[which_rotation]
        if which_rotation == 'lie':
            self.to_SO3 = SO3.exp
        else:
            self.to_SO3 = SO3.InitFromVec
        self.num_knn = num_knn
        self.num_frames = 0
        self.canonical_time_id = canonical_time_id
        self.canonical_replace_steps = canonical_replace_steps
        self.test_time_interpolate = test_time_interpolate
        self.hyper_dim = hyper_dim

        self.register_buffer('train_db_times', torch.empty(self.num_frames))
        self.register_buffer('train_db_Tv2c', torch.empty(self.num_frames, 4, 4))
        self.register_buffer('train_db_Tw2v', torch.empty(self.num_frames, 4, 4))
        if self.hyper_dim > 0:
            self.hyper_feature = nn.Parameter(torch.empty(0, self.hyper_dim))
            self.param_names_map['hyper_feature'] = 'hyper'
        else:
            self.hyper_feature = None

        self.gs_knn_update_interval = gs_knn_update_interval
        self.gs_knn_num = gs_knn_num
        self.register_buffer('gs_knn_index', torch.empty(0, self.gs_knn_num, dtype=torch.long), persistent=False)
        self.register_buffer('gs_knn_dist', torch.empty(0, self.gs_knn_num, dtype=torch.float), persistent=False)
        self._is_gs_knn_updated = False

        self.deform_net = None
        self.deform_net_name = deform_net_name
        if deform_net_name == 'sc_gs':
            self.deform_net = SC_GS_DeformNetwork(**utils.merge_dict(deform_net_cfg))
        elif deform_net_name == 'd3d':
            self.deform_net = DeformationNetwork(**utils.merge_dict(deform_net_cfg))
        elif deform_net_name == 'mlp':
            self.deform_net = MLP_with_skips(**utils.merge_dict(deform_net_cfg))
        else:
            raise ValueError(f"{deform_net_name} not supported")
        # init stage
        self.init_only_dxyz = init_only_dxyz
        self.init_sp_step = init_sp_step
        self.init_num_times = init_num_times
        if use_canonical_net and self.canonical_time_id >= 0:
            self.canonical_net = self.deform_net.__class__(**utils.merge_dict(deform_net_cfg))
        else:
            self.canonical_net = None
        # sp stage
        self.smooth_time_init = smooth_time_init
        self.smooth_time_steps = smooth_time_steps

        self.sp_prune_threshold = sp_prune_threshold
        self.sp_split_threshold = sp_split_threshold
        self.sp_merge_threshold = sp_merge_threshold

        self.sp_points = nn.Parameter(torch.randn(num_superpoints, 3))
        if self.hyper_dim > 0:
            self.sp_hyper_feature = nn.Parameter(torch.zeros(num_superpoints, self.hyper_dim))
        else:
            self.sp_hyper_feature = None

        assert LBS_method in ['W', 'dist', 'kernel', 'weighted_kernel']
        self.LBS_method = LBS_method
        self._sp_radius: Optional[Tensor] = None
        self._sp_weight: Optional[Tensor] = None
        self.sp_W: Optional[Tensor] = None
        if self.LBS_method == 'W':
            self.sp_W = nn.Parameter(torch.zeros(0, self.num_superpoints))
            self.param_names_map['sp_W'] = 'sp_W'
        if self.LBS_method == 'weighted_kernel' or self.LBS_method == 'kernel':
            self._sp_radius = nn.Parameter(torch.randn(num_superpoints))
        if self.LBS_method == 'weighted_kernel':
            self._sp_weight = nn.Parameter(torch.zeros(num_superpoints))
        assert warp_method in ['largest', 'LBS', 'LBS_c']
        self.warp_method = warp_method
        if self.warp_method == 'largest':
            self.register_buffer('p2sp', torch.empty(0, dtype=torch.int32))
        else:
            self.p2sp = None

        self.register_buffer('sp_is_init', torch.tensor(False))
        self.register_buffer('sp_weights', torch.empty(0, self.num_knn))  # [N, K]
        self.register_buffer('sp_knn', torch.empty(0, self.num_knn, dtype=torch.long))  # [N, K]
        self.cache_dim = (14 if deform_net_cfg.get('sep_rot', False) else 10) - ('2d_gs' in self.render_method)
        self.register_buffer('sp_cache', torch.empty(self.num_frames, self.num_superpoints, self.cache_dim))
        self.node_max_num_ratio_during_init = node_max_num_ratio_during_init

        assert apply_rotation in ['add', 'qmul']
        self.apply_rotate = apply_rotation

        # Cached nn_weight to speed up
        self.cached_nn_weight = False
        self.nn_weight, self.nn_dist, self.nn_idxs = None, None, None

        self.reset_parameters()
        # other
        self.lr_feature_scale = lr_feature_scale
        self.lr_deform_scale = lr_deform_scale
        self.lr_deform_max_steps = lr_deform_max_steps
        self.lr_scheduler = {}
        self.lr_spatial_scale_fix = lr_spatial_scale_fix
        self.time_interval = 0.05
        self.loss_arap_start_step = loss_arap_start_step
        self.loss_flow_decompose_motion = loss_flow_decompose_motion
        assert init_sp_from in ['inputs', 'sp', 'before']
        self.init_sp_from = init_sp_from
        self.init_sp_temp = []
        self.init_sp_same_points = init_sp_same_points
        assert init_same_scale in ['none', 'single', 'all']
        self.init_same_scale = init_same_scale
        # MiniSplatting2
        self.imp_metric = imp_metric
        self.sampling_factor = sampling_factor
        self.depth_reinit_steps = depth_reinit_steps
        self.num_depth_factor = num_depth_factor
        self.register_buffer('_culling', torch.zeros(0, self.num_frames, dtype=torch.bool), persistent=False)
        self.register_buffer('factor_culling', torch.ones(0, 1))
        self.register_buffer('mask_blur', torch.ones(0))

    def reset_parameters(self):
        pass

    @property
    def kernel_radius(self):
        return torch.exp(self._sp_radius)

    @property
    def kernel_weight(self):
        return torch.sigmoid(self._sp_weight)

    @property
    def num_superpoints(self):
        return self.sp_points.shape[0]

    @property
    def get_scaling(self):
        scales = self._scaling
        if 0 < self._step < self.stages['sp_fix'][0]:
            if self.init_same_scale == 'single':
                scales = scales.mean(dim=1, keepdim=True).expand_as(scales)
            elif self.init_same_scale == 'all':
                scales = scales.mean(dim=(0, 1), keepdim=True).expand_as(scales)
        return self.scaling_activation(scales)

    @property
    def device(self):
        return self._xyz.device

    def set_from_dataset(self, dataset):
        super().set_from_dataset(dataset)
        self.num_frames = dataset.num_frames  # the number of frames
        M = self.num_superpoints
        self.train_db_times = dataset.times  # [dataset.camera_ids == dataset.camera_ids[0]]
        self.train_db_Tv2c = dataset.cameras.Tv2c
        self.train_db_Tw2v = dataset.cameras.Tw2v
        print(utils.show_shape(dataset.times, dataset.images, dataset.num_frames))
        assert self.num_frames == len(self.train_db_times)
        assert self.canonical_time_id < self.num_frames
        self.sp_cache = torch.zeros(self.num_frames, M, self.cache_dim)
        self.time_interval = 1. / dataset.num_frames

    def get_params(self, cfg):
        print(f'{self.lr_spatial_scale=}')
        # self.lr_spatial_scale = 5.
        params_groups = super().get_params(cfg)
        lr = self.lr_deform_scale * cfg.lr * self.lr_spatial_scale * self.lr_position_init
        if self.canonical_net is not None:
            params_groups.append({'params': list(self.canonical_net.parameters()), 'lr': lr, 'name': 'canonical'})
        params_groups.append({'params': list(self.deform_net.parameters()), 'lr': lr, 'name': 'deform'})
        params_groups.append({'params': [self.sp_points], 'lr': lr, 'name': 'sp_points'})
        if self._sp_radius is not None:
            params_groups.append({'params': [self._sp_radius], 'lr': lr, 'name': 'sp_radius'})
        if self._sp_weight is not None:
            params_groups.append({'params': [self._sp_weight], 'lr': lr, 'name': 'sp_weight'})
        if self.sp_W is not None:
            params_groups.append({'params': [self.sp_W], 'lr': lr, 'name': 'sp_W'})
        if self.hyper_dim > 0:
            lr_f = cfg.lr * self.lr_feature_scale
            params_groups.extend([
                {'params': [self.hyper_feature], 'lr': lr_f, 'name': 'hyper', 'fix': True},
                {'params': [self.sp_hyper_feature], 'lr': lr_f, 'name': 'sp_hyper', 'fix': True}
            ])

        self.lr_scheduler['deform'] = get_expon_lr_func(
            lr, cfg.lr * self.lr_position_final * self.lr_deform_scale, 0,
            self.lr_position_delay_mult, self.lr_deform_max_steps
        )
        # assert sum(len(g['params']) for g in params_groups) == len(list(self.parameters()))
        return params_groups

    def update_learning_rate(self, optimizer=None, *args, **kwargs):
        if optimizer is None:
            optimizer = self._task.optimizer
        if self._step <= self.stages['sp_fix'][0]:
            step = self._step
        elif self._step <= self.stages['sp'][1]:
            step = (self._step - self.stages['sp_fix'][0])
        else:
            step = (self._step - self.stages['sk_init'][0])
        for group in optimizer.param_groups:
            if group.get('name', None) in ['deform', 'canonical']:
                group['lr'] = self.lr_scheduler['deform'](step)
            elif group.get('name', None) == 'xyz':
                group['lr'] = self.lr_scheduler['xyz'](step)
            # elif group.get('name', None) in [ 'joints', 'global_tr' ]:
            #     group['lr'] = 0
        return

    def create_from_pcd(self, pcd: BasicPointCloud, lr_spatial_scale: float = None):
        if self.init_sp_from == 'inputs':
            self.init_sp_temp = [pcd]
        if self.lr_spatial_scale_fix > 0:
            self.lr_spatial_scale = self.lr_spatial_scale_fix
        else:
            self.lr_spatial_scale = self.cameras_extent if lr_spatial_scale is None else lr_spatial_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        logging.info(f"Number of points at initialisation: {fused_point_cloud.shape[0]} ")

        distCUDA2 = get_C_function('simple_knn')
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3 if '3d_gs' in self.render_method else 2)
        if self.use_so3:
            rots = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
        else:
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, -1] = 1

        opacities = self.opacity_activation_inverse(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D.data = torch.zeros((self.points.shape[0]), device=self._xyz.device)

        N = len(self._xyz)
        if self.hyper_dim > 0:
            self.hyper_feature = nn.Parameter(torch.full([N, self.hyper_dim], -1e-2, device=self.device))
        if self.sp_W is not None:
            self.sp_W = nn.Parameter(torch.ones([N, self.num_superpoints]))
        if self.p2sp is not None:
            self.p2sp = self.p2sp.new_zeros(N)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs):
        N = state_dict['_xyz'].shape[0]
        M = state_dict['sp_points'].shape[0]
        self.sp_points = nn.Parameter(torch.randn(M, 3))
        if self._sp_radius is not None:
            self._sp_radius = nn.Parameter(torch.randn(M))
        if self._sp_weight is not None:
            self._sp_weight = nn.Parameter(torch.zeros(M))
        if self.sp_W is not None:
            self.sp_W = nn.Parameter(torch.empty((N, M)))
        # if self.p2sp is not None:
        #     self.p2sp = self.p2sp.new_zeros(N)
        for key in ['sp_cache', 'sp_weights', 'sp_knn', 'p2sp', 'sp_hyper_feature', 'hyper_feature',
                    'train_db_times', 'train_db_Tw2v', 'train_db_Tv2c',
                    'factor_culling', 'mask_blur']:
            if key in state_dict:
                if isinstance(getattr(self, key), nn.Parameter):
                    setattr(self, key, nn.Parameter(state_dict[key]))
                else:
                    setattr(self, key, state_dict[key])
        super().load_state_dict(state_dict, strict, **kwargs)

    def training_setup(self):
        super().training_setup()
        self._culling = self._culling.new_zeros((self._xyz.shape[0], self.num_frames))
        self.factor_culling = self.factor_culling.new_ones((self._xyz.shape[0], 1))
        self.mask_blur = self.mask_blur.new_zeros((self._xyz.shape[0]))

    @torch.no_grad()
    def init_superpoints(self, force=False, use_hyper=True, points=None, return_sp=False):
        if not self.training:
            return
        if self.sp_is_init and not force:
            return
        times = torch.linspace(0, 1, self.init_num_times, device=self.device)
        points = self.points if points is None else points
        if use_hyper:
            trans_samp = [self.deform_net(points, times[i])[0] for i in range(self.init_num_times)]
            trans_samp = torch.stack(trans_samp, dim=1)
            hyper_pcl = (trans_samp + points[:, None]).reshape(points.shape[0], -1)
        else:
            hyper_pcl = None
        if return_sp or self.canonical_net is None:
            init_pcl = points
        else:
            init_pcl = self.deform_net(points, self.train_db_times[self.canonical_time_id])[0] + points
            self.deform_net.load_state_dict(self.canonical_net.state_dict())
            logging.info('[red]Apply canonical_net')
        # Initialize Superpoints
        pcl_to_samp = init_pcl if hyper_pcl is None else hyper_pcl
        init_nodes_idx = FurthestSampling(pcl_to_samp.detach()[None].cuda(), self.num_superpoints)[0].to(self.device)
        self.sp_points.data = init_pcl[init_nodes_idx].clone()
        if self.hyper_dim > 0:
            nn.init.constant_(self.sp_hyper_feature, 1e-2)
        scene_range = init_pcl.max() - init_pcl.min()
        if self._sp_radius is not None:
            self._sp_radius.data = torch.log(.1 * scene_range + 1e-7) * scene_range.new_ones([self.num_superpoints])
        if self._sp_weight is not None:
            self._sp_weight.data = torch.zeros_like(self._sp_radius)
        if return_sp:
            return self.sp_points[..., :3]
        if self.init_sp_same_points:
            opt = self._task.optimizer
            optimizable_tensors = self.change_optimizer(
                opt, {opt_name: getattr(self, name)[init_nodes_idx] for name, opt_name in self.param_names_map.items()},
                op='replace'
            )
            for param_name, opt_name in self.param_names_map.items():
                if opt_name in optimizable_tensors:
                    setattr(self, param_name, optimizable_tensors[opt_name])
            opt.zero_grad(set_to_none=True)
            self._xyz.data.copy_(self.sp_points[..., :3])
        self.active_sh_degree = 0
        P = len(self.points)
        self.xyz_gradient_accum = self.xyz_gradient_accum.new_zeros((P, 1))
        self.denom = self.denom.new_zeros((P, 1))
        self.max_radii2D.data = self.max_radii2D.new_zeros((P))
        self._culling = self._culling.new_zeros((P, self.num_frames))
        self.factor_culling = self.factor_culling.new_zeros((P, 1))
        self.mask_blur = self.mask_blur.new_zeros((P))

        self.sp_is_init = self.sp_is_init.new_tensor(True)
        logging.info(f'[red]Control node initialized with {self.sp_points.shape[0]} from {init_pcl.shape[0]} points.')
        return init_nodes_idx

    def smooth_time(self, t: Tensor, lr_final=1e-15, lr_delay_steps=0.01, lr_delay_mult=1.0):
        if self._step < 0 or self.smooth_time_steps <= 0 or self._step > self.smooth_time_steps:
            return t
        lr_init = self.smooth_time_init
        max_steps = self.smooth_time_steps
        step = self._step  # if self._step <= self.stages['sp_fix'][0] else self._step - self.stages['sp_fix'][0]
        if lr_init == 0.0 and lr_final == 0.0:
            return t
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        _t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - _t) + lr_final * _t
        t_noise = torch.randn_like(t) * self.time_interval * delay_rate * log_lerp
        return t + t_noise

    def init_stage(self, x, t, use_canonical_net=False):
        if use_canonical_net:
            d_xyz, d_rot, d_scale = self.canonical_net(x.detach(), t)[:3]
        else:
            d_xyz, d_rot, d_scale = self.deform_net(x.detach(), t)[:3]
        if self.init_only_dxyz:
            return d_xyz, d_xyz.new_tensor(0), d_xyz.new_tensor(0)
        else:
            return d_xyz, d_rot, d_scale

    def calc_LBS_weight(self, points: Tensor, sp_points: Tensor, feature=None, sp_feature=None, K=None, temperature=1.):
        ## calculate knn
        if feature is not None and sp_feature is not None:
            points = torch.cat([points.detach(), feature], dim=-1)
            sp_points = torch.cat([sp_points.detach(), sp_feature], dim=-1)
        K = self.num_knn if K is None else K
        nn_dist, indices, _ = knn_points(points[None], sp_points[None], None, None, K=K)  # N, K
        nn_dist, indices = nn_dist[0], indices[0]  # N, K
        ## calculate weights
        if self._sp_radius is not None:
            radius = self.kernel_radius[indices]  # N, K
            weights = torch.exp(-nn_dist / (2 * radius ** 2))  # N, K
            if self._sp_weight is not None:
                weights = weights * self.kernel_weight[indices]
            weights = weights + 1e-7
            weights = weights / weights.sum(dim=-1, keepdim=True)  # N, K
        elif self.sp_W is not None:
            weights = torch.gather(self.sp_W, dim=1, index=indices).softmax(dim=-1)
        else:  # LBS_method = dist
            weights = torch.softmax(-nn_dist / temperature, dim=-1)

        # self.sp_weights = weights.detach()
        # self.sp_knn = indices.detach()
        return weights, indices

    def warp(self, points, sp_points, sp_t, sp_r, sp_s, sp_rot, weights, indices, method='LBS'):
        """

        Args:
            points (Tensor): the position of Gaussians, shape: [P, 3]
            sp_points (Tensor): the postion of  superpoints, shape: [M, 3]
            sp_t (Tensor): the translation or transform maxtrix predicted by superpoints
            sp_r (Tensor | None): the rotation predicted by superpoints
            sp_s (Tensor | None): the residual scale of Gaussians predicted by superpoints
            sp_rot (Tensor | None): the residual direction of Gaussians predicted by superpoints
            weights (Tensor): LBS weights, shape: [P, K]
            indices (Tensor): LBS indices, ie the index of KNN, shape: [P, K]
            method (str): LBS_c, LBS, largest

        Returns:
            (Tensor, Optional[Tensor], Optional[Tensor], Tensor)

            - d_points: shape: [P, 3]
            - d_rotation: shape: [P, 4]
            - d_scale: shape: [P, 3]
            - sp_t: The transformation matrix of superpoints, represented by quaternion, shape: [M, 4]
        """
        if self.apply_rotate == 'mul' or sp_rot is not None:
            sp_rot = sp_r if sp_rot is None else sp_rot
            sp_rot = ops_3d.quaternion.normalize(sp_rot + sp_rot.new_tensor([0, 0, 0, 1.]))
            if method == 'LBS_c':
                sp_t = sp_t + sp_points + SO3.InitFromVec(sp_rot).act(-sp_points)
            spT = SE3.InitFromVec(torch.cat([sp_t, sp_rot], dim=-1))

            if isinstance(spT, Tensor):
                if method == 'LBS_c' or method == 'LBS':
                    d_points = (ops_3d.apply(points[:, None], spT[indices]) * weights[..., None]).sum(dim=1) - points
                else:
                    d_points = ops_3d.apply(points, spT[self.p2sp]) - points
            else:  # SE3
                if method == 'LBS_c' or method == 'LBS':
                    d_points = (spT[indices].act(points[:, None]) * weights[..., None]).sum(dim=1) - points
                else:
                    d_points = spT[self.p2sp].act(points) - points
        else:
            spT = None
            d_points = (sp_t[indices] * weights[..., None]).sum(dim=1)
        if self.apply_rotate == 'add':
            d_rotation = (sp_r[indices] * weights[..., None]).sum(dim=1)
        else:
            d_rotation = None
            raise NotImplementedError  # TODO: 基于四元数的加权平均
        d_scales = (sp_s[indices] * weights[..., None]).sum(dim=1) if sp_s is not None else None
        if isinstance(spT, SE3):
            spT = spT.vec()
        return d_points, d_rotation, d_scales, spT

    def sp_stage(
            self, points: Tensor, t, time_id: int = None, use_canonical_net=False, sp_points: Tensor = None, **kwargs
    ):
        points = points.detach()
        sp_points = self.sp_points if sp_points is None else sp_points
        # Calculate nn weights: [N, K]
        if use_canonical_net:
            outs = self.canonical_net(sp_points.detach(), t)
            weights, indices = self.sp_weights, self.sp_knn
        else:
            weights, indices = self.calc_LBS_weight(points, sp_points, self.hyper_feature, self.sp_hyper_feature)
            outs = self.deform_net(sp_points.detach(), t)

        d_xyz = outs[0]
        d_rot = outs[1] if len(outs) > 1 else 0.
        d_scale = outs[2] if len(outs) > 2 else 0.  # [:, :self._scaling.shape[1]]
        g_rot = outs[3] if len(outs) > 3 else None

        if self.warp_method == 'largest' and self.training:
            self.p2sp = torch.gather(indices, -1, weights.argmax(dim=-1, keepdim=True))[:, 0]
        d_points, d_rotation, d_scales, spT = self.warp(
            points, sp_points, d_xyz, d_rot, d_scale, g_rot,
            weights, indices, self.warp_method
        )
        return d_points, d_rotation, d_scales, spT, g_rot, d_scale, weights, indices

    def get_now_stage(self, stage=None, now_step: int = None):
        now_step = self._step if now_step is None else now_step
        if stage is None:
            for stage, (start, end, num) in self.stages.items():
                if start < now_step <= end:
                    return stage
        return stage

    def forward(self, t: Tensor = None, campos: Tensor = None, stage=None, time_id: int = None, **kwargs):
        stage = self.get_now_stage(stage)
        outputs = {'opacity': self.opacity_activation(self._opacity)}
        t = self.smooth_time(t.view(-1, 1))

        points, scales, rotations = self._xyz, self._scaling, self._rotation
        if stage == 'static':
            d_xyz, d_rotation, d_scaing = 0, 0, 0
        elif stage == 'init' or stage == 'init_fix':
            d_xyz, d_rotation, d_scaing = self.init_stage(points, t)
            if self.init_same_scale == 'single':
                scales = scales.mean(dim=1, keepdim=True).expand_as(scales)
            elif self.init_same_scale == 'all':
                scales = scales.mean(dim=(0, 1), keepdim=True).expand_as(scales)
            if stage == 'init_fix':
                d_xyz, d_rotation, d_scaing = d_xyz.detach(), d_rotation.detach(), d_scaing.detach()
        else:  # stage == 'sp_fix' or stage == 'sp':
            d_xyz, d_rotation, d_scaing, spT, sp_d_rot, sp_d_scale, knn_w, knn_i = self.sp_stage(points, t, time_id)
            if stage == 'sp_fix':
                d_xyz, d_rotation, d_scaing = d_xyz.detach(), d_rotation.detach(), d_scaing.detach()
            outputs.update(_spT=spT, _knn_w=knn_w, _knn_i=knn_i, _sp_rot=sp_d_rot, _sp_scale=sp_d_scale)
        outputs['points'] = points + d_xyz
        # if self.convert_SHs_python and campos is not None:
        #     outputs['colors'] = self.get_colors(sh_features, points, campos)
        # else:
        outputs['sh_features'] = self._features_dc
        outputs['sh_features_rest'] = self._features_rest

        # if self.compute_cov3D:
        #    outputs['covariance'] = self.covariance_activation(scales * kwargs.get('scaling_modifier', 1.0), rotations)
        #    assert False
        # else:
        if isinstance(d_scaing, Tensor) and d_scaing.numel() > 1:
            d_scaing = d_scaing[:, :scales.shape[1]]
        outputs['scales'] = self.scaling_activation(scales) + d_scaing
        if self.apply_rotate == 'add':
            outputs['rotations'] = self.rotation_activation(rotations + d_rotation)
        else:
            outputs['rotations'] = ops_3d.quaternion.mul(
                ops_3d.quaternion.normalize(d_rotation), self.rotation_activation(rotations))
        return outputs

    def render(
            self,
            *args,
            t: Tensor = None,
            info,
            background: Tensor = None,
            time_id=None,
            scale_modifier=1.0,
            stage=None,
            render_type='imp',
            **kwargs
    ):
        stage = self.get_now_stage(stage)
        Tw2v = info['Tw2v'].view(-1, 4, 4)
        Tv2c = info['Tv2c'].view(-1, 4, 4)
        Tv2s = info['Tv2s'].view(-1, 3, 3)
        campos = info['campos'].view(-1, 3)
        if 'focals' in info:
            focals = utils.to_tensor(info['focals']).view(-1, 2).expand(Tw2v.shape[0], 2)
            FoV = None
        else:
            FoV = utils.to_tensor(info['FoV']).view(-1, 2).expand(Tw2v.shape[0], 2)
            focals = None
        size = utils.to_tensor(info['size']).view(-1, 2).expand(Tw2v.shape[0], 2)
        if t is not None:
            t = t.view(-1)
        if info['Tw2v'].ndim == 2:
            if background is not None and background.ndim > 1:
                background = background.unsqueeze(0)
        culling = self._culling[:, info['index']].view(-1, Tw2v.shape[0]) if self.training else None
        outputs = {}
        sh_degree = self.active_sh_degree  # if self.training else self.max_sh_degree
        for b in range(Tw2v.shape[0]):
            if self.use_official_gaussians_render:
                if background is None:
                    bg = Tw2v.new_zeros(3)
                else:
                    bg = background[b, ..., :3] if background.ndim > 1 else background
                    if bg.numel() <= 3:
                        bg = bg.view(-1).expand(3).contiguous()
                    else:
                        bg = bg.view(-1, 3).mean(0)
            else:
                bg = (background[b] if background.shape[0] > 1 else background[0]) if background is not None else None

            net_out = self(t=t, campos=campos, stage=stage, time_id=time_id)
            if 'hook' in kwargs:
                net_out = kwargs['hook'](net_out)
            if 'my' not in self.render_method:
                net_out['bg'] = bg
            outputs_b = {}
            for k in list(net_out.keys()):  # type: str
                if k.startswith('_'):
                    outputs_b[k] = net_out.pop(k)
            P = self._xyz.shape[0]
            net_out['accum_max_count'] = torch.zeros(P, device=self.device, dtype=torch.int32)
            if render_type == 'simp':
                net_out['accum_weights_p'] = Tw2v.new_zeros(P)
                net_out['accum_weights_count'] = torch.zeros_like(net_out['accum_max_count'])
            cam_info = dict(Tw2v=Tw2v[b], Tv2c=Tv2c[b], campos=campos[b], focals=None if focals is None else focals[b],
                            FoV=None if FoV is None else FoV[b], )
            if '2d' in self.render_method:
                cam_info['Tv2s'] = Tv2s[b]
            outputs_b.update(self.gs_rasterizer(
                **net_out, **cam_info, sh_degree=sh_degree, size=size[b],
                culling=culling[:, b].contiguous() if self.training else None,
            ))
            outputs_b['area_max'] = net_out['accum_max_count']
            if render_type == 'simp':
                outputs_b['accum_weights'] = net_out['accum_weights_p']
                outputs_b['area_proj'] = net_out['accum_weights_count'].float()
            images = outputs_b['images']
            if images.shape[-1] != 3:
                images = torch.permute(images, (1, 2, 0))
            if 'my' in self.render_method:
                images = images + (1 - outputs_b['opacity'][..., None]) * bg.squeeze(0)
            outputs_b['images'] = images
            outputs_b['points'] = net_out['points']

            # print(utils.show_shape(outputs_b))
            # max_id = outputs_b['buffer'][6][1]
            # label, max_id = torch.unique(max_id, return_inverse=True)
            # print(utils.show_shape(label, max_id))
            # img = utils.color_labels(max_id.cpu())
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()
            # utils.save_image(self._task.output.joinpath('max_id.jpg'), img)
            # exit()
            if b == 0:
                outputs = {k: [v] for k, v in outputs_b.items() if v is not None}
            else:
                for k, v in outputs_b.items():
                    if v is not None:
                        outputs[k].append(v)
        outputs = {k: torch.stack(v, dim=0) if k != 'viewspace_points' and isinstance(v[0], Tensor) else v for k, v in
                   outputs.items()}  # noqa
        outputs['stage'] = stage
        return outputs

    def loss_weight_sparsity(self, weight: Tensor, eps=1e-7):
        return -(weight * torch.log(weight + eps) + (1 - weight) * torch.log(1 - weight + eps)).mean()

    def update_gs_knn(self, force=False):
        if self._is_gs_knn_updated:
            return
        self._is_gs_knn_updated = True
        if not (force or (self.gs_knn_index.shape[0] != self.points.shape[0]) or
                utils.check_interval(self._step, *self.gs_knn_update_interval, force_end=False)):
            return
        from pykdtree.kdtree import KDTree
        points = self.points.detach().cpu().numpy()
        kdtree = KDTree(points)
        knn_dist, knn_index = kdtree.query(points, k=self.gs_knn_num + 1)
        self.gs_knn_index = torch.from_numpy(knn_index.astype(np.int32)).to(self.gs_knn_index)
        self.gs_knn_dist = torch.from_numpy(knn_dist).to(self.gs_knn_dist)
        logging.info('update guassian knn')

    def loss_weight_smooth(self, weight: Tensor):
        self.update_gs_knn()
        return (weight[:, None] - weight[self.gs_knn_index]).abs().mean()

    def loss_points_arap(self, points_t: Tensor):
        # self.update_gs_knn()
        points_c = self.points
        # indices = self.gs_knn_index[:, 1:]
        nn_dist, indices, _ = knn_points(points_t[None], points_t[None], K=self.gs_knn_num + 1)
        indices = indices[0, :, 1:]
        dict_c = (points_c[:, None] - points_c[indices]).square().sum(dim=-1)
        dict_t = (points_t[:, None] - points_t[indices]).square().sum(dim=-1)
        return (dict_c - dict_t).abs().mean()

    def loss_sp_arap(self, sp_se3: SE3):
        sp_points_c = self.sp_points[..., :3]
        sp_points_t = sp_se3.act(sp_points_c)
        with torch.no_grad():
            sp_dist = torch.cdist(sp_points_c, sp_points_c)
            k_dist, knn = torch.topk(sp_dist, dim=1, k=min(self.num_superpoints, self.sk_knn_num + 1), largest=False)
            knn = knn[:, 1:]
        # loss = F.mse_loss(sp_tr[:, None].repeat(1, num_knn, 1), sp_tr[knn])
        loss = (sp_se3[:, None].inv() * sp_se3[knn]).log().norm(dim=-1).mean()
        dist_c = (sp_points_c[:, None] - sp_points_c[knn]).square().sum(dim=-1)
        dist_t = (sp_points_t[:, None] - sp_points_t[knn]).square().sum(dim=-1)
        arap_ct_loss = (dist_c - dist_t).abs().mean()
        return loss, arap_ct_loss

    def loss_arap(self, t=None, delta_t=0.05, t_samp_num=2, points: Tensor = None):
        if points is None:
            points = self.sp_points
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t_samp = torch.rand(t_samp_num).cuda() * delta_t + t - .5 * delta_t
        t_samp = t_samp[None, :, None].expand(points.shape[0], t_samp_num, 1)  # M, T, 1
        x = points[:, None, :].repeat(1, t_samp_num, 1).detach()
        dx = self.deform_net(x=x.reshape(-1, 3), t=t_samp.reshape(-1, 1))[0]
        nodes_t = points[:, None, :3].detach() + dx.reshape(-1, t_samp_num, 3)  # M, T, 3
        hyper_nodes = nodes_t[:, 0]  # M, 3
        ii, jj, knn, weight = cal_connectivity_from_points(hyper_nodes, K=10)  # connectivity of control nodes
        error = cal_arap_error(nodes_t.permute(1, 0, 2), ii, jj, knn)
        return error

    def loss_elastic(self, t=None, delta_t=0.005, K=2, t_samp_num=8, points: Tensor = None, hyper: Tensor = None):
        if points is None:
            points = self.sp_points
            hyper = self.sp_hyper_feature
        num_points = points.shape[0]
        # Calculate nodes translate
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t_samp = torch.rand(t_samp_num).cuda() * delta_t + t - .5 * delta_t
        t_samp = t_samp[None, :, None].expand(num_points, t_samp_num, 1)
        x = points[:, None, :].repeat(1, t_samp_num, 1).detach()
        node_trans = self.deform_net(x=x.reshape(-1, 3), t=t_samp.reshape(-1, 1))[0]
        nodes_t = points[:, None, :3].detach() + node_trans.reshape(-1, t_samp_num, 3)  # M, T, 3

        # Calculate weights of nodes NN
        nn_weight, nn_idx = self.calc_LBS_weight(points, points, hyper, hyper, K=K + 1)
        nn_weight, nn_idx = nn_weight[:, 1:], nn_idx[:, 1:]  # M, K

        # Calculate edge deform loss
        edge_t = (nodes_t[nn_idx] - nodes_t[:, None]).norm(dim=-1)  # M, K, T
        edge_t_var = edge_t.var(dim=2)  # M, K
        edge_t_var = edge_t_var / (edge_t_var.detach() + 1e-5)
        arap_loss = (edge_t_var * nn_weight).sum(dim=1).mean()
        return arap_loss

    def loss_acc(self, t=None, delta_t=.005, points: Tensor = None):
        if points is None:
            points = self.sp_points
        # Calculate nodes translate
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t = torch.stack([t - delta_t, t, t + delta_t])
        t = t[None, :, None].expand(points.shape[0], 3, 1)
        x = points[:, None, :].repeat(1, 3, 1).detach()
        node_trans = self.deform_net(x=x.reshape(-1, 3), t=t.reshape(-1, 1))[0]
        nodes_t = points[:, None, :3].detach() + node_trans.reshape(-1, 3, 3)  # M, 3, 3
        acc = (nodes_t[:, 0] + nodes_t[:, 2] - 2 * nodes_t[:, 1]).norm(dim=-1)  # M
        acc = acc / (acc.detach() + 1e-5)
        acc_loss = acc.mean()
        return acc_loss

    def loss_reconstruct(self, losses, outputs):
        weights, indices = outputs['_knn_w'][0], outputs['_knn_i'][0]
        if self.loss_funcs.w('re_tr'):
            re_sp_tr = get_superpoint_features(outputs['_tr'], indices, weights, self.num_superpoints)
            losses['re_tr'] = self.loss_funcs('re_tr', F.mse_loss(re_sp_tr, outputs['_sp_tr']))
        if self.loss_funcs.w('re_pos') > 0:
            sp_se3 = SE3.InitFromVec(outputs['_spT'][0])
            re_sp = get_superpoint_features(outputs['points'][0], indices, weights, self.num_superpoints)
            sp = sp_se3.act(self.sp_points)
            losses['re_pos'] = self.loss_funcs('re_pos', F.mse_loss(sp, re_sp))
        return losses

    def loss_canonical_net(self, points, t, stage='init'):
        if stage == 'sp' and len(self.canonical_replace_steps) == 0:
            return 0
        tc = self.train_db_times[self.canonical_time_id]
        if stage == 'init':
            with torch.no_grad():
                points_c = self.init_stage(self._xyz, tc)[0] + self._xyz
            points_t = self.init_stage(points_c, t, use_canonical_net=True)[0] + points_c
        else:
            with torch.no_grad():
                d_xyz, d_rotation, d_scaing, spT, sp_d_rot, sp_d_scale, knn_w, knn_i = self.sp_stage(self._xyz, tc)
                points_c = d_xyz + self._xyz
                sp_points_c = SE3.InitFromVec(spT).act(self.sp_points)
            points_t = self.sp_stage(points_c, t, sp_points=sp_points_c, use_canonical_net=True)[0] + points_c
        return F.mse_loss(points_t, points.detach())

    def loss_flow(self, inputs, outputs, targets, info):
        # get tensors
        index_t1 = info['index']
        if not isinstance(index_t1, int):
            index_t1 = int(index_t1)
        if 'last_index' not in info and index_t1 == self.num_frames:
            return outputs['images'].new_tensor(0.)
        Tw2v_t1 = info['Tw2v'].view(4, 4)
        Tv2s_t1 = info['Tv2s'].view(3, 3)
        Tv2c_t1 = info['Tv2c'].view(4, 4)
        depths = outputs['surf_depth'].squeeze()
        means2D_t1 = outputs['viewspace_points']
        visible = outputs['visibility_filter']
        buffer = outputs['buffer'][0]
        t1 = self.train_db_times[index_t1]
        FoV = info['FoV'].view(2)
        tan_fov = torch.tan(0.5 * FoV)
        # cov2D_t1_inv = buffer.conic_opacity

        W, H = buffer.W, buffer.H

        index_t2 = info['last_index'] if 'last_index' in info else info['index'] + 1
        if not isinstance(index_t2, int):
            index_t2 = int(index_t2)
        t2 = self.train_db_times[index_t2]
        Tv2c_t2 = self.train_db_Tv2c[index_t2] if self.train_db_Tv2c.ndim > 2 else self.train_db_Tv2c
        Tw2v_t2 = self.train_db_Tw2v[index_t2]
        Tv2s_t2 = info['Tv2s'].view(3, 3)
        # if self.refine_camera:
        #     delta_Tw2v = torch.cat([self.delta_tsl[index_t2], self.delta_rot[index_t2]], dim=-1)
        #     Tw2v_t2 = SE3.exp(delta_Tw2v).matrix() @ Tw2v_t2
        # if 'last_index' in info:
        #     flow = targets['flow_bwd'].squeeze(0)
        #     flow_mask = targets['flow_b_m'].squeeze(0)
        # else:
        flow = targets['flow_fwd' if self.loss_flow_decompose_motion else 'flow_bwd'].squeeze(0)
        # flow_mask = targets['flow_f_m'].squeeze(0)
        if flow.shape[:2] != (H, W):
            H_, W_ = flow.shape[:2]
            flow = F.interpolate(flow[None].permute(0, 3, 1, 2), (H, W), mode='bilinear')[0].permute(1, 2, 0)
            flow = flow * flow.new_tensor([W / W_, H / H_])
            # flow_mask = F.interpolate(flow_mask[None, None].float(), (H, W), mode='nearest')[0, 0]
        # flow = torch.zeros((H, W, 2), device=Tw2v_t1.device)

        # compute motion flow
        if self.loss_flow_decompose_motion:
            pixels = torch.stack(torch.meshgrid(
                torch.arange(W, device=depths.device), torch.arange(H, device=depths.device), indexing='xy'
            ), dim=-1)
            with torch.no_grad():
                points_t1 = ops_3d.pixel2points(depths, Tv2s=Tv2s_t1, Tw2v=Tw2v_t1, pixel=pixels)
                pixels_t1 = ops_3d.point2pixel(points_t1, Tw2v=Tw2v_t2, Tv2c=Tv2c_t2, size=(W, H))[0]
                camera_flow = (pixels_t1 - pixels).squeeze()
                pixel_n = pixels_t1 * pixels_t1.new_tensor([2 / (W - 1), 2 / (H - 1)]) - 1

            net_out_t1 = self(t=t1)
            net_out_t2 = self(t=t2)
            outputs_c2_t2 = self.gs_rasterizer(
                **net_out_t2,  # trans_mat_t2=outputs['buffer'][0].trans_mat,
                Tw2v=Tw2v_t2, Tv2c=Tv2c_t2, campos=Tw2v_t2.inverse()[:3, 3].contiguous(), FoV=FoV, size=(W, H))
            outputs_c2_t1 = self.gs_rasterizer(
                **net_out_t1, trans_mat_t2=outputs_c2_t2['buffer'].trans_mat,
                Tw2v=Tw2v_t2, Tv2c=Tv2c_t2, campos=Tw2v_t2.inverse()[:3, 3].contiguous(), FoV=FoV, size=(W, H))

            gs_flow = outputs_c2_t1['flow']
            gs_flow = gs_flow.permute(2, 0, 1)
            gs_flow = F.grid_sample(gs_flow[None], pixel_n[None], padding_mode="border", align_corners=True)[0]  # 2 H W
            gs_flow = gs_flow.permute(1, 2, 0)
            motion_flow = flow - camera_flow
            # if 'motion_mask' in targets:
            #     motion_flow = motion_flow * (1 - motion_mask)
            # self._debug(index_t1, index_t2, flow,camera_flow + gs_flow)#,outputs['images'], outputs_c2_t2['images'])
        else:
            net_out_t2 = self(t=t2)
            outputs_c2_t2 = self.gs_rasterizer(
                **net_out_t2, trans_mat_t2=outputs['buffer'][0].trans_mat,
                Tw2v=Tw2v_t2, Tv2c=Tv2c_t2, campos=Tw2v_t2.inverse()[:3, 3].contiguous(), FoV=FoV, size=(W, H))
            gs_flow = outputs_c2_t2['flow']
            motion_flow = flow
            # self._debug(index_t2, index_t1, flow, gs_flow)
        ## calculate flow loss
        motion_flow = motion_flow.mul(motion_flow.new_tensor([1. / W, 1. / H])).clamp(-1, 1)
        gs_flow = gs_flow.mul(gs_flow.new_tensor([1. / W, 1. / H])).clamp(-1, 1)
        return F.l1_loss(gs_flow, motion_flow)

    def _debug(self, index_t1: int, index_t2: int, flow1: Tensor, flow2: Tensor, img1=None, img2=None):
        print(f"{index_t1=} -> {index_t2=}")
        db = self._task.train_db
        if img1 is None:
            img1 = db.images[index_t1].contiguous().detach().cpu().numpy()[:, :, ::-1]
        else:
            img1 = utils.as_np_image(img1)[:, :, ::-1] / 255.
        if img2 is None:
            img2 = db.images[index_t2].contiguous().detach().cpu().numpy()[:, :, ::-1]
        else:
            img2 = utils.as_np_image(img2)[:, :, ::-1] / 255.
        H, W, C = img1.shape
        images = np.zeros((H * 2, W * 3, C), dtype=np.float32)
        images[:H, W * 0:W * 1, :] = img1
        images[:H, W * 1:W * 2, :] = img2
        # images[:H, W * 2:W * 3, :] = np.abs(img2 - img1)
        print(utils.show_shape(flow1, flow2))
        flow1 = flow1.squeeze().detach().cpu().numpy()
        flow2 = flow2.squeeze().detach().cpu().numpy()
        flows = np.concatenate([flow1, flow2], axis=1)
        images[H:H * 2, :W * 2, :] = utils.flow_colorize(flows) / 255.
        images[H:H * 2, W * 2:W * 3, :] = np.abs(images[H:H * 2, 0:W] - images[H:H * 2, W * 1:W * 2, :])
        WIN_NAME = 'Debug'
        images_ = images.copy()
        if images_.shape[0] == 3:
            images_ = np.transpose(images, (1, 2, 0))

        def onmouse_pick_points(event, x, y, flags, param: np.ndarray):
            if event == cv2.EVENT_LBUTTONDOWN:
                param[:] = images_
                if x >= W or y >= H:
                    return
                print(f"{x=}, {y=}, dx={flow1[y, x, 0]}, dy={flow1[y, x, 1]}", end=' ')
                cv2.drawMarker(param, (x, y), (0, 255, 0))
                x2, y2 = int(x + W + flow1[y, x, 0]), int(y + flow1[y, x, 1])
                if W <= x2 < W * 2 and 0 <= y2 < H:
                    cv2.drawMarker(param, (x2, y2), (0, 255, 0))
                print(f"dx2 = {flow2[y, x, 0]}, dy2 = {flow2[y, x, 1]}", end=' ')
                x3, y3 = int(x + W + flow2[y, x, 0]), int(y + flow2[y, x, 1])
                if W <= x3 < W * 2 and 0 <= y3 < H:
                    cv2.drawMarker(param, (x3, y3), (255, 0, 0))
                print('')
            return

        cv2.namedWindow(WIN_NAME)
        cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, images)
        while True:
            cv2.imshow(WIN_NAME, images)
            key = cv2.waitKey(30)
            if key == ord('q') or key == 27:
                exit()

    def loss(self, inputs, outputs, targets, info):
        self._is_gs_knn_updated = False
        time_id = inputs['time_id'].item() if 'time_id' in inputs else None
        t = inputs['t'].view(-1)
        stage = outputs['stage']

        losses = {}
        image = outputs['images']
        gt_img = targets['images'][..., :3]
        H, W, C = image.shape[-3:]
        image, gt_img = image.view(1, H, W, C), gt_img.view(1, H, W, C)
        losses['rgb'] = self.loss_funcs('image', image, gt_img)
        losses['ssim'] = self.loss_funcs('ssim', image, gt_img)
        # if stage == 'sp':
        #     losses['elastic'] = self.loss_funcs('elastic', self.loss_elastic, t, self.time_interval)
        #     losses['acc'] = self.loss_funcs('acc', self.loss_acc, t, 3.0 * self.time_interval)
        #     losses['arap'] = self.loss_funcs('arap', self.loss_arap)
        # if stage == 'sp' or stage == 'init':
        #     if self.canonical_net is not None and self._step <= max(self.canonical_replace_steps) + 5:
        #         losses['c_net'] = self.loss_funcs('c_net', self.loss_canonical_net, outputs['points'][0], t, stage)
        # if stage == 'init':
        #     if self.points.shape[0] <= self.num_superpoints:
        #         points, hyper = self.points, self.hyper_feature
        #     else:
        #         index = torch.randperm(self.points.shape[0], device=self.device)[:self.num_superpoints]
        #         points, hyper = self.points[index], self.hyper_feature[index] if self.hyper_dim > 0 else None
        #     losses['elastic'] = self.loss_funcs(
        #         'elastic', self.loss_elastic, t, self.time_interval, points=points, hyper=hyper
        #     )
        #     losses['acc'] = self.loss_funcs('acc', self.loss_acc, t, 3.0 * self.time_interval, points=points)
        #     losses['arap'] = self.loss_funcs('arap', self.loss_arap, points=points)
        #     losses['arap_p'] = self.loss_funcs('p_arap_ct_init', self.loss_points_arap, outputs['points'][0])
        # if stage == 'sp' and self.training:
        #     with torch.no_grad():
        #         cache = torch.cat([outputs[k] for k in ['_spT', '_sp_rot', '_sp_scale'] if k in outputs], dim=-1)
        #         self.sp_cache[time_id] = cache.squeeze(0)
        #     self.loss_reconstruct(losses, outputs)
        # if stage == 'sp' and self._step >= self.loss_arap_start_step >= 0:
        #     if self.loss_funcs.w('sp_arap_t') > 0:
        #         arap_t, arap_ct = self.loss_sp_arap(SE3.InitFromVec(outputs['_spT'][0]))
        #         losses['arap_t'] = self.loss_funcs('sp_arap_t', arap_t)
        #         losses['arap_ct'] = self.loss_funcs('sp_arap_ct', arap_ct)
        # if stage == 'sp':
        #     losses['sparse'] = self.loss_funcs('sparse', self.loss_weight_sparsity, outputs['_knn_w'])
        #     losses['smooth'] = self.loss_funcs('smooth', self.loss_weight_smooth, outputs['_knn_w'][0])
        #
        # if 'normal' in outputs:
        #     rend_normal = outputs['normals']
        #     surf_normal = outputs['surf_normal']
        #     losses['normal'] = self.loss_funcs('normal', lambda: (1 - (rend_normal * surf_normal).sum(dim=0)).mean())
        # if 'distortion' in outputs:
        #     losses['dist'] = self.loss_funcs('dist', lambda: outputs['distortion'].mean())
        # if self.training and self._step >= self.warmup_steps:
        losses['flow'] = self.loss_funcs('flow', self.loss_flow, inputs, outputs, targets, info)
        # self.test_flow_loss(inputs, outputs, targets, info)
        return {k: v for k, v in losses.items() if isinstance(v, Tensor)}

    def change_with_training_progress(self, step=0, num_steps=1, epoch=0, num_epochs=1):
        total_step = epoch * num_steps + step + 1
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if total_step > self.stages['sp_fix'][0] and self.active_sh_degree < self.max_sh_degree and (
                total_step - self.stages['sp_fix'][0]) % 1000 == 0:
            self.active_sh_degree = self.active_sh_degree + 1
            logging.info(f'increase active sh degree to {self.active_sh_degree} at step {total_step}')
        self._step = total_step

    def hook_before_train_step(self):
        if self._step == self.init_sp_step:
            if self.init_sp_from == 'before':
                self.init_sp_temp = [getattr(self, name).clone() for name in
                                     ['_xyz', '_features_dc', '_scaling', '_rotation', '_opacity']]
            self.init_superpoints(True, True)
            # self.reset_optimizer(self._task.optimizer)
        if self.canonical_net is not None and self._step > self.stages['sp_fix'][0] and \
                (self._step in self.canonical_replace_steps):
            with torch.no_grad():
                tc = self.train_db_times[self.canonical_time_id]
                d_xyz, d_rotation, d_scaing, spT, sp_d_rot, sp_d_scale, knn_w, knn_i = self.sp_stage(self._xyz, tc)
                points_c = d_xyz + self._xyz
                sp_points_c = SE3.InitFromVec(spT).act(self.sp_points)
                self._xyz.data = points_c
                self.sp_points.data = sp_points_c
                self.deform_net.load_state_dict(self.canonical_net.state_dict())
                logging.info(f"Replace canonical_net at step {self._step}")

    @torch.no_grad()
    def hook_after_train_step(self):
        if self._step == self.stages['sp_fix'][0] and self.stages['sp'][2] > 0:
            if self.init_sp_same_points:
                self.sp_points.data[:, :3] = self.points
            if self.init_sp_from == 'inputs':  # use input points to initialize GS
                self.create_from_pcd(self.init_sp_temp[0], self.lr_spatial_scale)  # noqa
                self.to(self.sp_points.device)
            elif self.init_sp_from == 'before':  # use the gaussians at init_stage
                for i, name in enumerate(['_xyz', '_features_dc', '_scaling', '_rotation', '_opacity']):
                    setattr(self, name, nn.Parameter(self.init_sp_temp[i]))
                self._features_rest = nn.Parameter(self._features_rest.new_zeros(
                    self._xyz.shape[0], *self._features_rest.shape[1:])
                )

            if self.hyper_dim > 0:
                self.hyper_feature = nn.Parameter(
                    torch.full([self._xyz.shape[0], self.hyper_dim], -1e-2, device=self.sp_points.device))
            self.init_sp_temp = []
            new_params = {v: getattr(self, k) for k, v in self.param_names_map.items()}
            if self.sp_W is not None:
                _, p2sp = utils.cdist_top(self.points, self.sp_points)
                scale = math.log(9 * (self.num_knn - 1))
                new_params['sp_W'] = F.one_hot(p2sp, self.num_superpoints).float() * scale  # [0.9, 0.1/(K-1), ...]
            new_params = self.change_optimizer(self._task.optimizer, new_params, op='replace')
            for param_name, opt_name in self.param_names_map.items():
                setattr(self, param_name, new_params[opt_name])
            self.active_sh_degree = 0
            self.training_setup()
            logging.info('Finish superpoints initialization')
            self._task.save_model('init.pth')
        if self._step == self.simplify_steps[0]:
            self.important_sample_with_culling(self._task.optimizer, False, False)
            self.reset_optimizer(self._task.optimizer)
            self.xyz_gradient_accum = self.xyz_gradient_accum.new_zeros((self._xyz.shape[0], 1))
            self.denom = self.denom.new_zeros((self._xyz.shape[0], 1))
        if self._step == self.simplify_steps[1]:
            self.important_sample_with_culling(self._task.optimizer, False, True)
        if self._step == self.simplify_steps[2]:  # self._step == (simplify_stepss[1] + epochs) // 2:
            self._culling = self._culling.new_zeros((len(self.points), self.num_frames))
            self.factor_culling = self.factor_culling.new_ones((len(self.points), 1))
            logging.info(f'init culling at {self._step}, have {len(self.points)} Gaussians')

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if isinstance(viewspace_point_tensor, Tensor):
            grad = viewspace_point_tensor.grad
        elif len(viewspace_point_tensor) == 1:
            grad = viewspace_point_tensor[0].grad
        else:
            grad = torch.zeros_like(viewspace_point_tensor[0])
            for p in viewspace_point_tensor:
                grad += p.grad
        pixel_grad = torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += pixel_grad * self.factor_culling[update_filter]
        self.denom[update_filter] += 1

    def densify_and_split(self, optimizer: torch.optim.Optimizer, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.points.shape[0]
        device = self.points.device
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = selected_pts_mask & torch.gt(torch.amax(self.get_scaling, dim=1), scene_extent)

        if self.adaptive_control_cfg['blur_split']:
            selected_pts_mask[:len(self.mask_blur)] = selected_pts_mask[:len(self.mask_blur)] | self.mask_blur

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        if '2d_gs' in self.render_method:
            stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
            means = torch.zeros_like(stds)
        else:
            means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        if self.use_so3:
            rots = ops_3d.rotation.lie_to_R(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        else:
            rots = ops_3d.rotation.quaternion_to_R(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_params = {
            '_xyz': torch.bmm(rots, samples[..., None]).squeeze(-1) + self.points[selected_pts_mask].repeat(N, 1),
            '_scaling': self.scaling_activation_inverse(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        }
        for param_name, opt_name in self.param_names_map.items():
            if param_name == '_xyz' or param_name == '_scaling':
                new_params[opt_name] = new_params.pop(param_name)
            else:
                param = getattr(self, param_name)
                new_params[opt_name] = param[selected_pts_mask].repeat(N, *[1] * (param.ndim - 1))

        self.densification_postfix(optimizer, **new_params, mask=selected_pts_mask, N=N)

        prune_filter = torch.cat((selected_pts_mask, selected_pts_mask.new_zeros(N * selected_pts_mask.sum())))
        self.prune_points(optimizer, prune_filter)

    def prune_points(self, optimizer: torch.optim.Optimizer, mask):
        super().prune_points(optimizer, mask)
        valid_points_mask = ~mask
        self._culling = self._culling[valid_points_mask]
        self.factor_culling = self.factor_culling[valid_points_mask]
        if self.p2sp is not None:
            self.p2sp = self.p2sp[valid_points_mask]

    def densification_postfix(self, optimizer, mask=None, N=None, **kwargs):
        super().densification_postfix(optimizer, **kwargs, N=N, mask=mask)
        if N is None:
            self._culling = torch.cat((self._culling, self._culling[mask]))
            self.factor_culling = torch.cat((self.factor_culling, self.factor_culling[mask]))
        else:
            self._culling = torch.cat((self._culling, self._culling[mask].repeat(N, 1)))
            self.factor_culling = torch.cat((self.factor_culling, self.factor_culling[mask].repeat(N, 1)))
        if self.p2sp is not None:
            if N is None:
                self.p2sp = torch.cat([self.p2sp, self.p2sp[mask]], dim=0)
            else:
                self.p2sp = torch.cat([self.p2sp, self.p2sp[mask].repeat(N)], dim=0)

    # def adaptive_control_init_stage(self, inputs, outputs, optimizer, step: int):
    #     if step < self.init_sampling_step:
    #         radii = outputs['radii']
    #         if radii.ndim == 2:
    #             radii = radii.amax(dim=0)
    #         mask = radii > 0
    #         self.add_densification_stats(outputs['viewspace_points'], mask)
    #         #if step % self.adaptive_control_cfg['densify_interval'][0] == 0 or step == self.stages['init_fix'][1]- 1:
    #         if utils.check_interval_v2(step, *self.adaptive_control_cfg['init_densify_prune_interval']):
    #             size_threshold = 20 if step > self.adaptive_control_cfg['opacity_reset_interval'][0] else None
    #             if self.deform_net.is_blender:
    #                 grad_max = self.adaptive_control_cfg['densify_grad_threshold']
    #             else:
    #                 if self.points.shape[0] > self.num_superpoints * self.node_max_num_ratio_during_init:
    #                     grad_max = torch.inf
    #                 else:
    #                     grad_max = self.adaptive_control_cfg['densify_grad_threshold']
    #             self.densify(optimizer, grad_max, self.cameras_extent,
    #                 self.adaptive_control_cfg['densify_percent_dense'])
    #             self.prune(optimizer, self.adaptive_control_cfg['prune_opacity_threshold'],
    #                 self.cameras_extent, size_threshold, self.adaptive_control_cfg['prune_percent_dense'])
    #             logging.info(f'Node after densify and prune, there are {len(self.points)} points at step {step}')
    #         # if (step >= 100 and (step - 1) % self.adaptive_control_cfg['opacity_reset_interval'][0] == 0) or \
    #         #     (self.background_type == 'white' and step == self.adaptive_control_cfg['densify_interval'][1]):
    #         if utils.check_interval_v2(step, *self.adaptive_control_cfg['init_opacity_reset_interval']):
    #             self.reset_opacity(optimizer)
    #             logging.info(f'reset opacity at init step {step}')
    #     elif step == self.init_sampling_step:
    #         if self.init_sp_from == 'before':
    #             self.init_sp_temp = [getattr(self, name).clone() for name in
    #                 ['_xyz', '_features_dc', '_scaling', '_rotation', '_opacity']]
    #         self.init_superpoints(True, True)
    #         optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def adaptive_control(self, inputs, outputs, optimizer, step: int):
        stage = outputs['stage']
        step = step + 1  # - self.stages[stage][0]
        if stage == 'sp_fix':
            return
        cfg: Dict[str, Any] = self.adaptive_control_cfg  # noqa
        densify_interval = cfg['densify_interval']
        densify_interval = cfg.get(f'{stage}_densify_interval', densify_interval)
        opacity_reset_interval = cfg['opacity_reset_interval']
        opacity_reset_interval = cfg.get(f'{stage}_opacity_reset_interval', opacity_reset_interval)
        # if step <= self.stages['sp_fix'][0]:
        #     self.adaptive_control_init_stage(inputs, outputs, optimizer, step)
        #     return
        # if step <= self.stages['sk_init'][0]:
        #     step = step - self.stages['sp_fix'][0]
        # elif step > self.stages['sk_fix'][0]:
        #     if not self.sk_densify_gs:
        #         return
        #     step = step - self.stages['sk_fix'][0]
        # else:
        #     return
        if stage == 'init' and step >= self.init_sp_step:
            return
        if step >= densify_interval[2] > 0:
            return
        radii = outputs['radii']
        if radii.ndim == 2:
            radii = radii.amax(dim=0)
        mask = radii > 0
        # Keep track of max radii in image-space for pruning
        self.max_radii2D[mask] = torch.max(self.max_radii2D[mask], radii[mask])
        self.add_densification_stats(outputs['viewspace_points'], mask)

        if utils.check_interval_v2(step, *densify_interval, close='()') and step not in self.depth_reinit_steps:
            if cfg['blur_split']:
                area_threshold = outputs['images'].shape[-2] * outputs['images'].shape[-3] * 2e-4
                self.mask_blur = torch.logical_or(self.mask_blur, outputs["area_max"].squeeze() > area_threshold)

            num0 = len(self.points)
            self.densify(
                optimizer,
                max_grad=cfg['densify_grad_threshold'],
                extent=self.cameras_extent,
                densify_percent_dense=cfg['densify_percent_dense']
            )
            num1 = len(self.points)
            if step > opacity_reset_interval[0] and cfg['prune_max_screen_size'] > 0:
                size_threshold = cfg['prune_max_screen_size']
            else:
                size_threshold = None
            self.prune(
                optimizer,
                min_opacity=cfg['prune_opacity_threshold'],
                extent=self.cameras_extent,
                max_screen_size=size_threshold,
                prune_percent_dense=cfg['prune_percent_dense'],
            )
            num2 = len(self.points)
            if cfg['blur_split']:
                self.mask_blur = self.mask_blur.new_zeros(self._xyz.shape[0])
            logging.info(f'densify & prune: {num2} (+{num1 - num0}, -{num1 - num2}) points at step {step}')
        if utils.check_interval_v2(step, *opacity_reset_interval, close='()') and step not in self.depth_reinit_steps:
            # or (            self.background_type == 'white' and step == densify_interval[1]):
            self.reset_opacity(optimizer)
            logging.info(f'reset opacity at step {step}')
        if step in self.depth_reinit_steps:
            self.depth_reinit(optimizer)
        if utils.check_interval_v2(step, *cfg['aggressive_clone_interval']) and step not in self.depth_reinit_steps:
            self.aggressive_clone(optimizer)
        if hasattr(optimizer, 'set_before_step'):
            optimizer.set_before_step(mask, radii.shape[0])

    def extra_repr(self) -> str:
        return f"render_method={self.render_method}, LBS={self.LBS_method}, warp={self.warp_method}"

    @torch.no_grad()
    def compute_important_score_and_update_culling(self, masked_score=False):
        P, device = self._xyz.shape[0], self.device
        imp_score = torch.zeros(P, device=device)
        accum_area_max = torch.zeros(P, device=device)
        count_rad = torch.zeros((P, 1), device=device)
        count_vis = torch.zeros((P, 1), device=device)
        self._culling = self._culling.new_zeros((P, self.num_frames))

        db = self._task.train_db  # type: NERF_Base_Dataset # noqa
        size = torch.as_tensor(db.image_size)
        # if self._step <= self.warmup_step:
        #     size = size // 2
        bg = torch.ones(3, device=device) if db.background_type == 'white' else torch.zeros(3, device=device)
        for i in range(self.num_frames):
            Tw2v, Tv2c, Tv2s, campos, FoV, t = utils.tensor_to(
                db.cameras.Tw2v[i], db.cameras.Tv2c[i], db.cameras.Tv2s[i], db.cameras.Tv2w[i, :3, 3],
                db.cameras.FoV[i], db.times[i], device=device
            )
            outputs = self.render(
                t=t, background=bg,
                info=dict(Tw2v=Tw2v, Tv2c=Tv2c, Tv2s=Tv2s, campos=campos, FoV=FoV, size=size, index=torch.tensor(i)),
                render_type='simp'
            )
            accum_weights = outputs["accum_weights"].squeeze()
            area_proj = outputs["area_proj"].squeeze()
            area_max = outputs["area_max"].squeeze()

            if masked_score:
                mask_t = area_max != 0
                temp = imp_score + accum_weights / area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score = imp_score + accum_weights

            accum_area_max = accum_area_max + area_max
            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)
            self._culling[:, i] = torch.logical_not(non_prune_mask)
            count_rad[outputs["radii"].squeeze() > 0] += 1
            count_vis[non_prune_mask] += 1
            del outputs
        return imp_score, accum_area_max, count_vis, count_rad

    @torch.no_grad()
    def important_sample_with_culling(self, optimizer, is_dense=False, preserving=True):
        """
        based mini splatting2
        culling_with_interesction_preserving
        culling_with_interesction_sampling
        culling_with_importance_pruning
        """
        P, device = self._xyz.shape[0], self.device
        imp_score, accum_area_max, count_vis, count_rad = self.compute_important_score_and_update_culling(
            not is_dense or self.imp_metric == 'outdoor'
        )
        if preserving or is_dense:
            if not is_dense:
                imp_score[accum_area_max == 0] = 0
            non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99)
        else:
            imp_score[accum_area_max == 0] = 0

            prob = (imp_score / imp_score.sum()).cpu().numpy()
            num_sampled = int(P * self.sampling_factor * np.sum(prob != 0) / prob.shape[0])
            indices = np.random.choice(P, size=num_sampled, p=prob, replace=False)

            non_prune_mask = np.zeros(P, dtype=bool)
            non_prune_mask[indices] = True
            non_prune_mask = torch.from_numpy(non_prune_mask).to(device)

        self.factor_culling = count_vis / (count_rad + 1e-1)
        prune_mask = (count_vis <= 1)[:, 0]
        prune_mask = torch.logical_or(prune_mask, torch.logical_not(non_prune_mask))
        self.prune_points(optimizer, prune_mask)

        logging.info(f'important_sample_with_culling at {self._step}, have {len(self.points)} Gaussians')
        torch.cuda.empty_cache()

    def aggressive_clone(self, optimizer):
        imp_score, accum_area_max, count_vis, count_rad = self.compute_important_score_and_update_culling()
        self.factor_culling = count_vis / (count_rad + 1e-1)
        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.999)
        prune_mask = (count_vis <= 1)[:, 0]
        prune_mask = torch.logical_or(prune_mask, torch.logical_not(non_prune_mask))
        self.prune_points(optimizer, prune_mask)

        imp_score[accum_area_max == 0] = 0
        intersection_pts_mask = init_cdf_mask(importance=imp_score, thres=0.99)
        intersection_pts_mask = intersection_pts_mask[~prune_mask]
        self.clone(optimizer, intersection_pts_mask)
        torch.cuda.empty_cache()
        self.mask_blur = self.mask_blur.new_zeros(self._xyz.shape[0])
        logging.info(
            f"aggressive clone at {self._step}, have {len(self.points)}(+{intersection_pts_mask.sum().item()}, -{prune_mask.sum().item()}) Gaussians")

    def clone(self, optimizer, selected_pts_mask):
        temp_opacity_old = self.get_opacity[selected_pts_mask]
        new_opacity = 1 - (1 - temp_opacity_old) ** 0.5

        temp_scale_old = self.get_scaling[selected_pts_mask]
        new_scaling = (temp_opacity_old / (2 * new_opacity - 0.5 ** 0.5 * new_opacity ** 2)) * temp_scale_old

        new_opacity = torch.clamp(new_opacity, max=1.0 - torch.finfo(torch.float32).eps, min=0.0051)
        new_opacity = self.opacity_activation_inverse(new_opacity)
        new_scaling = self.scaling_activation_inverse(new_scaling)

        self._scaling[selected_pts_mask] = new_scaling
        self._opacity[selected_pts_mask] = new_opacity

        new_params = {}
        for param_name, opt_name in self.param_names_map.items():
            param = getattr(self, param_name)
            new_params[opt_name] = param[selected_pts_mask]
        new_params['_scaling'] = new_scaling
        new_params['_opacity'] = new_opacity
        self.densification_postfix(optimizer, **new_params, mask=selected_pts_mask)

    def depth_reinit(self, optimizer):
        P, device = len(self.points), self.device
        num_depth = P * self.num_depth_factor
        # intersection_preserving
        imp_score, accum_area_max, count_vis, count_rad = self.compute_important_score_and_update_culling(
            self.imp_metric == 'outdoor'
        )
        imp_score[accum_area_max == 0] = 0
        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99)
        self.prune_points(optimizer, torch.logical_not(non_prune_mask))

        points = []
        colors = []
        db = self._task.train_db  # type: NERF_Base_Dataset # noqa
        bg = torch.ones(3, device=device) if db.background_type == 'white' else torch.zeros(3, device=device)
        size = db.image_size
        # if step <= self.warmup_step:
        #     size = size // 2
        for i in range(self.num_frames):
            Tw2v, Tv2c, Tv2s, campos, FoV, gt, t = utils.tensor_to(
                db.cameras.Tw2v[i], db.cameras.Tv2c[i], db.cameras.Tv2s[i], db.cameras.Tv2w[i, :3, 3],
                db.cameras.FoV[i], db.get_image(i),
                db.times[i if self.canonical_time_id < 0 else self.canonical_time_id],
                device=device
            )
            gt = gt[..., :3].reshape(-1, 3)
            # assert self.render_method.startswith('3d_gs')

            net_out = self(t=t, stage='static' if self.canonical_time_id < 0 else '')
            if '3d' in self.render_method:
                render_depth_pkg = render_mid_depth(
                    **{k: v for k, v in net_out.items() if not k.startswith('_')},
                    Tw2v=Tw2v.contiguous(), Tv2c=Tv2c, Tv2s=Tv2s, size=size, FoV=FoV, campos=campos, is_opengl=False,
                    # culling=self._culling[:, i].contiguous()
                )
                out_pts = render_depth_pkg["out_pts"]
                prob = 1 - render_depth_pkg["accum_alpha"]
                if i == 0:
                    plt.imshow(utils.as_np_image(render_depth_pkg['images']))
                    plt.show()
            else:
                outputs = self.gs_rasterizer(**{k: v for k, v in net_out.items() if not k.startswith('_')},
                                             Tw2v=Tw2v.contiguous(), Tv2c=Tv2c, Tv2s=Tv2s, size=size, FoV=FoV,
                                             campos=campos, is_opengl=False)
                out_pts = ops_3d.pixel2points(outputs['depths'].squeeze(0), Tv2s=Tv2s, Tw2v=Tw2v)
                prob = outputs['opacity']

            prob = prob / prob.sum()
            prob = prob.reshape(-1).cpu().numpy()

            factor = 1 / (size[0] * size[1] * self.num_frames / num_depth)
            indices = np.random.choice(prob.shape[0], size=int(prob.shape[0] * factor), p=prob, replace=False)

            out_pts = out_pts.permute(1, 2, 0).reshape(-1, 3)

            points.append(out_pts[indices])
            colors.append(gt[indices])

        points = torch.cat(points)
        colors = RGB2SH(torch.cat(colors))
        # reinitialize_pts
        dist2 = torch.clamp_min(get_C_function('simple_knn')(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3 if '3d_gs' in self.render_method else 2)
        rots = torch.zeros((points.shape[0], 4), device="cuda")
        rots[:, -1] = 1

        opacities = self.opacity_activation_inverse(0.1 * self._opacity.new_ones((points.shape[0], 1)))
        new_params = {
            'xyz': points.contiguous(),
            'f_dc': colors[:, None, :],
            'f_rest': self._features_rest.new_zeros(len(colors), (self.max_sh_degree + 1) ** 2 - 1, 3),
            'scaling': scales,
            'rotation': rots,
            'opacity': opacities,
        }
        params = self.change_optimizer(optimizer, new_params, op='replace')
        for param_name, opt_name in self.param_names_map.items():
            setattr(self, param_name, params[opt_name])
        self.reset_optimizer(optimizer)
        self.training_setup()
        torch.cuda.empty_cache()
        logging.info(f"after depth_reinit at {self._step}, have {len(self.points)} Gaussians")


class MLP_with_skips(nn.Module):
    def __init__(
            self, in_channels: int, dim_hidden: int, out_channels: Union[int, Sequence[int]] = 0, num_layers: int = 0,
            skips: Sequence[int] = (), bias=True, weight_norm=False, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.skips = tuple(skips)
        self.bias = bias
        self.weight_norm = weight_norm

        net = []
        for i in range(num_layers):
            net.append(nn.Linear(in_channels, self.dim_hidden, bias=bias))
            if weight_norm:
                nn.utils.weight_norm(net[-1])
            in_channels = self.dim_hidden + (self.in_channels if i in self.skips else 0)
        self.net = nn.ModuleList(net)
        if isinstance(out_channels, int):
            self.last = nn.Linear(in_channels, out_channels, bias=bias) if out_channels > 0 else None
        else:
            self.last = nn.ModuleList(nn.Linear(in_channels, oc, bias=bias) for oc in out_channels)
        if weight_norm:
            nn.utils.weight_norm(self.last)

    def forward(self, inputs: Tensor):
        x = inputs
        for i in range(self.num_layers):
            x = self.net[i](x)
            x = F.relu(x, inplace=True)
            if i in self.skips:
                x = torch.cat([x, inputs], dim=-1)

        if isinstance(self.last, nn.ModuleList):
            return [m(x) for m in self.last]
        elif self.last is not None:
            x = self.last(x)
        return x

    def __repr__(self):
        return (f"{self.__class__.__name__}(in={self.in_channels}, out={self.out_channels}, "
                f"hidden={self.dim_hidden}, num_layers={self.num_layers}, skips={self.skips}, bias={self.bias}"
                f"{'weight_norm=True' if self.weight_norm else ''}"
                f")")


def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()
    if thres != 1.0:
        percent_sum = thres
        vals, idx = torch.sort(importance + 1e-6)
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val / vals.sum()) > (1 - percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance > split_val_nonprune
    else:
        non_prune_mask = torch.ones_like(importance).bool()

    return non_prune_mask


@torch.no_grad()
def render_mid_depth(
        Tw2v: Tensor,
        Tv2c: Tensor,
        size: Tuple[int, int],
        FoV: Tensor,
        campos: Optional[Tensor],
        points: Tensor,
        opacity: Tensor,
        scales: Tensor = None,
        rotations: Tensor = None,
        sh_features: Tensor = None,
        sh_features_rest: Tensor = None,
        cov3Ds: Tensor = None,
        cov2Ds: Tensor = None,
        colors: Tensor = None,
        sh_degree=0,
        is_opengl=False,
        culling: Tensor = None,
        bg: Tensor = None,
        **kwargs
):
    means2D = points.new_zeros((points.shape[0], 2))
    W, H = size
    tanFoV = torch.tan(FoV * 0.5)
    means2D, depths, colors, radii, rects, tiles_touched, cov3D, conic_opacity = get_C_function(
        'gs_preprocess_forward_v2')(
        W, H, sh_degree if colors is None else -1, is_opengl,
        points, scales, rotations, opacity, sh_features if colors is None else colors, sh_features_rest,
        Tw2v, Tv2c, campos, tanFoV,
        cov3Ds, cov2Ds, means2D, culling
    )
    Tc2w = (Tv2c @ Tw2v).inverse().contiguous()
    if bg is None:
        bg = colors.new_zeros(3)
    tile_ranges, point_list = get_C_function('GS_prepare_v3')(
        W, H, means2D, conic_opacity, depths, radii, tiles_touched, rects)
    outputs = get_C_function('gs_mid_depth')(
        W, H, means2D, conic_opacity, colors, point_list, tile_ranges, points, scales, rotations, Tc2w, campos, bg
    )
    return {
        'images': outputs[0],
        'out_pts': outputs[1],
        'rendered_depth': outputs[2],
        'accum_alpha': outputs[3],
        'gidx': outputs[4],
        'discriminants': outputs[5],
    }


# @try_use_C_extension
def get_superpoint_features(value: Tensor, neighbor: Tensor, G: Tensor, num_sp: int):
    """ value_sp[j] = 1 / w[j] sum_{i=0}^{N} [j in neighbor[i]] G[i, j] value[i]
    w[j] = sum_{i=0}^{N} [j in neighbor[i]] G[i, j]

    Args:
        G: shape [N, K]
        neighbor: shape: [N, K] The indices of K-nearest superpoints for each point
        value: [N, C]
        num_sp: The number of superpoints
    Returns:
        Tensor: the value for superpoints, shape: [num_sp, C]
    """
    C = value.shape[-1]
    assert 0 <= neighbor.min() and neighbor.max() < num_sp
    value_sp = value.new_zeros([num_sp, C])
    value_sp = torch.scatter_reduce(
        value_sp,
        dim=0,
        index=neighbor[:, :, None].repeat(1, 1, C).view(-1, C),
        src=(value[:, None, :] * G[:, :, None]).view(-1, C),
        reduce='sum'
    )
    w = value.new_zeros([num_sp]).scatter_reduce_(dim=0, index=neighbor.view(-1), src=G.view(-1), reduce='sum')
    return value_sp / w[:, None].clamp_min(1e-5)


class SC_GS_DeformNetwork(nn.Module):
    def __init__(
            self,
            D=8,
            W=256,
            input_ch=3,
            output_ch=59,
            pos_enc_p='freq',
            pos_enc_p_cfg: dict = None,
            pos_enc_t='freq',
            pos_enc_t_cfg: dict = None,
            is_blender=False,
            local_frame=False,
            resnet_color=True,
            color_wrt_dir=False,
            max_d_scale=-1,
            **kwargs
    ):  # t_multires 6 for D-NeRF; 10 for HyperNeRF
        super(SC_GS_DeformNetwork, self).__init__()
        self.name = 'mlp'
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.pos_enc_p = FreqEncoder(**utils.merge_dict(pos_enc_p_cfg, input_dim=3))
        self.pos_enc_t = FreqEncoder(**utils.merge_dict(pos_enc_t_cfg, input_dim=1))
        self.input_ch = self.pos_enc_p.output_dim + self.pos_enc_t.output_dim

        self.resnet_color = resnet_color
        self.color_wrt_dir = color_wrt_dir
        self.max_d_scale = max_d_scale

        self.reg_loss = 0.

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(self.pos_enc_t.output_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out)
            )

            self.linear = nn.ModuleList([nn.Linear(self.pos_enc_p.output_dim + self.time_out, W)])
            for i in range(D - 1):
                if i not in self.skips:
                    self.linear.append(nn.Linear(W, W))
                else:
                    self.linear.append(nn.Linear(W + self.pos_enc_p.output_dim + self.time_out, W))

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_scaling = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)

        self.local_frame = local_frame
        if self.local_frame:
            self.local_rotation = nn.Linear(W, 4)
        self.reset_parameters()

    def reset_parameters(self):
        if self.local_frame:
            nn.init.normal_(self.local_rotation.weight, mean=0, std=1e-4)
            nn.init.zeros_(self.local_rotation.bias)

        for layer in self.linear:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.gaussian_warp.weight, mean=0, std=1e-5)
        nn.init.normal_(self.gaussian_scaling.weight, mean=0, std=1e-8)
        nn.init.normal_(self.gaussian_rotation.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.gaussian_warp.bias)
        nn.init.zeros_(self.gaussian_scaling.bias)
        nn.init.zeros_(self.gaussian_rotation.bias)

    def forward(self, x: Tensor, t: Tensor, **kwargs):
        t_emb = self.pos_enc_t(t.view(-1, 1)).expand(x.shape[0], self.pos_enc_t.output_dim)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.pos_enc_p(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        if self.max_d_scale > 0:
            scaling = torch.tanh(scaling) * np.log(self.max_d_scale)
        local_rotation = self.local_rotation(h) if self.local_frame else None
        return d_xyz, rotation, scaling, local_rotation


class DeformationNetwork(nn.Module):
    def __init__(
            self,
            net_width=256,
            net_depth=8,
            net_skips=(4,),
            pos_enc_p='freq',
            pos_enc_p_cfg: dict = None,
            pos_enc_t='freq',
            pos_enc_t_cfg: dict = None,
            use_6dof=False,
            use_time_net=False,
            delta_scale=True,
    ):
        super(DeformationNetwork, self).__init__()
        self.pos_enc_p = FreqEncoder(**utils.merge_dict(pos_enc_p_cfg, input_dim=3))
        self.pos_enc_t = FreqEncoder(**utils.merge_dict(pos_enc_t_cfg, input_dim=1))
        if use_time_net:
            self.time_out = 30
            self.timenet = nn.Sequential(
                nn.Linear(self.pos_enc_t.output_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out)
            )
            in_channels = self.pos_enc_p.output_dim + self.time_out
        else:
            in_channels = self.pos_enc_p.output_dim + self.pos_enc_t.output_dim
        self.linear = nn.ModuleList(
            [nn.Linear(in_channels, net_width)] + [
                nn.Linear(net_width + (in_channels if i in net_skips else 0), net_width) for i in range(net_depth - 1)
            ])

        if use_6dof:
            self.branch_w = nn.Linear(net_width, 3)
            self.branch_v = nn.Linear(net_width, 3)
        else:
            self.gaussian_warp = nn.Linear(net_width, 3)
        self.gaussian_rotation = nn.Linear(net_width, 4)
        self.gaussian_scaling = nn.Linear(net_width, 3) if delta_scale else None

        self.use_6dof = use_6dof
        self.use_time_net = use_time_net
        self.net_skips = net_skips

    def forward(self, points: Tensor, t: Tensor):
        t_emb = self.pos_enc_t(t.view(-1, 1).expand(points.shape[0], 1))
        if self.use_time_net:
            t_emb = self.timenet(t_emb)
        x_emb = self.pos_enc_p(points)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.net_skips:
                h = torch.cat([x_emb, t_emb, h], dim=-1)
        delta_r = self.gaussian_rotation(h)
        delta_s = self.gaussian_scaling(h) if self.gaussian_scaling is not None else 0

        if self.use_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            delta_points = torch.cat([w, v], dim=-1)
        else:
            delta_points = self.gaussian_warp(h)
        return delta_points, delta_r, delta_s
