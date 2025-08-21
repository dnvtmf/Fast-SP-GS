import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Callable, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import datasets
import utils
import data_loader
from datasets.colmap_dataset import fetchPly, storePly
from fast_2d_gs.gaussian_splatting import BasicPointCloud, SH2RGB
from fast_2d_gs.sparse_gaussian_adam import SparseGaussianAdam
from fast_2d_gs.network import FastSuperpoint2DGaussianSplatting


# 总共训练iterations次
class GaussianTrainTask:
    def __init__(self, config_args=None, *args, **kwargs):
        # module
        self.description = self.__doc__
        self.model_name = ""
        self.model_cfg = ""
        self.output = Path(__file__).absolute().parent.parent / 'results'
        self.device = None  # type: Optional[torch.device]
        self.model = nn.Module()  # type: torch.nn.Module
        self.criterion = nn.Module()  # type: torch.nn.Module
        self.optimizer = None  # type: Optional[Union[torch.optim.SGD, Dict[str,torch.optim.SGD]]]
        self.train_db = None  # type: Optional[datasets.NERF_Base_Dataset]
        self.eval_db = None  # type: Optional[datasets.NERF_Base_Dataset]
        self.test_db = None  # type: Optional[datasets.NERF_Base_Dataset]
        self.train_loader = None  # type: Optional[DataLoader]
        self.eval_loader = None  # type: Optional[DataLoader]
        self.test_loader = None  # type: Optional[DataLoader]
        self.cfg = None  # type:Optional[argparse.Namespace]
        self.logger = None  # type: Optional[logging.Logger]
        self.train_timer = utils.TimeEstimator()  # 运行时间估计
        self.checkpoint_manager = utils.checkpoint.CheckpointManager()
        self.interval_grad_acc = 1  # The interval step for gradient accumulation
        # running status
        self.global_step = 0
        self._num_accumulated_steps = 0
        self.epoch = 0
        self.step = 0
        self.num_steps = 1
        self.num_epochs = 1
        self.is_during_training = False
        self.metric_manager = utils.metrics.MetricManager()
        # hooks
        self.hook_manager = utils.HookManager()
        self.hook_manager.add_hook(
            lambda: self._set_now_state(step=self.step, is_during_training=True), 'before_train_epoch',
            insert=0,
        )
        self.hook_manager.add_hook(lambda: self._set_now_state(step=self.step + 1), 'after_train_step', insert=0)
        self.hook_manager.add_hook(lambda: self._set_now_state(epoch=0), 'after_train_epoch', insert=0)
        self.hook_manager.add_hook(lambda: self._set_now_state(is_during_training=False), 'before_eval_epoch', insert=0)
        self.configure(config_args, *args, **kwargs)

    def configure(self, config_args=None, *args, **kwargs):
        self.step_1_config(
            args=config_args,
            ignore_unknown=kwargs.setdefault('ignore_unknown', False),
            ignore_warning=kwargs.setdefault('ignore_warning', False),
            parser=kwargs.setdefault('parser', None),
        )
        self.step_2_environment()
        self.step_3_dataset()
        self.step_4_model()
        self.step_5_data_loader_and_transform()
        self.step_6_optimizer()
        self.step_7_lr()
        self.step_8_others()

    def step_1_config(self, args=None, ignore_unknown=False, ignore_warning=False, parser=None):
        parser = utils.config.get_parser(parser, self.description)
        # config
        utils.config.options(parser)
        # environment
        utils.checkpoint.options(parser)
        utils.trainer.options(parser)
        utils.my_logger.options(parser)
        utils.metrics.options(parser)
        datasets.options(parser)
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        group = parser.add_argument_group("Optimizer Options:")
        utils.add_bool_option(group, '--use-spare-adam', default=True, help='use spare adam optimizer')
        group.add_argument("-oc", "--optimizer-cfg", default={}, type=utils.str2dict, metavar="D",
                           help="The configure for optimizer")
        utils.add_cfg_option(group, '--arch-cfg', help="The configure for networks")
        # data load
        data_loader.options(parser)
        # extra
        parser.add_argument('--exp-name', default='nerf', help='The name of experiments')
        utils.add_cfg_option(parser, '--train-kwargs', help='extra train kwargs')
        utils.add_cfg_option(parser, '--eval-kwargs', help='extra eval kwargs')
        utils.add_cfg_option(parser, '--test-kwargs', help='extra test kwargs')
        utils.add_bool_option(parser, '--save-video', default=True, help='Save the test results to vedio')
        parser.add_argument('--eval-num-steps', default=-1, type=int, help='The steps when evaluate during training')
        parser.add_argument('--vis-interval', default=1_000, type=int)
        utils.add_cfg_option(parser, '--vis-kwargs', help='The config for visualize')
        utils.add_bool_option(parser, '--vis-clear', default=None, help='clear visualization')
        parser.add_argument('--num-init-points', default=100_000, type=int)
        utils.add_path_option(parser, '--init-ply', default=None)
        utils.add_bool_option(parser, '--random-pcd', default=False)

        self.cfg = utils.config.make(args, ignore_unknown, ignore_warning, parser=parser)
        return self.cfg

    def step_2_environment(self, *args, output_paths=(), log_filename=None, **kwargs):
        cfg = self.cfg
        self.checkpoint_manager = utils.checkpoint.make(self.cfg)
        self.store('cfg')
        self.device = torch.device('cuda')
        self.set_output_dir(*output_paths)
        if log_filename is None:
            log_filename = self.cfg.log_filename if self.cfg.log_filename else (self.mode + ".log")
        self.logger = utils.my_logger.make(cfg, self.output, log_filename, enable=True)
        utils.trainer.make(cfg)
        self.logger.info(f"==> the output path: {self.output}")
        self.logger.info(f'==> Task Name: {self.__class__.__name__}')

        self.store('global_step')
        if self.cfg.start_epoch is not None:
            self.epoch = self.cfg.start_epoch

        self.metric_manager = utils.metrics.make(self.cfg)
        self.store('metric_manager')
        self.logger.info(f'==> Metric: {self.metric_manager}')
        self.hook_manager.add_hook(self.metric_manager.summarize, 'after_eval_epoch')
        self.hook_manager.add_hook(self.metric_manager.reset, 'before_eval_epoch')
        if self.cfg.start_epoch is not None:
            self.step = self.cfg.start_epoch
        self.num_steps = self.cfg.epochs

    def step_3_dataset(self, *args, **kwargs):
        if self.mode == 'train':
            self.train_db = datasets.make(self.cfg, mode='train')
        if self.mode == 'eval' or (self.mode == 'train' and self.cfg.eval_interval > 0):
            self.eval_db = datasets.make(self.cfg, mode='eval')
        if self.mode == 'test':
            self.test_db = datasets.make(self.cfg, mode='test')

        if not self.cfg.random_pcd and self.cfg.init_ply is not None:
            self.logger.info(f"try to Load init point cloud from: {self.cfg.init_ply}")
            pcd = fetchPly(self.cfg.init_ply)
        # elif not self.cfg.random_pcd and isinstance(self.train_db, (ColmapDataset, DyNeRFColmapDataset)):
        #     pcd = self.train_db.point_cloud
        #     logging.info(f"[red]load point cloud from colmap")
        # elif not self.cfg.random_pcd and isinstance(self.train_db, NerfiesDataset) and self.train_db.points is not None:
        #     xyz = self.train_db.points
        #     shs = np.random.random((len(xyz), 3)) / 255.0
        #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((len(xyz), 3)))
        else:
            # ply_path = self.train_db.root.joinpath("points3d.ply")
            # if 1 or not ply_path.exists():
            # Since this data set has no colmap data, we start with random points
            num_pts = self.cfg.num_init_points

            self.logger.info(f"Generating random point cloud ({num_pts})...")

            # We create random points inside the bounds of the synthetic Blender scenes
            if hasattr(self.train_db, 'scene_size'):
                min_v = self.train_db.scene_center - self.train_db.scene_size * 0.5
                xyz = np.random.random((num_pts, 3)) * self.train_db.scene_size + min_v  # noqa
            else:
                xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
                self.logger.warning("scene bound are set to [-1.3, 1.3]")
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            # storePly(ply_path, xyz, SH2RGB(shs) * 255)

            # try:
            #     pcd = fetchPly(ply_path)
            # except:
            #     pcd = None
        self._pcd = pcd

    def step_4_model(self, *args, **kwargs):
        self.set_output_dir(self.cfg.exp_name, self.cfg.scene)
        self.model = FastSuperpoint2DGaussianSplatting(**utils.merge_dict(self.cfg.arch_cfg))
        self.model.set_from_dataset(utils.fnn(self.train_db, self.eval_db, self.test_db))
        self.criterion = self.model.loss

        self.load_model()
        self.store('model')
        if not self.cfg.load and not self.cfg.resume:
            self.model.create_from_pcd(self._pcd)
            self.logger.info('create_from_pcd')
            storePly(
                self.output.joinpath('init_points.ply'),
                self.model.points.detach().cpu().numpy(),
                SH2RGB(self.model._features_dc[:, 0].detach().cpu().numpy()) * 255
            )
        self.model.training_setup()
        # if self.mode != 'train':
        #     self.model.active_sh_degree = self.model.max_sh_degree
        self.model.to(self.device)
        if self.criterion and isinstance(self.criterion, nn.Module):  # Some criterion is in the model
            self.criterion.to(self.device)
        # torch.set_anomaly_enabled(True)
        self.logger.info(f"==> Model: {self.model}")
        self.model._task = self

    def step_5_data_loader_and_transform(self):
        if self.train_db is not None:
            self.train_loader = data_loader.make(
                self.cfg, self.train_db, mode='train', batch_sampler='iterable',
                batch_sampler_cfg=dict(length=self.cfg.epochs),
            )
            self.logger.info(f'==> Train db: {self.train_db}')

        if self.eval_db is not None:
            self.eval_loader = data_loader.make(self.cfg, self.eval_db, mode='eval', batch_size=1)
            self.logger.info(f'==> Eval db: {self.eval_db}')

        if self.test_db is not None:
            self.test_loader = data_loader.make(self.cfg, self.test_db, mode='test', batch_size=1)
            self.logger.info(f'==> Test db: {self.test_db}')

    def step_6_optimizer(self, *args, **kwargs):
        if self.mode != 'train':
            return
        m = utils.get_net(self.model)
        optimizer = SparseGaussianAdam if self.cfg.use_spare_adam else torch.optim.Adam
        if hasattr(m, 'get_params'):
            self.optimizer = optimizer(m.get_params(self.cfg), **self.cfg.optimizer_cfg)
        else:
            self.optimizer = optimizer(self.model.parameters(), **self.cfg.optimizer_cfg)
        self.store("optimizer")
        return

    def step_7_lr(self, *args, **kwargs):
        assert hasattr(self.model, 'update_learning_rate')
        self.hook_manager.add_hook(self.model.update_learning_rate, 'before_train_step')
        return

    def step_8_others(self, *args, **kwargs):
        self.hook_manager.add_hook(self.train_timer.start, 'before_train_epoch')
        self.hook_manager.add_hook(self.train_timer.step, 'after_train_step')
        self.hook_manager.add_hook(
            lambda: utils.trainer.
            change_with_training_progress(self.model, self.step, self.num_steps, self.epoch, self.num_epochs),
            'before_train_step'
        )
        self.hook_manager.add_hook(
            lambda: self.logger.info(f"Peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.3f} GiB"),
            'after_train'
        )
        if self.cfg.vis_clear is not None:
            clear_vis = self.cfg.vis_clear
        else:
            clear_vis = self.mode == 'train' and (not self.cfg.debug and not self.cfg.resume)
        if clear_vis:
            utils.dir_create_empty(self.output.joinpath('vis'))
        else:
            self.output.joinpath('vis').mkdir(exist_ok=True, parents=True)
        # self.hook_manager.add_hook(self.visualize, 'after_train_step')

    def save_checkpoint(self, filename="checkpoint.pth", **kwargs):
        kwargs.setdefault('epoch', self.step)
        kwargs.setdefault('save_dir', self.output)
        self.checkpoint_manager.save(filename, **kwargs)

    def run(self):
        self.loss_dict_meter = utils.DictMeter(float2str=utils.float2str)
        self.losses_meter = utils.AverageMeter()
        self.psnr_meter = utils.AverageMeter()

        self.progress = utils.Progress()
        if self.mode == 'train':
            self.progress.add_task('train', self.num_steps, self.step)
        if self.mode != 'test' and self.eval_loader is not None:
            self.progress.add_task('eval', len(self.eval_loader))
        with self.progress:
            self.run_()

    def run_(self):
        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        self.logger.info('{} Begin {} at {} {}'.format('=' * 20, self.mode, now_date.replace('_', ' '), '=' * 20))
        # if self.cfg.test:
        #     self.test()
        if self.cfg.eval:
            # self.hook_manager.before_eval_epoch()
            self.evaluation('eval')
            # self.hook_manager.after_eval_epoch()
        else:
            self.num_epochs = 1
            self.epoch = 0
            self.train_timer.reset(self.num_steps - self.step)
            data_iter = iter(self.train_loader)

            self.model.train()
            self.hook_manager.before_train()
            self.hook_manager.before_train_epoch()
            for self.step in range(self.step, self.num_steps):
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    data = next(data_iter)
                # self.execute_hooks('before_train_step')
                self.train_step(data)
                # self.execute_hooks('after_train_step')
                self.save_checkpoint()
                if utils.check_interval(self.step, self.cfg.eval_interval, self.num_steps):
                    self.hook_manager.after_train_epoch()
                    # self.hook_manager.before_eval_epoch()
                    self.evaluation()
                    # self.hook_manager.after_eval_epoch()
                    self.model.train()
                    self.hook_manager.before_train_epoch()
                    self.logger.info('')
                if self.cfg.debug:
                    exit(1)
            self.hook_manager.after_train_epoch()
            self.hook_manager.after_train()
            self.save_model('last.pth')
        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        self.logger.info("======================  End {} ========================".format(now_date.replace('_', ' ')))
        if self.metric_manager is None or self.metric_manager.best_score is None:
            utils.my_logger.copy_to(suffix=f"_{now_date}", ext='.txt')
        else:
            score = f"score={utils.float2str(self.metric_manager.best_score, 8)}"
            utils.my_logger.copy_to(suffix=f"_{now_date}_{score}", ext='.txt')
        if self.training:
            if self.metric_manager is None or self.metric_manager.best_score is None:
                self.save_model(f'model_{now_date}.pth')
            elif self.output.joinpath('best.pth').exists():
                score = f"score={utils.float2str(self.metric_manager.best_score, 8)}"
                shutil.copyfile(self.output.joinpath('best.pth'), self.output.joinpath(f'{score}_{now_date}.pth'))
            self.checkpoint_manager.remove_all()
            utils.config.save(self.cfg, os.path.join(self.output, 'config.yaml'))

    def train_step(self, data):
        inputs, targets, infos = utils.tensor_to(data, device=self.device, non_blocking=True)
        self.progress.start('train')
        self.model.train()
        self.hook_manager.before_train_step()
        if self.cfg.debug:
            self.logger.debug(f'inputs: {utils.show_shape(inputs)}')
            self.logger.debug(f'targets: {utils.show_shape(targets)}')
            self.logger.debug(f'infos: {utils.show_shape(infos)}')

        if hasattr(self.model, 'render'):
            outputs = self.model.render(**inputs, **self.cfg.train_kwargs, info=infos)
        else:
            outputs = self.model(**inputs, **self.cfg.train_kwargs, info=infos)
        if self.cfg.debug:
            self.logger.debug(f'outputs: {utils.show_shape(outputs)}')
        loss_dict = self.criterion(inputs, outputs, targets, infos)
        if self.cfg.debug:
            self.logger.debug(f'loss_dict: {loss_dict}')
        self.global_step += 1
        losses = sum(loss_dict.values())
        ## skip error gradient
        if torch.isnan(losses) or torch.isinf(losses):
            self.logger.error(f'==> Loss is {losses}')
            # loss.backward()
            exit(1)
            # return loss

        losses.backward()
        self.model.adaptive_control(inputs, outputs, self.optimizer, self.step)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True here can modestly improve performance

        if utils.check_interval(self.step + 1, self.cfg.vis_interval, self.cfg.epochs):
            gt = targets['images'][..., :3]
            gt = gt[0] if gt.ndim == 4 else gt
            image = outputs['images']
            image = image[0] if image.ndim == 4 else image
            diff = (image - gt).abs()
            image = torch.cat([image, gt, diff], dim=1)
            utils.save_image(self.output.joinpath('vis', f'train_{self.step + 1}.png'), image)
            del gt, diff, image
        self.loss_dict_meter.update(loss_dict)
        self.losses_meter.update(losses)

        with torch.no_grad():
            if 'mse' in loss_dict:
                mse = loss_dict['mse']
            else:
                mse = F.mse_loss(outputs['images'][..., :3].reshape(-1), targets['images'][..., :3].reshape(-1))
            psnr = -10 * torch.log10(mse)
            # if self.cfg.weighted_sample:
            #     self.train_db.update_errors(outputs['images'], targets['images'])
        self.psnr_meter.update(psnr)
        self.hook_manager.after_train_step()

        if utils.check_interval(self.step, self.cfg.print_f, self.num_steps):
            lr = [g['lr'] for g in self.optimizer.param_groups if g.get('name', None) == 'xyz'][0]
            self.logger.info(
                f"[{self.step}]/[{self.num_steps}]: "
                f"loss={utils.float2str(self.losses_meter.avg)}, {self.loss_dict_meter.average}, "
                f"psnr={utils.float2str(self.psnr_meter.avg)}, "
                f"lr={utils.float2str(lr)}, "
                f"{self.train_timer.progress}",
            )
            self.psnr_meter.reset()
            self.loss_dict_meter.reset()
            self.losses_meter.reset()
        self.progress.step('train')
        self.visualize()

    def eval_step(self, step, data, **eval_kwargs):
        self.hook_manager.before_eval_step()
        # self.logger.debug(f'inputs: {utils.show_shape(data[0])}')
        # self.logger.debug(f'targets: {utils.show_shape(data[1])}')
        # self.logger.debug(f'infos: {utils.show_shape(data[2])}')

        inputs, targets, infos = utils.tensor_to(*data, device=self.device, non_blocking=True)
        # inputs = {k: None if v is None else v.squeeze(0) for k, v in inputs.items()}
        # targets = {k: None if v is None else v.squeeze(0) for k, v in targets.items()}
        self.logger.debug(f'inputs: {utils.show_shape(inputs)}')
        self.logger.debug(f'targets: {utils.show_shape(targets)}')
        self.logger.debug(f'infos: {utils.show_shape(infos)}')
        # self.logger.debug(f"split: {utils.show_shape(inputs, targets, infos)}")

        if hasattr(self.model, 'render'):
            outputs = self.model.render(**inputs, **eval_kwargs, info=infos)
        else:
            outputs = self.model(**inputs, **eval_kwargs, info=infos)
        pred_images = outputs['images'].clamp(0., 1.)
        self.logger.debug(f'outputs: {utils.show_shape(outputs)}')
        loss_dict = self.criterion(inputs, outputs, targets, infos)
        self.logger.debug(f'losses: {loss_dict}')
        self.metric_manager.update('image', pred_images[..., :3], data[1]['images'][..., :3])
        self.metric_manager.update('loss', sum(loss_dict.values()), **loss_dict)
        if self.cfg.debug:
            plt.figure(dpi=200)
            plt.subplot(121)
            plt.imshow(utils.as_np_image(data[1]['images'].flatten(0, -4)[0]))
            plt.title('gt')
            plt.subplot(122)
            plt.imshow(utils.as_np_image(pred_images.flatten(0, -4)[0]))
            plt.title('predict')
            plt.show()

        if step == 0:
            gt = targets['images'][..., :3]
            gt = gt[0] if gt.ndim == 4 else gt
            diff = (pred_images[0] - gt).abs()
            image = torch.cat([pred_images[0], gt, diff], dim=1)
            utils.save_image(self.output.joinpath('vis', f'eval_{self.step + 1}.png'), image)
            del gt, diff, image

        self.progress.step('eval', self.metric_manager.str())
        self.hook_manager.after_eval_step()
        return

    def evaluation(self, name=''):
        if self.mode == 'train':
            self.progress.pause('train')
        self.hook_manager.before_eval_epoch()
        self.model.eval()
        self.progress.reset('eval', start=True)
        self.progress.start('eval', len(self.eval_loader))
        eval_kwargs = self.cfg.eval_kwargs.copy()
        batch_size = eval_kwargs.pop('batch_size', self.cfg.batch_size[1])
        for step, data in enumerate(self.eval_loader):
            self.eval_step(step, data, **eval_kwargs)
            if self.mode == 'train' and 0 < self.cfg.eval_num_steps <= step:
                break
            if self.cfg.debug:
                break
        self.hook_manager.after_eval_epoch()
        self.logger.info(f"Eval [{self.step}/{self.num_steps}]: {self.metric_manager.str()}")
        if self.mode == 'train':
            if self.metric_manager.is_best:
                self.save_model('best.pth')
            self.progress.stop('eval')
        return

    @torch.no_grad()
    def visualize(self, index=None):
        if not utils.check_interval(self.step, self.cfg.vis_interval):
            return
        self.model.eval()
        torch.cuda.empty_cache()
        vis_kwargs = self.cfg.vis_kwargs.copy()  # type: dict
        batch_size = self.cfg.batch_size[1]
        self.progress.pause('train')
        if index is None:
            index = np.random.randint(0, len(self.train_db))
        logging.info(f"visualize image {index} as step {self.step}")
        inputs, targets, info = utils.tensor_to(*self.train_db[index], device=self.device, non_blocking=True)
        inputs = {k: utils.to_tensor(v, device=self.device) for k, v in inputs.items()}
        targets = {k: utils.to_tensor(v, device=self.device) for k, v in targets.items()}
        self.logger.debug(f'inputs: {utils.show_shape(inputs)}')
        self.logger.debug(f'targets: {utils.show_shape(targets)}')
        self.logger.debug(f'info: {utils.show_shape(info)}')
        info = utils.tensor_to(info, device=self.device)
        if hasattr(self.model, 'render'):
            outputs = self.model.render(**inputs, **vis_kwargs, info=info)
        else:
            outputs = self.model(**inputs, **vis_kwargs, info=info)
        self.logger.debug(f'outputs: {utils.show_shape(outputs)}')
        images = outputs['images'] if 'images' in outputs else None
        images_c = outputs['images_c'] if 'images_c' in outputs else None

        cat_dim = 0 if self.train_db.aspect > 1. else 1
        if images[0] is not None:
            img_pred = images[..., :3].cpu()
            img_gt = targets['images'][..., :3].cpu()
            if img_pred.ndim == 4:
                assert img_pred.shape[0] == 1
                img_pred = img_pred[0]
            image_list = [img_pred, img_gt, (img_pred - img_gt).abs()]
            if images_c is not None:
                image_list.append(images_c[0, ..., :3].cpu())
            image = torch.cat(image_list, dim=cat_dim)
            utils.save_image(self.output.joinpath('vis', f"img_{self.step}_{index}.png"), image)
        self.model.train()

    def load_model(self):
        if not self.cfg.load or self.cfg.resume:
            return
        if self.cfg.load.suffix == '.ply':
            self.model.load_ply(self.cfg.load)
            logging.warning(f"Load ply from {self.cfg.load}")
        else:
            if not self.cfg.load or self.cfg.resume:
                return
            self.logger.info('==> Loading model from {}, strict: {}'.format(self.cfg.load, not self.cfg.load_no_strict))
            loaded_state_dict = torch.load(self.cfg.load, map_location=torch.device("cpu"))
            loaded_state_dict = utils.convert_pth(loaded_state_dict, **self.cfg.load_cfg)
            return self.model.load_state_dict(loaded_state_dict, strict=not self.cfg.load_no_strict)

    def save_model(self, name="model.pth", net=None):
        if name.endswith('.ply'):
            self.model.save_ply(self.output.joinpath(name).with_suffix('.ply'))
            self.logger.info(f"save model to {self.output.joinpath(name).with_suffix('.ply')}")
        else:
            path = os.path.join(self.output, name)
            data = (self.model if net is None else net).state_dict()
            data = utils.state_dict_strip_prefix_if_present(data, "module.")
            self.logger.info(f"Saving model to {path}")
            if torch.__version__ >= '1.6.0':
                torch.save(data, path, _use_new_zipfile_serialization=True)
            else:
                torch.save(data, path)

    @property
    def mode(self):
        if self.cfg.test:
            return "test"
        if self.cfg.eval:
            return "eval"
        return "train"

    @property
    def training(self):
        return not (self.cfg.test or self.cfg.eval)

    def store(self, name: str, attr: str = None):
        """
        Store and resume the attribution <name>
        if <name> can "load_state_dict" and "state_dict", using them. (For model, optimizer, lr_scheduler)
        """
        self.checkpoint_manager.store(name, self, attr)

    def resume(self, name, default=None):
        """
        Load <name> item form checkpoint
        """
        value = self.checkpoint_manager.resume(name)
        if value is None:
            return default
        self.logger.info("==> Resume `{}`={}".format(name, value))
        return value

    def set_output_dir(self, *path, log_filename=None):
        if self.cfg.output_dir is not None:
            self.output = Path(self.cfg.output_dir)
        else:
            if len(path) == 0:
                if hasattr(self.cfg, 'exp_name'):
                    path = [self.cfg.exp_name]
                else:
                    path = [f"{self.model_name}_{self.model_cfg}".strip('_')]
            self.output = Path(self.cfg.output).joinpath(*path, str(self.cfg.log_suffix))
        self.output = self.output.expanduser()
        self.output.mkdir(exist_ok=True, parents=True)
        self.checkpoint_manager.set_save_dir(self.output)

    def execute_hooks(self, hook_type='before_train_epoch'):
        getattr(self.hook_manager, hook_type)()

    def add_hooks(self, f: Callable, hook_type='before_train_epoch', *args, **kwargs):
        self.hook_manager.add_hook(f, hook_type, *args, **kwargs)

    def enable_model_hooks(self, *modules: nn.Module):
        num_add_hook = self.hook_manager.add_module_hooks(*modules)
        if num_add_hook > 0:
            self.logger.info(f'==> Add {num_add_hook} module hooks')

    def _set_now_state(self, step=None, epoch=None, is_during_training=None):
        if step is not None:
            self.step = step
        if epoch is not None:
            self.epoch = epoch
        if is_during_training is not None:
            self.is_during_training = is_during_training


if __name__ == '__main__':
    GaussianTrainTask().run()
