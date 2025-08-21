import argparse
import math
from pathlib import Path
from typing import Union, Callable

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from lietorch import SE3
from torch import Tensor

import datasets
import utils
from fast_2d_gs.network import FastSuperpoint2DGaussianSplatting
from fast_2d_gs.renderer.gaussian_render_origin import render_gs_official
from utils import ops_3d

D_NERF_SCENES = ['bouncingballs', 'hellwarrior', 'hook', 'jumpingjacks', 'lego', 'mutant', 'standup', 'trex']


# noinspection PyArgumentList
class Fast_SP_GS_GUI:
    net: FastSuperpoint2DGaussianSplatting

    def __init__(self):
        utils.my_logger.basic_config()
        self.net, self.db, init_stage = self.build_dataset_and_net()
        # self.image_index = 0
        self.camera_index = 0
        self.mean_point_scale = self.net.get_scaling.mean()
        self.sp_colors = None
        if hasattr(self.net, 'num_superpoints'):
            self.sp_colors = utils.get_colors(self.net.num_superpoints).cuda().float()
        self.net.gs_rasterizer = render_gs_official
        self.device = torch.device('cuda')
        self.joint_color = torch.tensor([[1., 0, 0], [1., 0, 0]], device=self.device)
        self.now_pose = torch.zeros(self.net.num_superpoints, 3, device=self.device)
        self.now_joints = None
        self.saved_videos = []

        dpg.create_context()
        dpg.create_viewport(
            title='Superpoint Gaussian Splatting',
            width=self.db.image_size[0],
            height=self.db.image_size[1]
        )

        self.is_vary_time = False
        self.is_vary_view = False
        self.is_vary_pose = False
        self.is_save_video = False

        dpg.push_container_stack(dpg.add_window(tag='Primary Window'))
        self.viewer = Viewer3D(self.rendering, size=self.db.image_size, no_resize=False, no_move=True)
        with (dpg.window(tag='control', label='FPS:', collapsed=False, no_close=True, width=256, height=800)):
            with dpg.collapsing_header(label='camears', default_open=False):
                self.control_camera()
            with dpg.collapsing_header(label='render', default_open=True):
                self.control_render(init_stage)
            with dpg.collapsing_header(label='show', default_open=True):
                self.control_show()
            with dpg.collapsing_header(label='joints', default_open=True):
                self.control_joint()
        dpg.pop_container_stack()
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(callback=self.viewer.callback_mouse_drag)
            dpg.add_mouse_wheel_handler(callback=self.viewer.callback_mouse_wheel)
            dpg.add_mouse_release_handler(callback=self.viewer.callback_mouse_release)
            # dpg.add_mouse_wheel_handler(callback=self.callback_mouse_wheel)
            dpg.add_mouse_move_handler(callback=self.callback_mouse_hover)
            dpg.add_mouse_click_handler(callback=self.callback_mouse_click)
            # dpg.add_key_press_handler(callback=self.callback_keypress)
        self.change_image_index()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        # dpg.set_primary_window(self.viewer.win_tag, True)
        dpg.set_primary_window("Primary Window", True)
        # dpg.start_dearpygui()

    def control_camera(self):
        # dpg.add_text(tag='fps')
        with dpg.group(horizontal=True):
            dpg.add_text('fovy')
            dpg.add_slider_float(
                min_value=15.,
                max_value=180.,
                default_value=math.degrees(self.viewer.fovy),
                callback=lambda *args: self.viewer.set_fovy(dpg.get_value('set_fovy')),
                tag='set_fovy'
            )
        with dpg.group():
            item_width = 50
            with dpg.group(horizontal=True):
                dpg.add_text('eye')
                dpg.add_input_float(tag='eye_x', step=0, width=item_width)
                dpg.add_input_float(tag='eye_y', step=0, width=item_width)
                dpg.add_input_float(tag='eye_z', step=0, width=item_width)
            with dpg.group(horizontal=True):
                dpg.add_text('at ')
                dpg.add_input_float(tag='at_x', step=0, width=item_width)
                dpg.add_input_float(tag='at_y', step=0, width=item_width)
                dpg.add_input_float(tag='at_z', step=0, width=item_width)

            def change_eye(*args):
                print('change camera position', args)
                self.viewer.eye = self.viewer.eye.new_tensor([dpg.get_value(item) for item in
                                                              ['eye_x', 'eye_y', 'eye_z']])
                self.viewer.at = self.viewer.at.new_tensor([dpg.get_value(item) for item in
                                                            ['at_x', 'at_y', 'at_z']])
                self.viewer.need_update = True

            dpg.add_button(label='change', callback=change_eye)

    def change_image_index(self, *args, **kwargs):
        if self.db is None:
            return
        image_index = self.image_index = dpg.get_value('img_id') % len(self.db)
        dpg.set_value('img_id', image_index)
        camera_id = self.db.camera_ids[image_index] if getattr(self.db, 'num_cameras', -1) > 0 else image_index

        Tw2v = self.db.Tw2v[camera_id].cpu()
        Tw2v = ops_3d.convert_coord_system_matrix(Tw2v, self.db.coord_dst, ops_3d.get_coord_system())
        self.viewer.set_pose(Tw2v=Tw2v)
        if hasattr(self.db, 'times') and self.db.times is not None:
            dpg.set_value('time', self.db.times[image_index].item())
        self.viewer.set_fovy(math.degrees((self.db.FoV[camera_id] if self.db.FoV.ndim == 2 else self.db.FoV)[1].item()))
        self.viewer.resize(self.db.image_size[0], self.db.image_size[1])
        print('change_image_index:', image_index, camera_id)

    def control_render(self, init_stage=None):
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label='offical rasterizer',
                             default_value=getattr(self.net, 'use_official_gaussians_render', True),
                             tag='official_rasterizer', callback=self.viewer.set_need_update)
        with dpg.group(horizontal=True):
            dpg.add_text('white background')
            dpg.add_checkbox(tag='bg_white', callback=self.viewer.set_need_update)
        with dpg.group(horizontal=True, show=hasattr(self.db, 'times') and self.db.times is not None):
            dpg.add_slider_float(label='t', tag='time', max_value=1.0, callback=self.viewer.set_need_update)

            def vary_time():
                self.is_vary_time = not self.is_vary_time
                if dpg.get_value('save_video'):
                    self.is_save_video = True
                    self.saved_videos = []
                    dpg.configure_item('save_video', label='(0)')

            dpg.add_button(label='A', callback=vary_time)

            # set camera by image_index
        with dpg.group(horizontal=True):
            dpg.add_input_int(label='img_id',
                              tag='img_id',
                              min_value=0,
                              min_clamped=True,
                              max_value=len(self.db) - 1,
                              max_clamped=True,
                              step=1,
                              callback=self.change_image_index
                              )

        with dpg.group(horizontal=True):
            dpg.add_text('cmp')
            dpg.add_radio_button(items=['no', 'GT', "blend", "error"], tag='cmp_GT',
                                 callback=self.viewer.set_need_update, horizontal=True, default_value='no')
        # interpolate two camera
        with dpg.group(horizontal=True):
            dpg.add_text('view_int')
            dpg.add_checkbox(tag='iterp_view', callback=self.vary_view)
            dpg.add_input_int(
                tag='img_id_1',
                max_value=len(self.db.images) - 1,
                max_clamped=True,
                min_clamped=True,
                step=0,
                width=50,
                callback=self.vary_view
            )
            dpg.add_input_int(
                tag='img_id_2',
                max_value=len(self.db.images) - 1,
                max_clamped=True,
                min_clamped=True,
                step=0,
                width=50,
                callback=self.vary_view
            )

            def set_random_two_image_id():
                img_id_1, img_id_2 = np.random.choice(len(self.db.images), 2).tolist()
                dpg.set_value('img_id_1', img_id_1)
                dpg.set_value('img_id_2', img_id_2)
                self.vary_view()

            dpg.add_button(label='R', callback=set_random_two_image_id)
        with dpg.group(horizontal=True):
            dpg.add_slider_float(tag='view_t', min_value=0, max_value=1, width=150, callback=self.vary_view)
            dpg.add_input_int(tag='view_speed', default_value=120, width=50, step=0)

            def switch_vary_view():
                self.is_vary_view = not self.is_vary_view
                print('switch vary view:', self.is_vary_view)

            dpg.add_button(label='A', tag='A', callback=switch_vary_view)

        def set_rotate_index_limit():
            dpg.configure_item('rotate_index', max_value=dpg.get_value('rotate_total'))
            dpg.set_value('rotate_index', 0)
            self.viewer.set_need_update()

        with dpg.group(horizontal=True):
            dpg.add_text('Rotate Obj: auto')
            dpg.add_checkbox(tag='rotate_auto', callback=self.viewer.set_need_update)
            dpg.add_button(tag='roate_reset', label='R', callback=set_rotate_index_limit)
        with dpg.group(horizontal=True):
            dpg.add_slider_int(tag='rotate_index', callback=self.viewer.set_need_update, width=100, max_value=360)
            dpg.add_text('/')

            dpg.add_input_int(tag='rotate_total',
                              step=0,
                              default_value=360,
                              min_value=10,
                              min_clamped=True,
                              width=50,
                              callback=set_rotate_index_limit)
        with dpg.group(horizontal=True):
            dpg.add_text('stage:')
            stages = []
            for _, stage in reversed(self.net.train_schedule):
                if stage != 'canonical' and stage not in stages:
                    stages.append(stage)
            if hasattr(self.net, 'train_schedule'):
                dpg.add_combo(
                    items=stages,
                    default_value=stages[0] if init_stage is None else init_stage,
                    tag='stage',
                    callback=self.viewer.set_need_update
                )

    def control_show(self):
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text('show')
            dpg.add_text('size:')
            dpg.add_slider_float(tag='point_size',
                                 min_value=0,
                                 max_value=2.0,
                                 default_value=1.0,
                                 width=100,
                                 callback=self.viewer.set_need_update
                                 )
        with dpg.group(horizontal=True):
            dpg.add_text('points')
            dpg.add_checkbox(tag='show_points', callback=self.viewer.set_need_update)
            dpg.add_text('superpoints')
            dpg.add_checkbox(tag='show_superpoints', callback=self.viewer.set_need_update)
            dpg.add_text('2D')
            dpg.add_checkbox(tag='show_sp_2D', callback=self.viewer.set_need_update)
        with dpg.group(horizontal=True):
            dpg.add_text('point_sp')
            dpg.add_checkbox(tag='show_p2sp', callback=self.viewer.set_need_update)

            dpg.add_text('skeleton 2D')
            dpg.add_checkbox(tag='show_skeleton_2D', callback=self.viewer.set_need_update)

        with dpg.group(horizontal=True):
            def save_image(sender, app_data):
                utils.save_image(app_data['file_path_name'], self.viewer.data)
                print('save image to', app_data['file_path_name'])

            with dpg.file_dialog(directory_selector=False,
                                 show=False,
                                 callback=save_image,
                                 id="save_file_dialog_id",
                                 default_filename=self.db.scene,
                                 width=700,
                                 height=400):
                dpg.add_file_extension(".jpg", color=(150, 255, 150, 255))
                dpg.add_file_extension(".png", color=(255, 150, 150, 255))

            # dpg.add_button(label='save_image', callback=save_image)
            dpg.add_button(label='save_image', callback=lambda: dpg.show_item('save_file_dialog_id'))

            def save_video(sender, app_data):
                videos = np.stack(self.saved_videos, axis=0)
                save_path = Path(app_data['file_path_name'])
                utils.save_mp4(save_path, videos)
                self.saved_videos = []
                dpg.configure_item('save_video', label='save video')
                print(f"save videos {videos.shape} to {save_path}")

            with dpg.file_dialog(directory_selector=False,
                                 show=False,
                                 callback=save_video,
                                 id="save_video_dialog_id",
                                 default_filename=self.db.scene,
                                 width=700,
                                 height=400):
                dpg.add_file_extension(".mp4", color=(150, 255, 150, 255))

            def save_video_callback():
                if self.is_save_video:
                    self.is_save_video = False
                    dpg.show_item('save_video_dialog_id')
                    dpg.configure_item('save_video', label='save video')
                else:
                    self.is_save_video = True
                    self.saved_videos = []
                    dpg.configure_item('save_video', label='save video (0)')
                self.viewer.set_need_update()

            dpg.add_button(label='save video', tag='save_video', callback=save_video_callback)
            dpg.add_input_int(tag='save_video_num', default_value=120, step=0, width=40)

    def vary_view(self):
        if self.db is None or not dpg.get_value('iterp_view'):
            return
        idx1, idx2 = dpg.get_value('cam_id_1'), dpg.get_value('cam_id_2')
        view_1 = ops_3d.rigid.Rt_to_lie(self.db.Tw2v[idx1])
        view_2 = ops_3d.rigid.Rt_to_lie(self.db.Tw2v[idx2])
        t = dpg.get_value('view_t')
        Tw2v = ops_3d.rigid.lie_to_Rt(view_1 * (1 - t) + view_2 * t).cpu()
        Tw2v = ops_3d.convert_coord_system_matrix(Tw2v, self.db.coord_dst, ops_3d.get_coord_system())
        self.viewer.set_pose(Tw2v=Tw2v)
        self.viewer.set_fovy(math.degrees(self.db.FoV[idx1, 1] if self.db.FoV.ndim == 2 else self.db.FoV[1]))
        self.viewer.resize(self.db.image_size[0], self.db.image_size[1])
        self.viewer.need_update = True

    def control_joint(self):
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text('show sp1')
            dpg.add_checkbox(tag='show_sp1', callback=self.viewer.set_need_update)
            dpg.add_input_int(tag='sp_1', min_value=0,
                              max_value=self.net.num_superpoints - 1, width=100, callback=self.viewer.set_need_update)
        with dpg.group(horizontal=True):
            dpg.add_text('show sp2')
            dpg.add_checkbox(tag='show_sp2', callback=self.viewer.set_need_update)
            dpg.add_input_int(tag='sp_2',
                              min_value=0,
                              max_value=self.net.num_superpoints - 1,
                              width=100,
                              callback=self.viewer.set_need_update)
        with dpg.group(horizontal=True):
            def set_joint():
                i = dpg.get_value('joint_idx') % self.net.num_superpoints
                if i < 0:
                    i += self.net.num_superpoints
                dpg.set_value('joint_idx', i)
                j = self.net.joint_parents[i, 0].item()
                if j >= 0:
                    dpg.set_value('sp_1', i)
                    dpg.set_value('sp_2', j)
                self.viewer.set_need_update()

            dpg.add_text('show joint')
            dpg.add_checkbox(tag='show_joint', callback=set_joint)
            dpg.add_input_int(
                tag='joint_idx',
                default_value=0,
                min_value=0,
                max_value=self.net.num_superpoints - 1,
                width=100,
                callback=set_joint
            )
            dpg.add_text('', tag='now_joint')
        with dpg.group(horizontal=True):
            def load_pose(sender, app_data):
                filepath = app_data['file_path_name']
                self.now_pose = torch.from_numpy(np.loadtxt(filepath, delimiter=',')).to(self.now_pose)
                print(f'load pose from {filepath}')

            def save_pose(sender, app_data):
                filepath = app_data['file_path_name']

                np.savetxt(filepath, self.now_pose.cpu().numpy(), delimiter=',')
                print(f'save pose to {filepath}')

            with dpg.file_dialog(directory_selector=False,
                                 show=False,
                                 callback=load_pose,
                                 id="load_pose_dialog",
                                 default_filename=self.db.scene,
                                 width=700,
                                 height=400):
                dpg.add_file_extension(".pose", color=(150, 255, 150, 255))

            with dpg.file_dialog(directory_selector=False,
                                 show=False,
                                 callback=save_pose,
                                 id="save_pose_dialog",
                                 default_filename=self.db.scene,
                                 width=700,
                                 height=400):
                dpg.add_file_extension(".pose", color=(150, 255, 150, 255))
            dpg.add_button(label='load', tag='load_pose', callback=lambda: dpg.show_item('load_pose_dialog'))
            dpg.add_button(label='save', tag='save_pose', callback=lambda: dpg.show_item('save_pose_dialog'))

            def reset_pose():
                self.now_pose.zero_()
                self.viewer.set_need_update()

            dpg.add_button(label='reset', tag='reset_pose', callback=reset_pose)
        with dpg.group(horizontal=True, show=hasattr(self.db, 'times') and self.db.times is not None):
            dpg.add_slider_float(label='t', tag='time_pose', max_value=1.0, callback=self.viewer.set_need_update)

            def vary_pose():
                self.is_vary_pose = not self.is_vary_pose
                if dpg.get_value('save_video'):
                    self.is_save_video = True
                    self.saved_videos = []
                    dpg.configure_item('save_video', label='(0)')

            dpg.add_button(label='A', callback=vary_pose)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label='enable', tag='joint_rot', callback=self.viewer.set_need_update)

            def set_pose():
                scale = math.radians(dpg.get_value('joint_rot_scale'))
                R = ops_3d.rotate(
                    dpg.get_value('joint_rot_x') * scale,
                    dpg.get_value('joint_rot_y') * scale,
                    dpg.get_value('joint_rot_z') * scale,
                    device=self.device
                )
                jid = dpg.get_value('joint_idx') % self.net.num_superpoints
                self.now_pose[jid] += ops_3d.rotation.R_to_lie(R)
                print(f'set joint {jid}')
                dpg.set_value('joint_rot_x', 0)
                dpg.set_value('joint_rot_y', 0)
                dpg.set_value('joint_rot_z', 0)
                self.viewer.set_need_update()

            dpg.add_button(label='set', tag='set_pose', callback=set_pose)
        for name in ['x', 'y', 'z']:
            dpg.add_slider_float(
                label=f'{name}',
                tag=f'joint_rot_{name}',
                min_value=-1,
                max_value=1,
                callback=self.viewer.set_need_update
            )
        dpg.add_slider_float(
            label=f'scale',
            tag=f'joint_rot_scale',
            min_value=0,
            max_value=360,
            default_value=45,
            callback=self.viewer.set_need_update
        )
        # dpg.add_checkbox(label='joint stage', tag='stage_joint', callback=self.viewer.set_need_update)

    def options(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default='results/SP_GS/joint/jumpingjacks/best.pth')
        parser.add_argument('-i', '--load', type=str, default='./exps/sp_gs/dnerf.yaml')
        parser.add_argument('-s', '--scene', type=str, default=None)
        parser.add_argument('--stage', type=str, default=None)
        parser.add_argument('--split', default='train')
        args = parser.parse_args()
        return args

    def build_dataset_and_net(self):
        args = self.options()
        pth_path = Path(args.load)
        assert args.load and pth_path.exists(), f"must load a model/checkpoint"
        cfg_path = args.config
        # scene = Path(pth_path).parts[-2]
        scene = args.scene
        if scene is None:
            for scene in D_NERF_SCENES:
                if scene in Path(pth_path).parts:
                    break
        print('scene:', scene)
        assert scene is not None

        pth = torch.load(pth_path, map_location='cpu')
        if 'model' in pth:  # checkpoints
            pth = pth['model']
        parser = argparse.ArgumentParser()
        utils.config.options(parser)
        utils.add_cfg_option(parser, '--arch-cfg', help="The configure for networks")
        datasets.options(parser)
        utils.my_logger.options(parser)
        cfg = utils.config.make(
            [f'-c={cfg_path}', f'--scene={scene}', '--no-log'], ignore_unknown=True, ignore_warning=True, parser=parser
        )
        print(cfg)
        db: datasets.DNerfDataset.DNeRFDataset = datasets.make(cfg, 'train')  # noqa
        print(db)
        net = make(cfg)  # type: SuperpointSkeletonGaussianSplatting # noqa
        # print(net)
        net.set_from_dataset(db)
        net.load_state_dict(pth, strict=False)
        print(net)
        print(f'There are {net.points.shape[0]} Gaussians')
        print(f'There are {net.num_superpoints} superpoints')
        print(f'There are {net.num_frames} frames in train dataset')
        print(f'Root is {net.joint_root}')
        net.eval()
        net.cuda()

        if args.split != 'train':
            db: datasets.MyTreeSegDataset.MyTreeSegDataset = datasets.make(cfg, args.split)  # noqa
            print(db)
        return net, db, args.stage

    @torch.no_grad()
    def rendering(self, Tw2v, fovy, size):
        Tw2v = Tw2v.cuda()
        if dpg.get_value('official_rasterizer'):
            Tw2v = ops_3d.convert_coord_system(
                Tw2v, ops_3d.get_coord_system(), self.db.coord_dst if self.db is not None else 'opencv')
            Tv2c = ops_3d.opencv.perspective(fovy, size=self.db.image_size).cuda()
        else:
            Tv2c = ops_3d.perspective(fovy, size=self.db.image_size).cuda()

        Tw2c = Tv2c @ Tw2v
        Tv2w = torch.inverse(Tw2v)
        t = torch.tensor([dpg.get_value('time')]).cuda()
        info = {
            'Tw2c': Tw2c,
            'Tw2v': Tw2v,
            'campos': Tv2w[:3, 3],
            'size': self.db.image_size,
            'index': self.image_index,
            'fov_xy': (ops_3d.fovx_to_fovy(fovy, size[1] / size[0]), fovy),
        }

        if dpg.get_value('official_rasterizer'):
            from diff_gaussian_rasterization import GaussianRasterizationSettings as Settings
            raster_settings = Settings(
                image_width=info['size'][0],
                image_height=info['size'][1],
                tanfovx=math.tan(0.5 * info['fov_xy'][0]),
                tanfovy=math.tan(0.5 * info['fov_xy'][1]),
                scale_modifier=1.0,
                viewmatrix=info['Tw2v'].view(4, 4).transpose(-1, -2),
                projmatrix=info['Tw2c'].view(4, 4).transpose(-1, -2),
                sh_degree=self.net.max_sh_degree,
                campos=info['campos'],
                prefiltered=False,
                debug=False,
                bg=torch.full((3,), 1. if dpg.get_value('bg_white') else 0.).cuda()
            )
            gs_rasterizer = render_gs_official
        else:
            from fast_2d_gs.renderer.gaussian_render import GaussianRasterizationSettings, render
            raster_settings = GaussianRasterizationSettings(
                image_width=info['size'][0],
                image_height=info['size'][1],
                tanfovx=math.tan(0.5 * info['fov_xy'][0]),
                tanfovy=math.tan(0.5 * info['fov_xy'][1]),
                scale_modifier=1.0,
                viewmatrix=info['Tw2v'].view(4, 4),  # .transpose(-1, -2),
                projmatrix=info['Tw2c'].view(4, 4),  # .transpose(-1, -2),
                sh_degree=self.net.max_sh_degree,
                campos=info['campos'],
                prefiltered=False,
                debug=False,
                detach_other_extra=False
            )
            gs_rasterizer = render

        kwargs = {}
        if hasattr(self.net, 'train_schedule'):
            stage = kwargs['stage'] = dpg.get_value('stage')
        else:
            stage = None
        if 'sk' in stage:
            kwargs['sk_r_delta'] = self.now_pose * dpg.get_value('time_pose')
            if dpg.get_value('joint_rot'):
                scale = math.radians(dpg.get_value('joint_rot_scale'))
                R = ops_3d.rotate(
                    dpg.get_value('joint_rot_x') * scale,
                    dpg.get_value('joint_rot_y') * scale,
                    dpg.get_value('joint_rot_z') * scale,
                    device=self.device
                )
                jid = dpg.get_value('joint_idx') % self.net.num_superpoints
                if jid != dpg.get_value('joint_idx'):
                    dpg.set_value('joint_idx', jid)
                kwargs['sk_r_delta'][jid] += ops_3d.rotation.R_to_lie(R)

        rotate_angle = 2.0 * torch.pi * dpg.get_value('rotate_index') / dpg.get_value('rotate_total')
        R = ops_3d.rotate_z(rotate_angle).cuda()
        net_out = self.net(t=t, campos=info['campos'], **kwargs)
        device = torch.device('cuda')
        M = getattr(self.net, 'num_superpoints', 0)
        if self.net.sk_is_init and hasattr(self.net, 'sk_W'):
            W = self.net.sk_W
        else:
            index = torch.arange(self.net.sp_knn.shape[0], device=device)[:, None].expand_as(self.net.sp_knn)
            W = torch.zeros((len(self.net.points), M), device=device)
            W[index, self.net.sp_knn] = self.net.sp_weights

        net_out['points'] = ops_3d.xfm(net_out['points'], R)
        net_out['rotations'] = ops_3d.quaternion.mul(ops_3d.rotation.R_to_quaternion(R)[None], net_out['rotations'])
        net_out['colors'] = ops_3d.SH_to_RGB(
            net_out.pop('sh_features'), net_out['points'], info['campos'], self.net.active_sh_degree, clamp=True)
        point_scale = 10 ** ((dpg.get_value('point_size') * 0.5 - 1) * 2)  # [1e-2, 1.]
        if dpg.get_value('show_points'):
            net_out['scales'] = torch.full_like(net_out['scales'], self.mean_point_scale * point_scale).float()

        sp1 = dpg.get_value('sp_1') % self.net.num_superpoints
        sp2 = dpg.get_value('sp_2') % self.net.num_superpoints
        if dpg.get_value('show_p2sp'):
            net_out['colors'] = torch.sum(self.sp_colors * W[..., None], dim=1)

        sp_xyz = self.xfm_superpoints(self.net.sp_points, net_out)
        sp_xyz = ops_3d.apply(sp_xyz, R)
        if dpg.get_value('show_superpoints'):
            net_out = self.add_gaussians(
                net_out,
                points=sp_xyz,
                colors=self.sp_colors,
                scales=self.mean_point_scale * point_scale * 5.,
                replace=not dpg.get_value('show_points')
            )
        elif dpg.get_value('show_sp1') or dpg.get_value('show_sp2'):
            w = 0
            if dpg.get_value('show_sp1'):
                w = W[:, sp1:sp1 + 1]
            if dpg.get_value('show_sp2') and not (dpg.get_value('show_sp1') and sp1 == sp2):
                w = w + W[:, sp2:sp2 + 1]
            net_out['opacity'] *= w

            if dpg.get_value('show_joint'):
                joints = []
                if self.net.sk_is_init:
                    joints.append(self.xfm_superpoints(ops_3d.apply(self.net.sp_points[sp1], R), net_out, sp1))
                    joints.append(self.xfm_superpoints(ops_3d.apply(self.net.sp_points[sp2], R), net_out, sp2))
                else:
                    joints.append(self.xfm_superpoints(ops_3d.apply(self.net.joint_pos[sp1][sp2], R), net_out, sp1))
                    joints.append(self.xfm_superpoints(ops_3d.apply(self.net.joint_pos[sp2][sp1], R), net_out, sp2))
                net_out = self.add_gaussians(
                    net_out,
                    points=torch.stack(joints),
                    colors=self.joint_color,
                    scales=self.mean_point_scale * point_scale * 10.0
                )
                # p2sp = torch.cat([p2sp, p2sp.new_tensor([sp2, sp1])], dim=0)
                # if dpg.get_value('joint_rot'):
                #     R = ops_3d.rotate(
                #         dpg.get_value('joint_rot_x') * torch.pi * 2.,
                #         dpg.get_value('joint_rot_y') * torch.pi * 2.,
                #         dpg.get_value('joint_rot_z') * torch.pi * 2.,
                #         device=self.device
                #     )
                #     mask = p2sp == sp1
                #     net_out['points'][mask] = ops_3d.xfm(net_out['points'][mask] - joint1, R) + joint1

                net_out = self.add_lines(net_out, sp_xyz[sp1:sp1 + 1], sp_xyz[sp2:sp2 + 1])

        elif dpg.get_value('show_joint'):
            mask = self.net.joint_parents[:, 0] >= 0
            if self.net.sk_is_init and 'sk' not in stage:
                joint = self.xfm_superpoints(self.net.sp_points, net_out)
            elif self.net.joint_is_init:
                ja = torch.arange(M, device=self.device)[mask]
                jb = self.net.joint_parents[:, 0][mask]
                # joint = self.net.joint_pos[ja, jb]
                joint = self.xfm_superpoints(self.net.joint_pos[ja, jb], net_out, jb)
                # joint2 = self.xfm_superpoints(self.net.joint_pos[ja, jb], net_out, ja)
                # joint = (joint + joint2) * 0.5
            else:
                joint = None
            if joint is not None:
                joint = ops_3d.apply(joint, R)
                net_out = self.add_gaussians(
                    net_out,
                    points=joint,
                    colors=joint.new_tensor([1., 0, 0.]),
                    scales=self.mean_point_scale * point_scale * 10.0
                )
                # root = torch.nonzero(torch.logical_not(mask)).item()
                root = self.net.joint_root.item()
                jb = self.net.joint_parents[:, 0]
                jb = jb[jb >= 0]
                jb = torch.where(jb.ge(root), jb - 1, jb)
                # net_out = self.add_lines(net_out, joint, joint[jb])
                # print(net_out['points'])

        render_out_f = gs_rasterizer(**{k: v for k, v in net_out.items() if not k.startswith('_')},
                                     raster_settings=raster_settings)

        images = torch.permute(render_out_f['images'], (1, 2, 0)).contiguous()
        # background = torch.rand_like(images)
        # background: Tensor = None
        # if background is not None:
        #     images = images + (1 - render_out_f['opacity'][..., None]) * background.squeeze(0)
        cmp_GT = dpg.get_value('cmp_GT')
        if cmp_GT != 'no':
            gt_img = self.db.get_image(dpg.get_value('img_id'))[..., :3].to(images.device)
            if gt_img.dtype == torch.uint8:
                gt_img = gt_img.float() / 255.
            if cmp_GT == 'GT':
                images = gt_img
            elif cmp_GT == 'blend':
                images = torch.lerp(images, gt_img, 0.5)
            else:
                images[:] = torch.abs(images - gt_img)  # .mean(dim=-1, keepdim=True)
        self.now_joints = None
        images = self.draw_skeleton(images, stage, net_out, R, Tw2c, size, M)
        if dpg.get_value('show_sp_2D'):
            images = self.draw_superpoints_2D(images, stage, net_out, R, Tw2c, size, M)

        return images

    def xfm_superpoints(self, points, net_out, mask=None, stage='static'):
        if '_sk_tr' in net_out:
            T = SE3.exp(net_out['_sk_tr'])
        elif '_spT' in net_out:
            T = SE3.InitFromVec(net_out['_spT'])
        elif '_skT' in net_out:
            T = SE3.InitFromVec(net_out['_skT'])
        elif '_sp_tr' in net_out:
            T = SE3.exp(net_out['_sp_tr'])
        else:
            T = None
        if T is not None and mask is not None:
            T = T[mask]
        if T is None:
            if stage != 'static':
                print('sp points is not transformed!!')
        else:
            points = T.act(points)
        return points

    def draw_superpoints_2D(self, images, stage, net_out, R, Tw2c, size, M):
        sp_points = self.net.sp_points
        if stage != 'static':
            sp_points = self.xfm_superpoints(sp_points, net_out)
        sp_points = ops_3d.xfm(sp_points, R)
        sp_points = ops_3d.xfm(sp_points, Tw2c, homo=True)
        sp_points = ((sp_points[:, :2] / sp_points[:, -1:] + 1) * sp_points.new_tensor(size) - 1) * 0.5
        sp_points = sp_points.cpu().numpy().astype(np.int32)
        images = np.ascontiguousarray(utils.as_np_image(images))
        for j in range(M):
            images = cv2.circle(images, sp_points[j].tolist(), radius=3, color=(0, 0, 255), thickness=-1)
        return images

    def draw_skeleton(self, images, stage, net_out, R, Tw2c, size, M):
        if not dpg.get_value('show_skeleton_2D'):
            return images
        if not self.net.sp_is_init:
            return images
        images = utils.as_np_image(images)
        a, b, mask = self.net.joint_pair
        if self.net.sk_is_init:
            joint = self.net.sp_points
        else:
            joint = self.net.sp_points.clone()
            joint[mask] = self.net.joint_pos[a, b]
        if stage != 'static':
            joint = self.xfm_superpoints(joint, net_out)
        joint = ops_3d.apply(joint, R)
        joint = ops_3d.apply(joint, Tw2c, homo=True)
        joint = ((joint[:, :2] / joint[:, -1:] + 1) * joint.new_tensor(size) - 1) * 0.5
        joint = joint.cpu().numpy().astype(np.int32)
        self.now_joints = joint
        images = np.ascontiguousarray(images)
        for j in range(M):
            if mask[j]:
                c = (0, 255, 0)
                if dpg.get_value('show_joint'):
                    if j == dpg.get_value('joint_idx'):
                        c = (255, 255, 0)
                else:
                    if str(j) == dpg.get_value('now_joint'):
                        c = (255, 255, 0)
                images = cv2.circle(images, joint[j].tolist(), radius=3, color=c, thickness=-1)
        for j in range(len(a)):
            images = cv2.line(images, joint[a[j]], joint[b[j]], color=(255, 0, 0), thickness=1)
        return images

    def add_lines(self, net_out, line_p1: Tensor, line_p2: Tensor, line_width=1.0):
        points = (line_p1 + line_p2) / 2
        scales = torch.ones_like(points) * line_width * self.mean_point_scale
        dist = torch.pairwise_distance(line_p1, line_p2)
        scales[:, 0] = dist / 6

        if torch.all(line_p1 == line_p2):
            rotations = None
        else:
            v = line_p2 - points
            rotations = ops_3d.rotation.direction_vector_to_quaternion(v.new_tensor([[1, 0, 0]]), v)
            rotations = ops_3d.quaternion.normalize(rotations)
        return self.add_gaussians(net_out, points, points.new_tensor([0., 1., 0.]), scales, rotations)

    def add_gaussians(self, net_out, points, colors, scales, rotations=None, opacity=None, replace=False):
        P = points.shape[0]
        colors = colors.view(-1, 3).expand(P, 3)
        if isinstance(scales, Tensor):
            scales = scales.view(-1, scales.shape[-1] if scales.ndim > 0 else 1).expand(P, 3)
        else:
            scales = net_out['scales'].new_tensor([scales]).view(-1, 1).expand(P, 3)
        if rotations is None:
            rotations = net_out['rotations'].new_zeros([P, 4])
            rotations[:, -1] = 1.
        else:
            rotations = rotations.view(-1, 4).expand(P, 4)
        if opacity is None:
            opacity = net_out['opacity'].new_ones([P, 1])
        else:
            opacity = opacity.view(-1, 1).expand(P, 1)
        if replace:
            net_out.update({
                'points': points,
                'colors': colors,
                'scales': scales,
                'rotations': rotations,
                'opacity': opacity
            })
        else:
            net_out['points'] = torch.cat([net_out['points'], points], dim=0)
            net_out['colors'] = torch.cat([net_out['colors'], colors], dim=0)
            net_out['scales'] = torch.cat([net_out['scales'], scales], dim=0)
            net_out['rotations'] = torch.cat([net_out['rotations'], rotations], dim=0)
            net_out['opacity'] = torch.cat([net_out['opacity'], opacity], dim=0)
        return net_out

    def run(self):
        last_size = None
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            if self.is_vary_time:
                if self.is_save_video:
                    t = len(self.saved_videos) / dpg.get_value('save_video_num')
                else:
                    t = dpg.get_value('time')
                    t = t + 0.01
                if t > 1:
                    t = 0.
                dpg.set_value('time', t)
                self.viewer.need_update = True
            if self.is_vary_pose:
                if self.is_save_video:
                    t = len(self.saved_videos) / dpg.get_value('save_video_num')
                else:
                    t = dpg.get_value('time_pose')
                    t = t + 0.01
                if t > 1:
                    t = 0.
                dpg.set_value('time_pose', t)
                self.viewer.need_update = True
            if self.is_vary_view:
                if self.is_save_video:
                    t = len(self.saved_videos) / dpg.get_value('save_video_num')
                else:
                    t = dpg.get_value('view_t')
                    t = t + 1. / dpg.get_value('view_speed')
                if t > 1:
                    t = 0.
                dpg.set_value('view_t', t)
                self.vary_view()

            if dpg.get_value('rotate_auto'):
                if self.is_save_video:
                    rotate_index = len(self.saved_videos) / dpg.get_value('save_video_num')
                    rotate_index = int(rotate_index * dpg.get_value('rotate_total'))
                else:
                    rotate_index = dpg.get_value('rotate_index') + 1
                    if rotate_index >= dpg.get_value('rotate_total'):
                        rotate_index = 0
                dpg.set_value('rotate_index', rotate_index)
                self.viewer.need_update = True
            if self.viewer.need_update:
                dpg.set_value('eye_x', self.viewer.eye[0].item())
                dpg.set_value('eye_y', self.viewer.eye[1].item())
                dpg.set_value('eye_z', self.viewer.eye[2].item())
                dpg.set_value('at_x', self.viewer.at[0].item())
                dpg.set_value('at_y', self.viewer.at[1].item())
                dpg.set_value('at_z', self.viewer.at[2].item())
            self.viewer.update()
            now_size = self.viewer.size
            if last_size != now_size:
                dpg.configure_item('control', pos=(dpg.get_item_width(self.viewer.win_tag), 0))
                dpg.set_viewport_width(dpg.get_item_width(self.viewer.win_tag) + dpg.get_item_width('control'))
                dpg.set_viewport_height(dpg.get_item_height(self.viewer.win_tag))
                last_size = now_size
            dpg.configure_item('control', label=f"FPS: {dpg.get_frame_rate()}")
            if self.is_save_video and len(self.saved_videos) < dpg.get_value('save_video_num'):
                self.saved_videos.append(utils.as_np_image(self.viewer.data).copy())
                dpg.configure_item('save_video', label=f"save video({len(self.saved_videos)})")
        dpg.destroy_context()

    def callback_mouse_click(self, sender, app_data):
        if dpg.is_item_clicked(self.viewer.image_tag):
            if self.now_joints is not None:
                x, y = self.viewer.get_mouse_pos()
                dist = np.linalg.norm(self.now_joints - np.array([x, y]), axis=-1)
                nearest = np.argmin(dist).item()
                if dist[nearest] < 10:
                    dpg.set_value('joint_idx', nearest)
                    dpg.set_value('sp_1', nearest)
                    dpg.set_value('sp_2', self.net.joint_parents[nearest, 0].item())
                self.viewer.set_need_update()

    def callback_mouse_hover(self, sender, app_data):
        if dpg.is_item_hovered(self.viewer.image_tag):
            old = dpg.get_value('now_joint')
            if self.now_joints is not None:
                x, y = self.viewer.get_mouse_pos()
                dist = np.linalg.norm(self.now_joints - np.array([x, y]), axis=-1)
                nearest = np.argmin(dist)
                dpg.set_value('now_joint', f'{nearest}' if dist[nearest] < 10 else '')
            else:
                dpg.set_value('now_joint', '')
            if old != dpg.get_value('now_joint'):
                self.viewer.set_need_update()

    def callback_keypress(self, sender, app_data):
        pass


class ImageViewer:

    def __init__(self, image=None, size=(100, 100), channels=3, pad=0, tag='image', **kwargs) -> None:
        self.pad = pad
        if image is None:
            image = np.ones((size[1], size[0], channels), dtype=np.float32)
        assert image.ndim == 3 and image.shape[-1] in [3, 4]
        self.size = (image.shape[1], image.shape[0])
        self.channels = channels
        assert self.channels in [3, 4]
        self._data = (image.astype(np.float32) / 255) if image.dtype == np.uint8 else image.astype(np.float32)
        self._origin_data = None
        self._can_dynamic_change = False
        self.pad = pad
        self.tag = tag
        with dpg.texture_registry(show=False) as self._registry_id:
            # self.registry_id = registry_id
            self._texture_id = dpg.add_raw_texture(
                self.width,
                self.height,
                default_value=self._data,  # noqa
                format=dpg.mvFormat_Float_rgba if self.channels == 4 else dpg.mvFormat_Float_rgb,
                tag=tag
            )
        W, H = self.size
        self._win_id = dpg.add_window(
            width=W + 2 * self.pad, height=H + 2 * self.pad, no_title_bar=True, no_scrollbar=True, **kwargs
        )
        self._img_id = dpg.add_image(self.tag, width=W, height=H, parent=self._win_id)

        with dpg.theme() as container_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, self.pad, self.pad, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme(self._win_id, container_theme)

        self.resize_with_window()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_image: np.ndarray):
        assert new_image.shape == self._data.shape
        self._data[:] = (new_image / 255. if new_image.dtype == np.uint8 else new_image).astype(np.float32)

    @property
    def origin_data(self):
        if not self._can_dynamic_change:
            return self.data
        if self._origin_data is None:
            self._origin_data = self.data.copy()
        return self._origin_data

    @property
    def win_tag(self):
        return self._win_id

    @property
    def image_tag(self):
        return self._img_id

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def resize_with_window(self):
        def resize_handler(sender):
            H, W = dpg.get_item_height(self._win_id), dpg.get_item_width(self._win_id)
            self.resize(W - 2 * self.pad, H - 2 * self.pad)

        with dpg.item_handler_registry() as hr_id:
            dpg.add_item_resize_handler(callback=resize_handler)
        dpg.bind_item_handler_registry(self._win_id, hr_id)

    def resize(self, W: int = None, H: int = None, channels: int = None):
        W = self.width if W is None else W
        H = self.height if H is None else H
        channels = self.channels if channels is None else channels
        if (W, H) == self.size and channels == self.channels:
            return False
        assert self.channels in [3, 4]
        new_image = np.ones((H, W, channels), dtype=np.float32)
        min_H, min_W, min_c = min(H, self.height), min(W, self.width), min(channels, self.channels)
        new_image[:min_H, :min_W, :min_c] = self.data[:min_H, :min_W, :min_c]
        self._data = new_image
        if self._origin_data is not None:
            new_image = np.ones_like(self.data)
            new_image[:min_H, :min_W, :min_c] = self._origin_data[:min_H, :min_W, :min_c]
            self._origin_data = new_image
        self.channels = channels
        self.size = W, H

        # console.log(f'resize "{self.tag}": W={W}, H={H}')
        dpg.delete_item(self.tag)
        dpg.remove_alias(self.tag)
        dpg.hide_item(self._img_id)  # can not delete old image due to segmentation fault (core dumped)

        self._texture_id = dpg.add_raw_texture(
            W,
            H,
            default_value=self.data,  # noqa
            format=dpg.mvFormat_Float_rgba if self.channels == 4 else dpg.mvFormat_Float_rgb,
            tag=self.tag,
            parent=self._registry_id
        )
        self._img_id = dpg.add_image(self._texture_id, parent=self._win_id)
        dpg.configure_item(self._win_id, width=W + 2 * self.pad, height=H + 2 * self.pad)
        return True

    def update(self, image: Union[np.ndarray, Tensor], resize=False):
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        elif image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=-1)
        if image.shape[-1] not in [3, 4]:
            assert image.shape[0] in [3, 4]
            image = image.transpose((1, 2, 0))
        if resize:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        else:
            self.resize(image.shape[1], image.shape[0], image.shape[2])
        self.data = image
        self._origin_data = None

    def get_mouse_pos(self):
        x, y = dpg.get_mouse_pos(local=False)
        wx, wy = dpg.get_item_pos(self._win_id)
        ix, iy = dpg.get_item_pos(self._img_id)
        return int(x - wx - ix), int(y - wy - iy)

    def enable_dynamic_change(self, hover_callback=None):
        self._can_dynamic_change = True
        if hover_callback is None:
            return
        with dpg.item_handler_registry() as handler:
            dpg.add_item_hover_handler(callback=hover_callback)
        dpg.bind_item_handler_registry(self._img_id, handler)


class Viewer3D(ImageViewer):

    def __init__(self, renderer: Callable, size=(100, 100), pad=0, tag='3d', no_resize=True, no_move=True, **kwargs):
        super().__init__(size=size, pad=pad, tag=tag, no_resize=no_resize, no_move=no_move, **kwargs)

        self.renderer = renderer
        self.fovy = math.radians(60.)
        fov = [ops_3d.fovx_to_fovy(self.fovy, size[1] / size[0]), self.fovy]
        self.Tv2s = ops_3d.camera_intrinsics(size=size, fov=fov)
        self.Ts2v = ops_3d.camera_intrinsics(size=size, fov=fov, inv=True)

        self.up = torch.tensor([0, 1., 0.])
        self.eye = torch.tensor([0., 0., 2.0])
        self.at = torch.tensor([0., 0., 0.])
        #
        self._last_mouse_pos = None
        self._last_mouse_idx = None
        self.rate_rotate = self.fovy / self.height  # 旋转速度
        self.rate_translate = 1.  # 平移速度
        self.need_update = True

    def resize(self, W: int = None, H: int = None, channels: int = None):
        if super().resize(W, H, channels):
            self.need_update = True

    def callback_mouse_down(self, sender, app_data):
        # if dpg.is_item_hovered(self._img_id):
        #     self._last_mouse_pos = self.get_mouse_pos()
        #     self._last_mouse_idx = app_data[0]
        #     print(sender, app_data, self._last_mouse_pos)
        # else:
        #     self._last_mouse_pos = None
        #     self._last_mouse_idx = None
        pass

    def callback_mouse_release(self, sender, app_data):
        self._last_mouse_pos = None
        self._last_mouse_idx = None

    def callback_mouse_wheel(self, sender, app_data):
        if not dpg.is_item_hovered(self._img_id):
            return
        self.scale(app_data)

    def callback_mouse_drag(self, sender, app_data):
        if not dpg.is_item_hovered(self._img_id):
            return
        if app_data[0] == dpg.mvMouseButton_Left:
            if self._last_mouse_pos is not None and self._last_mouse_idx == app_data[0]:
                now_pos = self.get_mouse_pos()
                self.rotate(now_pos[0] - self._last_mouse_pos[0], now_pos[1] - self._last_mouse_pos[1])
        elif app_data[0] == dpg.mvMouseButton_Right:
            if self._last_mouse_pos is not None and self._last_mouse_idx == app_data[0]:
                now_pos = self.get_mouse_pos()
                self.translate(now_pos[0] - self._last_mouse_pos[0], now_pos[1] - self._last_mouse_pos[1])
        self._last_mouse_pos = self.get_mouse_pos()
        self._last_mouse_idx = app_data[0]

    def rotate(self, dx: float, dy: float):
        if dx == 0 and dy == 0:
            return
        radiu = (self.eye - self.at).norm()
        dir_vec = ops_3d.normalize(self.eye - self.at)
        right_vec = ops_3d.normalize(torch.linalg.cross(self.up, dir_vec), dim=-1)
        theta = -dy * self.rate_rotate
        dir_vec = ops_3d.quaternion.xfm(dir_vec, ops_3d.quaternion.from_rotate(right_vec, right_vec.new_tensor(theta)))

        right_vec = ops_3d.normalize(torch.linalg.cross(self.up, dir_vec), dim=-1)
        up_vec = torch.linalg.cross(dir_vec, right_vec)
        theta = -dx * self.rate_rotate
        dir_vec = ops_3d.quaternion.xfm(dir_vec, ops_3d.quaternion.from_rotate(up_vec, up_vec.new_tensor(float(theta))))
        self.eye = self.at + ops_3d.normalize(dir_vec) * radiu
        self.up = up_vec
        self.need_update = True

    def translate(self, dx: float, dy: float):
        """在垂直于视线方向进行平移, 即在view space进行平移"""
        if dx == 0 and dy == 0:
            return
        Tw2v = ops_3d.look_at(self.eye, self.at, self.up)
        p1 = ops_3d.xfm(ops_3d.xfm(self.at, Tw2v), self.Tv2s)

        p2 = p1.clone()
        p2[0] += dx * p1[2]
        p2[1] += dy * p1[2]
        Tv2w = ops_3d.look_at(self.eye, self.at, self.up, inv=True)
        p1 = ops_3d.xfm(ops_3d.xfm(p1, self.Ts2v), Tv2w)
        p2 = ops_3d.xfm(ops_3d.xfm(p2, self.Ts2v), Tv2w)
        delta = (p1 - p2)[:3] * self.rate_translate
        self.at += delta
        self.eye += delta
        self.need_update = True

    def scale(self, delta=0.0):
        self.eye = self.at + (self.eye - self.at) * 1.1 ** (-delta)
        self.need_update = True

    def update(self, image: Union[np.ndarray, Tensor] = None, resize=False):
        if image is None and not self.need_update:
            return
        self.need_update = False
        if image is None:
            Tw2v = ops_3d.look_at(self.eye, self.at, self.up)
            image = self.renderer(Tw2v, self.fovy, self.size)
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        image = image.astype(np.float32)
        if image.ndim == 4:
            image = image[0]
        if image.shape[-1] not in [3, 4]:
            assert image.shape[0] in [3, 4]
            image = image.transpose(1, 2, 0)
        if resize:
            image = cv2.resize(image, self.size)
        self.resize(image.shape[1], image.shape[0], image.shape[2])
        self.data = image

    def set_fovy(self, fovy=60.):
        self.fovy = math.radians(fovy)
        fov = [ops_3d.fovx_to_fovy(self.fovy, self.size[1] / self.size[0]), self.fovy]
        self.Tv2s = ops_3d.camera_intrinsics(size=self.size, fov=fov)
        self.Ts2v = ops_3d.camera_intrinsics(size=self.size, fov=fov, inv=True)
        self.need_update = True

    def set_pose(self, eye=None, at=None, up=None, Tw2v=None, Tv2w=None):
        if Tv2w is None and Tw2v is not None:
            Tv2w = Tw2v.inverse()
        if Tv2w is not None:
            Tv2w = Tv2w.view(-1, 4, 4)[0].to(self.eye.device)
            eye, at, up = ops_3d.look_at_get(Tv2w)
        if eye is not None:
            self.eye = eye
        if at is not None:
            self.at = at
        if up is not None:
            self.up = up
        self.need_update = True

    def set_need_update(self, need_update=True):
        self.need_update = need_update

    def build_gui_camera(self):
        with dpg.group(horizontal=True):
            dpg.add_text('fovy')
            dpg.add_slider_float(
                min_value=15.,
                max_value=180.,
                default_value=math.degrees(self.fovy),
                callback=lambda *args: self.set_fovy(dpg.get_value('set_fovy')),
                tag='set_fovy'
            )
        with dpg.group():
            item_width = 50
            with dpg.group(horizontal=True):
                dpg.add_text('eye')
                dpg.add_input_float(tag='eye_x', step=0, width=item_width)
                dpg.add_input_float(tag='eye_y', step=0, width=item_width)
                dpg.add_input_float(tag='eye_z', step=0, width=item_width)
            with dpg.group(horizontal=True):
                dpg.add_text('at ')
                dpg.add_input_float(tag='at_x', step=0, width=item_width)
                dpg.add_input_float(tag='at_y', step=0, width=item_width)
                dpg.add_input_float(tag='at_z', step=0, width=item_width)

            def change_eye(*args):
                print('change camera position', args)
                self.eye = self.eye.new_tensor([dpg.get_value(item) for item in ['eye_x', 'eye_y', 'eye_z']])
                self.at = self.at.new_tensor([dpg.get_value(item) for item in ['at_x', 'at_y', 'at_z']])
                self.need_update = True

            def to_camera_pos(campos, up):
                def callback():
                    r = (self.eye - self.at).norm()
                    eye = self.eye.new_tensor(campos)
                    self.eye = eye / eye.norm(keepdim=True) * r + self.at
                    self.up = self.up.new_tensor(up)
                    self.set_need_update()

                return callback

            with dpg.group(horizontal=True):
                dpg.add_button(label='change', callback=change_eye)
                dpg.add_button(label='+X', callback=to_camera_pos((1, 0, 0), (0, 1, 0)))
                dpg.add_button(label='-X', callback=to_camera_pos((-1, 0, 0), (0, 1, 0)))
                dpg.add_button(label='+Y', callback=to_camera_pos((0, 1, 0), (0, 0, 1)))
                dpg.add_button(label='-Y', callback=to_camera_pos((0, -1, 0), (0, 0, 1)))
                dpg.add_button(label='+Z', callback=to_camera_pos((0, 0, 1), (0, 1, 0)))
                dpg.add_button(label='-Z', callback=to_camera_pos((0, 0, -1), (0, 1, 0)))

    def update_gui_camera(self):
        if self.need_update:
            dpg.set_value('eye_x', self.eye[0].item())
            dpg.set_value('eye_y', self.eye[1].item())
            dpg.set_value('eye_z', self.eye[2].item())
            dpg.set_value('at_x', self.at[0].item())
            dpg.set_value('at_y', self.at[1].item())
            dpg.set_value('at_z', self.at[2].item())


if __name__ == '__main__':
    Fast_SP_GS_GUI().run()
