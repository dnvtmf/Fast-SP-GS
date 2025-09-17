import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PIL_Image
import matplotlib.pyplot as plt

import utils
from datasets.base import NERF_DATASET_STYLE, NERF_DATASETS, NERF_Base_Dataset
from utils import ops_3d, Cameras

NERF_DATASETS['Nerfies'] = {
    'common': {
        'style': 'NerfiesDataset',
        'root': 'NeRF/Nerfies',
        'img_dir': 'rgb',
        'camera_dir': 'camera',
        'camera_suffix': '.json',
        'split_file': 'dataset.json',
        'meta_file': 'metadata.json',
        'scene_file': 'scene.json',
        'points_file': 'points.npy',
        'use_time': True,
        'downscale': 1,
        'background': 'white',
        'scene': 'vrig-chicken',
    },
    'train': {'split': 'train'},
    'eval': {'split': 'val'},
    'test': {'split': 'test'},
    'Nerfies': ['train', 'eval', 'test'],
}

NERF_DATASETS['HyperNeRF'] = {
    'common': {
        'style': 'NerfiesDataset',
        'root': 'NeRF/HyperNeRF',
        'img_dir': 'rgb',
        'img_s_dir': 'rgb/{scale}x',
        'camera_dir': 'camera',
        'camera_suffix': '.json',
        'split_file': 'dataset.json',
        'meta_file': 'metadata.json',
        'scene_file': 'scene.json',
        'points_file': 'points.npy',
        'use_time': True,
        'downscale': 1,
        'background': 'white',
        'scene': 'vrig-chicken',
    },
    'train': {'split': 'train'},
    'eval': {'split': 'val'},
    'test': {'split': 'test'},
    'HyperNeRF': ['train', 'eval', 'test'],
}

NERF_DATASETS['NeRF_DS'] = {
    'common': {
        'style': 'NerfiesDataset',
        'root': 'NeRF/NeRF_DS',
        'img_dir': 'rgb',
        'img_s_dir': 'rgb/{scale}x',
        'camera_dir': 'camera',
        'camera_suffix': '.json',
        'split_file': 'dataset.json',
        'meta_file': 'metadata.json',
        'scene_file': 'scene.json',
        'points_file': 'points.npy',
        'use_time': True,
        'downscale': 1,
        'background': 'white',
        'scene': 'as',
    },
    'train': {'split': 'train'},
    'eval': {'split': 'val'},
    'test': {'split': 'test'},
    'NeRF_DS': ['train', 'eval', 'test'],
}

NERF_DATASETS['DyCheck'] = {
    'common': {
        'style': 'NerfiesDataset',
        'root': 'NeRF/DyCheck/iphone',
        'img_dir': 'rgb',
        'img_s_dir': 'rgb/{scale}x',
        'camera_dir': 'camera',
        'camera_suffix': '.json',
        'split_file': 'dataset.json',
        'meta_file': 'metadata.json',
        'scene_file': 'scene.json',
        'points_file': 'points.npy',
        'time_id_key': 'warp_id',
        'use_time': True,
        'downscale': 1,
        'background': 'white',
        'scene': 'apple',
    },
    'train': {'split': 'train'},
    'eval': {'split': 'val'},
    'test': {'split': 'test'},
    'DyCheck': ['train', 'eval', 'test'],
}


@NERF_DATASET_STYLE.register()
class NerfiesDataset(NERF_Base_Dataset):

    def __init__(
            self,
            root: Path,
            scene='',
            camera_dir='camera',
            camera_suffix='.json',
            split='train',
            img_dir='rgb',
            img_s_dir: str = None,
            img_suffix='.png',
            mask_dir='',
            mask_suffix='',
            background='white',
            meta_file='metadata.json',
            split_file='dataset.json',
            points_file='points.npy',
            scene_file='scene.json',
            time_id_key='time_id',
            image_size=None,
            downscale: int = None,
            sample_stride=1,
            is_srgb_image=False,
            camera_radiu_scale=1.0,  # make sure camera are inside the bounding box.
            camera_noise=0.,  # Add noise to camera pose
            near: float = None,
            far: float = None,
            coord_src='opengl',
            coord_dst='opengl',
            use_time=True,
            scale_by_scene_info=False,
            dust3r_file='',
            motion_masks_dir='',
            segment_file='',
            flow_cfg: dict = None,
            depth_dir='',
            perspective_z01=True,
            **kwargs
    ):
        root = root.joinpath(scene)
        self.root = root
        self.scene = scene
        assert img_suffix in utils.image_extensions
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.ndc = False  # normalized device coordinates
        self.downscale = downscale
        self.sample_stride = sample_stride
        self.split = split
        if img_s_dir and root.joinpath(img_s_dir.format(scale=self.downscale)).exists():
            self.img_dir = root.joinpath(img_s_dir.format(scale=self.downscale))
            downscale = 1
        else:
            self.img_dir = root.joinpath(img_dir)
        self.mask_dir = mask_dir
        self.coord_src = ops_3d.coordinate_system[coord_src.lower()]
        self.coord_dst = ops_3d.coordinate_system[coord_dst.lower()]
        ops_3d.set_coord_system(self.coord_dst)
        self.points_file = points_file

        self.split_file = split_file
        if split_file:
            with root.joinpath(split_file).open('r') as f:
                dataset = json.load(f)
                if len(dataset['val_ids']) > 0:
                    image_names = dataset['train_ids' if self.split == 'train' else 'val_ids']
                else:
                    if self.split == 'train':
                        image_names = [img_name for i, img_name in enumerate(dataset['ids']) if i % 4 != 3]
                    else:
                        image_names = [img_name for i, img_name in enumerate(dataset['ids']) if i % 4 == 3]
        else:
            image_names = [img_path.stem for img_path in root.joinpath(self.img_dir).glob('*' + self.img_suffix)]
            image_names = sorted(image_names)
        assert len(image_names) > 0, f"Can not found any images for split {split}"

        self.meta_file = meta_file
        camera_ids = []
        time_ids = []
        max_time_id = 0
        with root.joinpath(meta_file).open('r') as f:
            metadata = json.load(f)
            for k, v in metadata.items():
                # warp_id for DyCheck dataset
                max_time_id = max(max_time_id, v[time_id_key])
        self.camera_dir = camera_dir
        self.camera_suffix = camera_suffix
        assert camera_suffix == '.json'

        self.scene_file = scene_file
        with open(root.joinpath(scene_file), 'r') as f:
            scene_info = json.load(f)
        near = scene_info['near'] if near is None else near
        far = scene_info['far'] if far is None else far
        scene_scale = scene_info['scale']
        scene_center = torch.tensor(scene_info['center'])
        # scene_info['scene_to_metric']

        Tw2v = torch.zeros((len(image_names), 4, 4))
        focal = torch.zeros((len(image_names), 2))
        paths = []
        principal_points = []
        for i, img_name in enumerate(image_names):
            camera_ids.append(metadata[img_name]['camera_id'])
            time_ids.append(metadata[img_name][time_id_key])
            camera_file = root.joinpath(camera_dir, f"{img_name}{camera_suffix}")
            with open(camera_file, 'r') as f:
                camera_info = json.load(f)
            R = torch.tensor(camera_info['orientation'])
            t = torch.tensor(camera_info['position'])
            if scale_by_scene_info:
                t = (t - scene_center) * scene_scale
            t = -t @ R.T
            Tw2v[i, :3, :3] = R
            Tw2v[i, :3, 3] = t
            focal_i = camera_info['focal_length'] / self.downscale
            focal[i, 0] = focal_i
            focal[i, 1] = focal_i  # * camera_info.get('pixel_aspect_ratio', 1.)
            self.image_size = tuple(camera_info['image_size'])  # noqa
            paths.append(root.joinpath(self.img_dir, img_name + img_suffix))
            principal_points.append([v / self.downscale for v in camera_info['principal_point']])
        Tw2v[:, 3, 3] = 1
        # self.Tw2v = ops_3d.convert_coord_system_matrix(self.Tw2v, self.coord_src, self.coord_dst)
        Tw2v = ops_3d.convert_coord_system(Tw2v, self.coord_src, self.coord_dst)
        self.time_ids = torch.tensor(time_ids)
        self.camera_ids = torch.tensor(camera_ids)
        self.num_frames = len(self.time_ids)
        self.num_cameras = -1  # every image have its camera
        self.times = torch.tensor(time_ids) / (max_time_id - 1) if use_time else None  # [0., 1]
        principal_points = torch.tensor(principal_points)

        self.images = self.load_images(paths, image_size, downscale, srgb=is_srgb_image)[..., :3]  # shape: [N, H, W, 3]
        self.image_names = image_names
        if mask_suffix:
            masks = [utils.load_image(root.joinpath(mask_dir, f"{path.stem}{mask_suffix}")) for path in paths]
            masks = np.stack(masks).astype(np.bool_)
            if masks.ndim == 4:
                masks = masks[..., 0]
            self.masks = torch.from_numpy(masks)
            assert self.images.shape[-1] == 3
            self.images = torch.cat([self.images, self.masks[..., None].to(self.images)], dim=-1)
        self.image_size = (self.images.shape[2], self.images.shape[1])
        self.aspect = self.image_size[0] / self.image_size[1]
        fovx = ops_3d.focal_to_fov(focal[..., 0], self.image_size[0])
        fovy = ops_3d.focal_to_fov(focal[..., 1], self.image_size[1])
        FoV = torch.stack([fovx, fovy], dim=-1)
        Tv2s = ops_3d.camera_intrinsics(focal, principal_points, self.image_size)
        Tv2c = ops_3d.perspective2(FoV=FoV, near_far=(near, far), size=self.image_size, pp=principal_points,
                                   z01=perspective_z01)

        self.background_type = background
        self.init_background(self.images)
        if self.background_type != 'random' and self.background_type != 'none' and self.images.shape[-1] == 4:
            torch.lerp(self.background, self.images[..., :3], self.images[..., -1:], out=self.images[..., :3])
        cameras = Cameras(Tw2v=Tw2v, FoV=FoV, Tv2s=Tv2s, Tv2c=Tv2c, image_size=self.image_size, focal=focal, )
        # self.camera_radius_scale = camera_radius_scale
        # self.Tv2w[:, :3, 3] = (self.Tv2w[:, :3, 3] * camera_radius_scale + 0)

        # self.camera_noise = camera_noise
        # if camera_noise > 0:
        #     self.Tv2w = ops_3d.rigid.lie_to_Rt(torch.randn(len(self.Tv2w), 6) * self.camera_noise) @ self.Tv2w
        if points_file:
            ply_path = root.joinpath(points_file)
            assert ply_path.exists(), f"Point file {ply_path} is not extisted"
            if ply_path.suffix == '.npy':
                points = np.load(root.joinpath(points_file))
            elif ply_path.suffix == '.ply':
                points = utils.to_np(utils.load_ply(ply_path)['v_pos'])
            else:
                raise NotImplementedError(f"not support read {ply_path}")
            if scale_by_scene_info:
                points = (points - scene_center.numpy()) * scene_scale
            self.points = points.astype(np.float32)
        else:
            self.points = None

        self.scene_size = 2.6 if scale_by_scene_info else scene_scale
        self.scene_center = 0 if scale_by_scene_info else scene_center.numpy()

        if dust3r_file:
            image_names = [path.name for path in paths]
            self.dust3r = torch.load(root.joinpath(dust3r_file), map_location='cpu')
            self.dust3r['pairs'] = np.array(
                [[image_names.index(img_a), image_names.index(img_b)] for img_a, img_b in self.dust3r['pairs']]
            )
            self.dust3r['conf1'] = self.dust3r['conf1'].log()
            self.dust3r['conf2'] = self.dust3r['conf2'].log()
        else:
            self.dust3r = None
        self.motion_masks = None
        if motion_masks_dir:
            motion_masks_dir = root.joinpath(motion_masks_dir)
            motion_masks = []
            for img_path in paths:
                mask = PIL_Image.open(motion_masks_dir.joinpath(img_path.name).with_suffix('.png'))
                motion_masks.append(np.array(mask))
            motion_masks = np.stack(motion_masks, axis=0)
            motion_masks = torch.from_numpy(motion_masks / motion_masks.max())
            if motion_masks.shape[-2:] != self.images.shape[1:3]:
                motion_masks = motion_masks[:, None].float()
                motion_masks = F.interpolate(motion_masks, size=self.images.shape[1:3], mode='bilinear')[:, 0]
            self.motion_masks = motion_masks > 0.5
            logging.info(f"Load motion masks from {motion_masks_dir}/*.png, {self.motion_masks.shape}")
        self.load_flow(root, paths, **utils.merge_dict(flow_cfg))
        self.depths = None
        if depth_dir:
            depth_dir = root.joinpath(depth_dir)
            depths = []
            for img_path in paths:
                data = np.load(depth_dir.joinpath(f"{img_path.stem}.npz"))
                depths.append(data['depth'])
            self.depths = torch.from_numpy(np.stack(depths)).to(torch.float32).squeeze()
            logging.info(f"Load depth from {depth_dir}/*.npz, {self.depths.shape}")
        super().__init__(root, cameras, self.images, near=near, far=far, perspective_z01=perspective_z01, **kwargs)

    def load_flow(self, root, paths, flow_dir='', fwd_suffix='_fwd.npz', bwd_suffix='_bwd.npz', fwd_mask='',
                  bwd_mask=''):
        self.flow_fwd = None
        self.flow_f_m = None
        self.flow_bwd = None
        self.flow_b_m = None
        if not flow_dir:
            return
        flow_dir = root.joinpath(flow_dir)
        flow_fwd = []
        flow_f_m = []
        flow_bwd = []
        flow_b_m = []
        if fwd_suffix[-4:] == '.npz':
            for img_path in paths[:-1]:
                data = np.load(flow_dir.joinpath(f"{img_path.stem}{fwd_suffix}"))
                flow_fwd.append(data['flow'])
                flow_f_m.append(data['mask'])
        elif fwd_suffix[-4:] == '.flo':
            for img_path in paths[:-1]:
                flow_fwd.append(utils.load_flow(flow_dir.joinpath(f"{img_path.stem}{fwd_suffix}")))
                if fwd_mask:
                    flow_f_m.append(np.array(PIL_Image.open(flow_dir.joinpath(f"{img_path.stem}{fwd_mask}"))))

        if fwd_suffix[-4:] == '.npz':
            for img_path in paths[1:]:
                data = np.load(flow_dir.joinpath(f"{img_path.stem}{bwd_suffix}"))
                flow_bwd.append(data['flow'])
                flow_b_m.append(data['mask'])
        if len(flow_fwd) > 0:
            self.flow_fwd = torch.from_numpy(np.stack(flow_fwd, axis=0)).to(torch.float32)
            self.flow_fwd = torch.cat([self.flow_fwd, torch.zeros_like(self.flow_fwd[:1])])
        if len(flow_bwd) > 0:
            self.flow_bwd = torch.from_numpy(np.stack(flow_bwd, axis=0)).to(torch.float32)
            self.flow_bwd = torch.cat([torch.zeros_like(self.flow_bwd[:1]), self.flow_bwd])
        if len(flow_f_m) > 0:
            self.flow_f_m = torch.from_numpy(np.stack(flow_f_m, axis=0))
            self.flow_f_m = torch.cat([self.flow_f_m, torch.zeros_like(self.flow_f_m[:1])])
        if len(flow_b_m) > 0:
            self.flow_b_m = torch.from_numpy(np.stack(flow_b_m, axis=0))
            self.flow_b_m = torch.cat([torch.zeros_like(self.flow_b_m[:1]), self.flow_b_m])
        logging.info(
            f"Load flow from {flow_dir} {utils.show_shape(self.flow_fwd, self.flow_f_m, self.flow_bwd, self.flow_b_m)}")

    def camera_ray(self, index, batch_size=None):
        inputs, targets, infos = super().camera_ray(index, batch_size)
        index = infos['index']
        if self.motion_masks is not None:
            targets['motion'] = self.motion_masks[index]
        if self.flow_fwd is not None:
            targets['flow_fwd'] = self.flow_fwd[index]
        if self.flow_f_m is not None:
            targets['flow_f_m'] = self.flow_f_m[index]
        if self.flow_bwd is not None:
            targets['flow_bwd'] = self.flow_bwd[index]
        if self.flow_b_m is not None:
            targets['flow_b_m'] = self.flow_b_m[index]
        if self.depths is not None:
            targets['depth'] = self.depths[index]
        return inputs, targets, infos

    def extra_repr(self):
        s = [
            f"img_dir: {self.img_dir}, img_suffix: {self.img_suffix}" if self.img_dir else None,
            f"meta_file: {self.meta_file}, split: {self.split_file}, scene_file: {self.scene_file}",
            f"mask_dir: {self.mask_dir}, mask_suffix: {self.mask_suffix}" if self.mask_suffix else None,
            f"points file: {self.points_file}",
            f"camera: {self.camera_dir}/*{self.camera_suffix}, coord system: {self.coord_src}→{self.coord_dst}",
            f"image size{'' if self.downscale is None else f'↓{self.downscale}'}="
            f"{self.image_size[0]} x {self.image_size[1]}",
            f"background={self.background_type}",
            f"num_frames: {self.num_frames}, num_cameras: {self.num_cameras}",
            # f"camera_radiu_scale={self.camera_radiu_scale}" if self.camera_radiu_scale != 1.0 else None,
            # f"camera noise: {self.camera_noise}" if self.camera_noise > 0 else None,
        ]
        return super().extra_repr() + s


def test():
    import matplotlib.pyplot as plt
    utils.set_printoptions()
    # cfg = {**NERF_DATASETS['HyperNeRF']['common'], **NERF_DATASETS['HyperNeRF']['train']}
    # cfg['scene'] = 'vrig-chicken'
    # cfg['scene'] = 'vrig-peel-banana'
    # cfg['scene'] = 'vrig-broom'
    cfg = {**NERF_DATASETS['DyCheck']['common'], **NERF_DATASETS['DyCheck']['train']}
    cfg['root'] = Path('~/data', cfg['root']).expanduser()
    cfg['with_rays'] = True
    cfg['downscale'] = 2
    cfg['coord_src'] = 'colmap'
    cfg['coord_dst'] = 'opengl'
    db = NerfiesDataset(**cfg)
    print(db)
    print(utils.show_shape(db.camera_ray(0)))

    inputs, targets, infos = db.camera_ray(0)
    rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
    print(inputs['time_id'])
    print(infos['Tw2v'])
    print(infos['Tv2c'])
    print(infos['campos'])
    print(infos['cam_id'])
    # exit()

    plt.subplot(131)
    plt.imshow(targets['images'][..., :3])
    plt.subplot(132)
    plt.imshow(db.images[1, ..., :3])
    plt.subplot(133)
    plt.imshow(inputs['background'].expand_as(targets['images'][..., :3]))
    # plt.subplot(133)
    # plt.imshow(torch.lerp(inputs['background'], targets['images'][..., :3], targets['images'][..., 3:]))
    plt.show()

    with utils.vis3d:
        utils.vis3d.add_camera_poses(db.cameras.Tv2w, None, np.rad2deg(db.cameras.FoV[0, 1].item()), db.aspect, 0.1,
                                     (1, 0, 0))
        utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2)[40::80, 40::80])
        inputs = db.random_ray(None, 512)[0]
        rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
        utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2), color=(0.1, 0.1, 0.1))
        inputs = db.random_ray(1, 10)[0]
        rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
        utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2), color=(0.3, 0.3, 0.3))
        utils.vis3d.add_lines(
            points=[
                [-1., -1, -1],  # 0
                [-1., -1., 1],  # 1
                [-1., 1., -1.],  # 2
                [-1., 1., 1.],  # 3
                [1., -1., -1],  # 4
                [1, -1, 1],  # 5
                [1, 1, -1],  # 6
                [1, 1, 1],  # 7
            ],
            line_index=[[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
        )
    db.batch_mode = False
    print('batch_mode=False', utils.show_shape(db[0, 5]))
    db.batch_mode = True
    print('batch_mode=True', utils.show_shape(db[0, 5]))

    points = np.load(db.root.joinpath(db.points_file))
    print(utils.show_shape(points))


if __name__ == '__main__':
    test()
