import json
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from datasets.base import NERF_DATASET_STYLE, NERF_DATASETS, NERF_Base_Dataset
import utils
from utils import ops_3d, Cameras

NERF_DATASETS['DNeRF'] = {
    'common': {
        'style': 'DNeRFDataset',
        'root': 'NeRF/D_NeRF',
        'use_time': True,
        'downscale': 1,
        'near': 2.,
        'far': 6.0,
        'background': 'white',
    },
    'train': {'split': 'train'},
    'eval': {'split': 'val'},
    'test': {'split': 'test'},
    'DNeRF': ['train', 'eval', 'test'],
}


@NERF_DATASET_STYLE.register()
class DNeRFDataset(NERF_Base_Dataset):

    def __init__(
            self,
            root: Path,
            scene='',
            camera_file='transforms_{}.json',
            split='train',
            img_dir='',
            img_suffix='.png',
            mask_dir='',
            mask_suffix='',
            background='white',
            near=2.,
            far=6.0,
            image_size=None,
            downscale: int = None,
            is_srgb_image=False,
            camera_radiu_scale=1.0,  # make sure camera are inside the bounding box.
            camera_noise_R=0.,  # Add noise to camera pose
            camera_noise_t=0.,  # Add noise to camera pose
            coord_src='opengl',
            coord_dst='opengl',
            sample_stride=1,
            use_time=False,
            weighted_sample=False,
            num_frames_max=-1,
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
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.coord_src = ops_3d.coordinate_system[coord_src.lower()]
        self.coord_dst = ops_3d.coordinate_system[coord_dst.lower()]
        ops_3d.set_coord_system(self.coord_dst)
        self.weighted_sample = weighted_sample

        cameras = Cameras()
        fovx = None
        if camera_file.endswith('.json'):
            self.camera_file = camera_file.format(split)
            fovx, cameras.Tv2w, paths, times = self.load_camera_from_json(root.joinpath(self.camera_file))
        elif camera_file.endswith('.npz'):
            self.camera_file = camera_file
            paths = sorted(list(root.joinpath(img_dir).glob('*' + img_suffix)))
            cameras.Tv2w, cameras.Tv2s, Ts = self.load_camera_from_npz(
                root.joinpath(camera_file), [int(p.stem) for p in paths])
            cameras.focal = cameras.Tv2s[:, 0, 0:1]
            # self.Ts2v = torch.inverse(self.Tv2s)
            times = None
        else:
            raise NotImplementedError
        if len(paths) > num_frames_max > 0:
            cameras.Tv2w = cameras.Tv2w[:num_frames_max]
            paths = paths[:num_frames_max]
            if times is not None:
                times = times[:num_frames_max]
            if cameras.Tv2s is not None and cameras.Tv2s.ndim == 3:
                cameras.Tv2s = cameras.Tv2s[:num_frames_max]
            # if cameras.Ts is not None and cameras.Ts.ndim == 3:
            #     cameras.Ts = cameras.Ts[:num_frames_max]
            logging.info(f'[red] only use first {len(paths)} images')

        cameras.Tv2w = ops_3d.convert_coord_system(cameras.Tv2w, coord_src, coord_dst, inverse=True)

        self.images = self.load_images(paths, image_size, downscale, srgb=is_srgb_image)  # shape: [N, H, W, 3]
        if mask_suffix:
            masks = [utils.load_image(root.joinpath(mask_dir, f"{path.stem}{mask_suffix}")) for path in paths]
            masks = np.stack(masks).astype(np.bool_)
            if masks.ndim == 4:
                masks = masks[..., 0]
            self.masks = torch.from_numpy(masks)
            assert self.images.shape[-1] == 3
            self.images = torch.cat([self.images, self.masks[..., None].to(self.images)], dim=-1)
        self.image_size = cameras.image_size = (self.images.shape[2], self.images.shape[1])
        self.aspect = self.image_size[0] / self.image_size[1]
        self.times = times if use_time else None  # * 2 - 1.
        self.num_frames = len(times)  # single camera
        self.num_cameras = -1
        self.time_ids = torch.arange(self.num_frames)
        self.camera_ids = torch.zeros_like(self.time_ids)

        if fovx is None:
            fovx = ops_3d.focal_to_fov(cameras.focal, self.image_size[0])
        cameras.FoV = torch.tensor([fovx, ops_3d.fovx_to_fovy(fovx, self.aspect)], dtype=torch.float)
        self.background_type = background
        self.init_background(self.images)
        if self.background_type not in ['random', 'random2', 'none']:
            torch.lerp(self.background, self.images[..., :3], self.images[..., -1:], out=self.images[..., :3])

        self.camera_radio_scale = camera_radiu_scale
        cameras.Tv2w[:, :3, 3] = (cameras.Tv2w[:, :3, 3] * camera_radiu_scale + 0)

        # self.camera_noise = (float(camera_noise_R), float(camera_noise_t))
        # if self.camera_noise != (0., 0.):
        #     self.Tv2w_origin = self.Tv2w.clone()
        #     noise_R = torch.randn(len(self.Tv2w), 3) * self.camera_noise[0]
        #     noise_t = torch.randn(len(self.Tv2w), 3) * self.camera_noise[1]
        #     self.Tv2w = ops_3d.rigid.lie_to_Rt(noise_R, noise_t) @ self.Tv2w

        self.scene_size = 2.6
        self.scene_center = 0  # [-1.3, 1.3]
        super().__init__(root, cameras, self.images, near=near, far=far, **kwargs)

    def load_camera_from_json(self, camera_file: Path):
        with camera_file.open('r') as f:
            meta = json.load(f)
        cams = []
        paths = []
        times = []
        for i in range(len(meta['frames'])):
            frame = meta['frames'][i]
            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
            paths.append(self.root.joinpath(self.img_dir, frame['file_path'] + self.img_suffix))
            times.append(frame['time'] if 'time' in frame else float(i) / (len(meta['frames']) - 1))
        fovx = float(meta["camera_angle_x"])
        Tv2w = torch.from_numpy(np.stack(cams, axis=0))
        times = torch.tensor(times, dtype=torch.float)
        return fovx, Tv2w, paths, times

    def load_camera_from_npz(self, camera_file: Path, indices: List[int]):
        meta = np.load(camera_file)
        Tw2s = np.stack([meta[f"world_mat_{i}"].astype(np.float64) for i in indices])
        scales = np.stack([meta[f"scale_mat_{i}"].astype(np.float64) for i in indices])
        Tw2s = Tw2s @ scales
        if False and f"camera_mat_{indices[0]}" in meta:
            Tv2s = np.stack([meta[f"camera_mat_{i}"].astype(np.float64) for i in indices])
            Tv2w = np.linalg.inv(Tw2s) @ Tv2s
        else:
            Tv2w = []
            Tv2s = []
            for i in range(len(Tw2s)):
                out = cv2.decomposeProjectionMatrix(Tw2s[i, :3, :4])
                K = out[0]
                R = out[1]
                t = out[2]

                K = K / K[2, 2]
                intrinsics = np.eye(4)
                intrinsics[:3, :3] = K

                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R.transpose()
                pose[:3, 3] = (t[:3] / t[3])[:, 0]
                Tv2s.append(intrinsics)
                Tv2w.append(pose)
            Tv2w = np.stack(Tv2w)
            Tv2s = np.stack(Tv2s)
        return torch.from_numpy(Tv2w), torch.from_numpy(Tv2s)[..., :3, :3], torch.from_numpy(scales)

    def extra_repr(self):
        s = [
            f"img_dir: {self.img_dir}, img_suffix: {self.img_suffix}" if self.img_dir else None,
            f"mask_dir: {self.mask_dir}, mask_suffix: {self.mask_suffix}" if self.mask_suffix else None,
            f"camera_file: {self.camera_file}, coord system: {self.coord_src}→{self.coord_dst}",
            f"image size{'' if self.downscale is None else f'↓{self.downscale}'}="
            f"{self.image_size[0]} x {self.image_size[1]}, split: {self.split}",
            f"background={self.background_type}",
            f'focal={utils.float2str(self.cameras.focal.mean().item())}',
            f"camera_radiu_scale={self.camera_radio_scale}" if self.camera_radio_scale != 1.0 else None,
            # f"camera noise: {self.camera_noise}" if self.camera_noise != (0., 0.) else None,
            f"num_frames: {self.num_frames}, num_cameras: {self.num_cameras}",
        ]
        return super().extra_repr() + s


def test():
    import matplotlib.pyplot as plt
    utils.set_printoptions()
    cfg = {**NERF_DATASETS['DNeRF']['common'], **NERF_DATASETS['DNeRF']['train']}
    scene = 'lego'
    cfg['root'] = Path('~/data', cfg['root']).expanduser()
    # cfg = {}
    # cfg['root'] = Path('~/data/NeRF/wan/static_lego').expanduser()
    # cfg['camera_file'] = 'camera_transforms.json'
    cfg['background'] = 'black'
    cfg['scene'] = scene
    # cfg['camera_noise_R'] = 1.0e-3
    # cfg['camera_noise_t'] = 0
    # cfg['near'] = 0.1
    # cfg['far'] = 1000.
    cfg['coord_src'] = 'blender'  # 'opengl'  #
    cfg['coord_dst'] = 'opengl'
    # cfg['coord_dst'] = 'opencv'
    db = DNeRFDataset(**cfg)
    # inputs, targets, infos = db.camera_ray(0)
    # print(inputs['rays_o'])
    # print(inputs['rays_d'])
    # print(infos['Tw2v'])
    # print(infos['Tw2v'].inverse())
    # print(targets['images'][..., 400:410, 400:410, :])
    # return
    print(db)
    print(utils.show_shape(db.camera_ray(0)))

    inputs, targets, infos = db.camera_ray(0)

    plt.subplot(131)
    plt.imshow(targets['images'][..., :3])
    plt.subplot(132)
    plt.imshow(inputs['background'].expand_as(targets['images'][..., :3]))
    plt.subplot(133)
    # plt.imshow(torch.lerp(inputs['background'], targets['images'][..., :3], targets['images'][..., 3:]))
    plt.imshow(db.images[2])
    plt.show()

    db.batch_mode = False
    print('batch_mode=False', utils.show_shape(db[0, 5]))
    db.batch_mode = True
    print('batch_mode=True', utils.show_shape(db[0, 5]))


def test2():
    import matplotlib.pyplot as plt
    utils.set_printoptions(4)
    print(Path(__file__).resolve())
    root = Path('/mnt/a/Projects/MeshNeRF/data/dtu')
    if not root.exists():
        root = Path('~/data/ReconComp2023/preprocessed_dtu').expanduser()
    assert root.is_dir()
    db = DNeRFDataset(root=root, camera_file='cameras_sphere.npz', img_dir='image', mask_dir='mask', mask_suffix='.png',
                      for_ingp=False, random_camera=False)
    # return
    print(db)
    print(utils.show_shape(db.camera_ray(0)))
    print(utils.show_shape(db.random_ray(0, 1024)))
    print(utils.show_shape(db.random_ray(None, 1024)))

    inputs, targets, infos = db.camera_ray(0)
    rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
    plt.subplot(131)
    plt.imshow(targets['images'][..., :3])
    plt.subplot(132)
    plt.imshow(inputs['background'])
    plt.subplot(133)
    plt.imshow(torch.lerp(inputs['background'], targets['images'][..., :3], targets['images'][..., 3:]))
    plt.show()
    print('camera position', infos['campos'])
    poses = np.array([[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1.]],
                      [[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1.]]],
                     dtype=np.float32)


if __name__ == '__main__':
    test()
