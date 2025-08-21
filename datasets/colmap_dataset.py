import logging
import os
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PIL_Image
from plyfile import PlyData, PlyElement

import utils
from utils import ops_3d
from utils.colmap import (
    read_points3D_text, read_points3D_binary, read_intrinsics_text,
    read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, readColmapCameras,
)
from datasets.base import NERF_DATASET_STYLE, NERF_Base_Dataset, NERF_DATASETS

NERF_DATASETS['Mip360'] = {
    'common': {
        'style': 'ColmapDataset',
        'colmap_dir': "sparse/0",
        'img_dir': "images",
        'img_s_dir': "images_{scale}",
        'root': 'NeRF/Mip360',
        'coord_src': 'colmap',
        'coord_dst': 'colmap',
        'background': 'black',
        'scene': 'bicycle'
    },
    'all': {'split': 'train', 'split_parts': -1},
    'train': {'split': 'train', 'split_parts': 8},
    'test': {'split': 'test', 'split_parts': 8},
    'Mip360': ['train', 'test', 'test']
}


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    if 'nx' in vertices:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=np.ascontiguousarray(positions), colors=np.ascontiguousarray(colors),
                           normals=np.ascontiguousarray(normals))


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


@NERF_DATASET_STYLE.register()
class ColmapDataset(NERF_Base_Dataset):
    def __init__(
            self,
            root: Path,
            scene='',
            img_dir='images',
            img_s_dir: str = None,
            mask_dir: str = None,
            colmap_dir='sparse/0',
            image_size=None,
            downscale=1,
            split='train',
            coord_src='colmap',
            coord_dst='opengl',
            background='black',
            split_file='',
            split_parts=8,
            near=0.01,
            far=100.,
            is_srgb_image=False,
            image_stride: int = 1,  # make W % image_stride == 0
            normalize_scene=False,
            **kwargs
    ):
        self.scene = scene
        root = root.joinpath(scene)
        self.coord_src = coord_src
        self.coord_dst = coord_dst
        ops_3d.set_coord_system(self.coord_dst)

        self.split_file = split_file
        self.split = split
        self.downscale = downscale

        image_names, cameras, self.point_cloud, self.cameras_extent = \
            self.read_colmap_camears(root.joinpath(colmap_dir), normalize_scene)
        max_time_id = len(image_names)
        if split_parts > 0:
            indices = [i for i in range(max_time_id) if (i % split_parts == 0) ^ (split == 'train')]
            image_names = [image_names[i] for i in indices]
            cameras = cameras[torch.tensor(indices)]
        else:
            indices = list(range(max_time_id))
        self.time_ids = torch.tensor(indices)
        self.camera_ids = torch.tensor(indices)
        self.num_frames = len(self.time_ids)  # max_time_id + 1
        self.num_cameras = -(self.camera_ids.max() + 1)  # <0
        self.times = self.time_ids / max_time_id  # * 2 - 1.
        cameras.Tw2v = ops_3d.convert_coord_system(cameras.Tw2v, self.coord_src, self.coord_dst)

        if img_s_dir and root.joinpath(img_s_dir.format(scale=self.downscale)).exists():
            img_dir = root.joinpath(img_s_dir.format(scale=self.downscale))
            downscale = 1
        else:
            img_dir = root.joinpath(img_dir)
        image_paths = [img_dir.joinpath(img_name) for img_name in image_names]
        self.images = self.load_images(image_paths, image_size, downscale, srgb=is_srgb_image)
        self.image_size = (self.images.shape[2], self.images.shape[1])
        if mask_dir and root.joinpath(mask_dir).exists():
            mask_dir = root.joinpath(mask_dir)
            masks = []
            for img_name in image_names:
                mask = PIL_Image.open(mask_dir.joinpath(img_name).with_suffix('.png'))
                mask = np.asarray(mask).squeeze()
                if mask.shape[-1] != self.image_size[0] and mask.shape[-2] != self.image_size[1]:
                    mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
                masks.append(mask)
            masks = torch.from_numpy(np.array(masks))[..., None]
            logging.info(f"Load masks from {mask_dir}, get {masks.shape} {masks.unique()}")
            if self.images.dtype == torch.uint8:
                masks = masks.to(torch.uint8) * 255
            self.images = torch.cat([self.images, masks], dim=-1)

        self.aspect = self.image_size[0] / self.image_size[1]
        self.image_names = image_names

        self.background_type = background
        self.init_background(self.images)

        if kwargs.get('dust3r_file', ''):
            image_names = [path.name for path in image_paths]
            self.dust3r = torch.load(root.joinpath(kwargs.pop('dust3r_file')), map_location='cpu')
            self.dust3r['pairs'] = np.array(
                [[image_names.index(img_a), image_names.index(img_b)] for img_a, img_b in self.dust3r['pairs']]
            )
        else:
            self.dust3r = None
        kwargs.pop('dust3r_file', '')
        self.motion_masks = None
        if kwargs.get('motion_masks_dir', ''):
            motion_masks_dir = root.joinpath(kwargs.pop('motion_masks_dir'))
            motion_masks = []
            for img_path in image_paths:
                mask = PIL_Image.open(motion_masks_dir.joinpath(img_path.name).with_suffix('.png'))
                motion_masks.append(np.array(mask))
            motion_masks = np.stack(motion_masks, axis=0)
            motion_masks = torch.from_numpy(motion_masks / motion_masks.max())
            if motion_masks.shape[-2:] != self.images.shape[1:3]:
                motion_masks = motion_masks[:, None].float()
                motion_masks = F.interpolate(motion_masks, size=self.images.shape[1:3], mode='bilinear')[:, 0]
            self.motion_masks = motion_masks > 0.5
            logging.info(f"Load motion masks from {motion_masks_dir}/*.png, {self.motion_masks.shape}")
        kwargs.pop('motion_masks_dir', '')
        if kwargs.get('flow_dir', ''):
            flow_dir = root.joinpath(kwargs.pop('flow_dir'))
            flow_fwb = []
            flow_mask = []
            for img_path in image_paths[:-1]:
                data = np.load(flow_dir.joinpath(f"{img_path.stem}_fwd.npz"))
                flow_fwb.append(data['flow'])
                flow_mask.append(data['mask'])
            self.flow_fwb = torch.from_numpy(np.stack(flow_fwb, axis=0)).to(torch.float32)
            self.flow_mask = torch.from_numpy(np.stack(flow_mask, axis=0))
            logging.info(f"Load flow from {flow_dir}/*.png, {self.flow_fwb.shape}")
        else:
            self.flow_fwb = None
            self.flow_mask = None
        kwargs.pop('flow_dir', '')
        self.depths = None
        if kwargs.get('depth_dir', ''):
            depth_dir = root.joinpath(kwargs.pop('depth_dir'))
            depth_fmt = kwargs.pop('depth_fmt', "{}.npz")
            depths = []
            for img_path in image_paths:
                depths.append(np.load(depth_dir.joinpath(depth_fmt.format(img_path.stem)))['pred'])
            self.depths = torch.from_numpy(np.stack(depths)).to(torch.float32)
            if self.depths.ndim == 3:
                self.depths = self.depths.unsqueeze(1)
            assert self.depths.ndim == 4
            if self.depths.shape[-2:] != self.images.shape[1:3]:
                self.depths = F.interpolate(self.depths, size=self.images.shape[1:3], mode='bilinear')[:, 0]
        kwargs.pop('depth_dir', '')
        super().__init__(root, cameras, self.images, near=near, far=far, **kwargs)
        self.cameras.resize_(self.image_size)
        if self.image_size[0] % image_stride != 0 or self.image_size[1] % image_stride != 0:
            W = (self.image_size[0] // image_stride + 1) * image_stride
            H = (self.image_size[1] // image_stride + 1) * image_stride
            new_images = self.images.new_zeros((self.images.shape[0], H, W, self.images.shape[-1]))
            new_images[:, :self.image_size[1], :self.image_size[0]] = self.images
            self.images = new_images
            self.cameras.pad_([0, 0, W - self.image_size[0], H - self.image_size[1]])
            self.image_size = (W, H)
            self.aspect = W / H
            logging.info(f"padding images from {self.image_size} to {(W, H)}")
            # index = np.random.randint(len(self.images))
            # plt.imshow(self.images[index])
            # plt.title(f"{index=}")
            # plt.show()
        self.cameras_extent *= 1.1
        self.scene_center = 0
        self.scene_size = self.cameras_extent * 3. * 2.

    @staticmethod
    def read_colmap_camears(colmap_dir: Path, normalize_scene=False):
        try:
            cameras_extrinsic_file = colmap_dir.joinpath("images.bin")
            cameras_intrinsic_file = colmap_dir.joinpath("cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except FileNotFoundError:
            cameras_extrinsic_file = colmap_dir.joinpath("images.txt")
            cameras_intrinsic_file = colmap_dir.joinpath("cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        names, cameras, _ = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics)
        Tv2w = cameras.Tw2v.inverse()
        camera_center, camera_scale = ops_3d.get_center_and_diag(Tv2w[:, :3, 3])
        indices = list(range(len(names)))
        indices = sorted(indices, key=lambda k: names[k])
        names = [names[k] for k in indices]
        cameras = cameras[torch.tensor(indices)]

        ply_path = colmap_dir.joinpath("points3D.ply")
        bin_path = colmap_dir.joinpath("points3D.bin")
        txt_path = colmap_dir.joinpath("points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except FileNotFoundError:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)

        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

        camera_scale = camera_scale.item()
        if normalize_scene:
            print(ops_3d.get_center_and_diag(Tv2w[:, :3, 3]))
            Tv2w = ops_3d.camera_translate_scale(Tv2w, translate=-camera_center, scale=1. / camera_scale)
            logging.info(f'normalized camera center {camera_center} and scale {camera_scale}')
            print(ops_3d.get_center_and_diag(Tv2w[:, :3, 3]))
            cameras.Tw2v = Tv2w.inverse()
            if pcd is not None:
                pcd.points.data = (pcd.points - camera_center.numpy()) / camera_scale
            camera_scale = 1.0
        return names, cameras, pcd, camera_scale

    def camera_ray(self, index, batch_size=None):
        inputs, targets, infos = super().camera_ray(index, batch_size)
        index = infos['index']
        if self.times is not None:
            inputs['t'] = self.times[index]
            if self.split == 'train':
                inputs['time_id'] = self.time_ids[index]
            infos['cam_id'] = self.camera_ids[index]
        if self.depths is not None:
            targets['depths'] = self.depths[index]
        if self.motion_masks is not None:
            targets['motion'] = self.motion_masks[index]
        # if self.seg_masks is not None:
        #     targets['segment'] = self.seg_masks[index]
        if self.flow_fwb is not None:
            if index < self.flow_fwb.shape[0]:
                targets['flow'] = self.flow_fwb[index]
                targets['flow_mask'] = self.flow_mask[index]
            else:
                targets['flow'] = torch.zeros_like(self.flow_fwb[0])
                targets['flow_mask'] = torch.zeros_like(self.flow_mask[0])
        return inputs, targets, infos

    def extra_repr(self):
        focal = self.cameras.focal.mean().item()
        s = [
            # f"img_dir: {self.img_dir}, img_suffix: {self.img_suffix}" if self.img_dir else None,
            # f"meta_file: {self.meta_file}, split: {self.split_file}, scene_file: {self.scene_file}",
            # f"mask_dir: {self.mask_dir}, mask_suffix: {self.mask_suffix}" if self.mask_suffix else None,
            # f"points file: {self.points_file}",
            f"coord system: {self.coord_src}→{self.coord_dst}",
            f"image size{'' if self.downscale is None else f'↓{self.downscale}'}="
            f"{self.image_size[0]} x {self.image_size[1]}, focal={utils.float2str(focal)}",
            f"background={self.background_type}",
            # f"num_frames: {self.num_frames}, num_cameras: {self.num_cameras}",
            # f"camera_radiu_scale={self.camera_radiu_scale}" if self.camera_radiu_scale != 1.0 else None,
            # f"camera noise: {self.camera_noise}" if self.camera_noise > 0 else None,
            f"scene center: {self.scene_center}, size: {self.cameras_extent}",
        ]
        return super().extra_repr() + s


def test_mip360():
    import matplotlib.pyplot as plt
    utils.set_printoptions()
    cfg = {**NERF_DATASETS['Mip360']['common'], **NERF_DATASETS['Mip360']['test']}
    scene = 'bicycle'
    cfg['root'] = Path('~/data', cfg['root']).expanduser()
    cfg['background'] = 'black'
    cfg['scene'] = scene
    cfg['near'] = 0.1
    cfg['far'] = 100.
    cfg['downscale'] = 4
    cfg['coord_dst'] = 'opengl'
    db = ColmapDataset(**cfg)
    print(db)
    print(utils.show_shape(db.images, db.cameras.Tw2v, db.cameras.Tv2c, db.cameras.FoV))
    print(utils.show_shape(db[0]))
    # print(utils.show_shape(db.random_ray(0, 1024)))
    # print(utils.show_shape(db.random_ray(None, 1024)))

    # inputs, targets, infos = db.random_ray(0, 1024)
    # aabb = torch.tensor([-1, -1., -1., 1., 1., 1.]).cuda()
    # rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
    # from networks.ray_sampler import near_far_from_aabb
    # near, far = near_far_from_aabb(rays_o.cuda(), rays_d.cuda(), aabb)
    # print(*near.aminmax(), *far.aminmax())
    # print()

    inputs, targets, infos = db[0]
    # rays_o, rays_d = inputs['rays_o'], inputs['rays_d']

    # plt.subplot(131)
    plt.imshow(targets['images'][..., :3])
    # plt.subplot(132)
    # plt.imshow(inputs['background'].expand_as(targets['images'][..., :3]))
    # plt.subplot(133)
    # plt.imshow(torch.lerp(inputs['background'], targets['images'][..., :3], targets['images'][..., 3:]))
    plt.show()
    fovy = db.cameras.FoV[0, 1]
    with utils.vis3d:
        utils.vis3d.add_camera_poses(
            db.cameras.Tv2w, None, np.rad2deg(fovy.item()), db.aspect, color=(1, 0, 0), size=0.5, is_opengl=False)
        # utils.vis3d.add_camera_poses(db.Tv2w_origin, None, np.rad2deg(db.FoV[1].item()), db.aspect, 0.5, (0, 1, 0))
        # utils.vis3d.add_lines(torch.stack([db.Tv2w[:, :3, 3], db.Tv2w_origin[:, :3, 3]], dim=1), color=(0.1, 0.1, 0.1))
        # utils.vis3d.add_lines(points=db.Tv2w[:, :3, 3], color=(0.1, 0.1, 0.1))
        # utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2)[40::80, 40::80])
        # inputs = db.random_ray(None, 512)[0]
        # rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
        # utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2), color=(0.1, 0.1, 0.1))
        # inputs = db.random_ray(1, 10)[0]
        # rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
        # utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2), color=(0.3, 0.3, 0.3))
        utils.vis3d.add_lines(points=[
            [-1., -1, -1],  # 0
            [-1., -1., 1],  # 1
            [-1., 1., -1.],  # 2
            [-1., 1., 1.],  # 3
            [1., -1., -1],  # 4
            [1, -1, 1],  # 5
            [1, 1, -1],  # 6
            [1, 1, 1],  # 7
        ],
            line_index=[[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])
    # db.batch_mode = False
    # print('batch_mode=False', utils.show_shape(db[0, 5]))
    # db.batch_mode = True
    # print('batch_mode=True', utils.show_shape(db[0, 5]))


def test():
    train_db = ColmapDataset(Path('~/data/NeRF/HyperNeRF/vrig-3dprinter/colmap').expanduser(), split='train')
    test_db = ColmapDataset(Path('~/data/NeRF/HyperNeRF/vrig-3dprinter/colmap').expanduser(), split='test')
    print(train_db)
    print(test_db)
    inputs, targets, infos = train_db[0]
    print('inputs:', utils.show_shape(inputs))
    print('targets:', utils.show_shape(targets))
    print('infos:', utils.show_shape(infos))
    plt.imshow(targets['images'].numpy())
    plt.axis('off')
    plt.show()

    with utils.vis3d:
        utils.vis3d.add_camera_poses(
            train_db.cameras.Tv2w,
            fovy=np.rad2deg(train_db.cameras.FoV[0, 1].item()),
            aspect=train_db.aspect,
            color=(1, 0, 0),
            size=0.1
        )
        utils.vis3d.add_camera_poses(
            test_db.cameras.Tv2w,
            fovy=np.rad2deg(test_db.cameras.FoV[0, 1].item()),
            aspect=test_db.aspect,
            color=(0, 1, 0),
            size=0.1
        )
        utils.vis3d.add_lines(points=[
            [-1., -1, -1],  # 0
            [-1., -1., 1],  # 1
            [-1., 1., -1.],  # 2
            [-1., 1., 1.],  # 3
            [1., -1., -1],  # 4
            [1, -1, 1],  # 5
            [1, 1, -1],  # 6
            [1, 1, 1],  # 7
        ],
            line_index=[[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])


if __name__ == '__main__':
    test()
