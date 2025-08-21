from pathlib import Path
from typing import Optional

import numpy as np
import torch

from datasets.base import NERF_DATASET_STYLE, NERF_DATASETS, NERF_Base_Dataset, find_best_img_dir
import utils
from utils import ops_3d, Cameras

NERF_DATASETS['DyNeRF'] = {
    'common': {
        'style': 'DyNeRFDataset',
        'root': 'NeRF/DyNeRF',
        'img_dir': 'images',
        'img_suffix': '.png',
        'camera_file': 'poses_bounds.npy',
        'use_time': True,
        'downscale': 2,
        'background': 'white',
        'scene': 'sear_steak',
        'coord_src': 'opengl',
        'coord_dst': 'opengl',
    },
    'train': {'split': 'train'},
    'eval': {'split': 'val'},
    'test': {'split': 'test'},
    'DyNeRF': ['train', 'eval', 'test'],
}


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


@NERF_DATASET_STYLE.register()
class DyNeRFDataset(NERF_Base_Dataset):
    """ for the dataset of DyNeRF
    reference: Neural 3D Video Synthesis from Multi-View Video, CVPR 2022
    dataset: https://github.com/facebookresearch/Neural_3D_Video
    """

    def __init__(
            self,
            root: Path,
            scene='',
            camera_file='poses_bounds.npy',
            split='train',
            img_dir='images',
            img_suffix='.png',
            mask_dir='',
            mask_suffix='',
            background='white',
            image_size=None,
            downscale: int = None,
            sample_stride=1,
            random_camera=False,
            batch_mode=True,
            coord_src='llff',
            coord_dst='opengl',
            use_time=True,
            with_rays=True,
            bd_factor: Optional[float] = 0.75,
            max_num_frame=300,
            debug=False,
            near=0,
            far=1000.,
            pose_scale_offset=(1.0, 0.),
            **kwargs
    ):
        root = root.joinpath(scene)
        self.root = root
        self.scene = scene
        assert img_suffix in utils.image_extensions
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.ndc = False  # normalized device coordinates
        self.random_camera = random_camera
        self.batch_mode = batch_mode
        self.downscale = downscale
        self.sample_stride = sample_stride
        self.split = split

        self.mask_dir = mask_dir
        self.coord_src = ops_3d.coordinate_system[coord_src.lower()]
        self.coord_dst = ops_3d.coordinate_system[coord_dst.lower()]
        ops_3d.set_coord_system(self.coord_dst)
        self.with_rays = with_rays

        camera_names = sorted(list([cam_name.name for cam_name in root.glob(f"cam*") if cam_name.is_dir()]))
        assert len(camera_names) > 0
        # For all the dataset, cam00.mp4 is the center reference camera which we held out for testing.
        if self.split == 'train':
            camera_names = camera_names[1:]
        else:
            camera_names = camera_names[:1]
        self.camera_indices = [int(camera_name[3:]) for camera_name in camera_names]

        downscale, self.img_dir = find_best_img_dir(root.joinpath(camera_names[0]), img_dir, downscale)
        paths = []
        camera_ids = []
        time_ids = []
        times = []
        num_frames = 0
        for i, camera_name in enumerate(camera_names):
            images_names = sorted(list(root.joinpath(camera_name, self.img_dir).glob(f'*{img_suffix}')))
            if max_num_frame > 0:
                images_names = images_names[:max_num_frame]
            assert num_frames == 0 or num_frames == len(images_names), f"{root.joinpath(camera_name, self.img_dir)}"
            num_frames = len(images_names)
            paths.extend(images_names)
            times.extend(range(num_frames))
            camera_ids.extend([i] * num_frames)
            time_ids.extend(range(num_frames))
        self.camera_ids = torch.tensor(camera_ids)
        self.time_ids = torch.tensor(time_ids)

        print(f'begin loading {len(paths)} images...')
        # TODO: 多线程加载数据
        if debug:
            self.images = torch.zeros(len(paths), 1014, 1352, 3)  # for debug
        else:
            self.images = self.load_images(paths, image_size, downscale, fp32=False)
        print('load images successful')
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
        self.num_frames = num_frames
        self.num_cameras = len(self.camera_indices)
        self.times = torch.tensor(times) / (self.num_frames - 1) if use_time else None  # * 2 - 1.
        ## Load Cameras
        cameras, near_fars = self.load_cameras(root, camera_file, pose_scale_offset, bd_factor)
        cameras.image_size = self.image_size
        cameras.Tv2w = cameras.Tv2w[1:] if self.split == 'train' else cameras.Tv2w[:1]
        cameras.FoV = cameras.FoV[1:] if self.split == 'train' else cameras.FoV[:1]
        near = 0. if self.ndc else float(near_fars.min() * .9)
        far = 1. if self.ndc else float(near_fars.max() * 1.)

        self.background_type = background
        self.init_background(self.images)
        if self.background_type != 'random' and self.background_type != 'none' and self.images.shape[-1] == 4:
            torch.lerp(self.background, self.images[..., :3], self.images[..., -1:], out=self.images[..., :3])

        # self.camera_radio_scale = camera_radiu_scale
        # self.Tv2w[:, :3, 3] = (self.Tv2w[:, :3, 3] * camera_radiu_scale + 0)

        # self.camera_noise = camera_noise
        # if camera_noise > 0:
        #     self.Tv2w = ops_3d.rigid.lie_to_Rt(torch.randn(len(self.Tv2w), 6) * self.camera_noise) @ self.Tv2w
        self.scene_size = 9
        self.scene_center = 0  # [-4.5, 4.5]
        super().__init__(root, cameras, self.images, near=near, far=far, **kwargs)

    def load_cameras(self, root, camera_file, pose_scale_offset=(1.0, 0.), bd_factor=0.75):
        assert camera_file == 'poses_bounds.npy'
        self.camera_file = camera_file
        poses = np.load(root.joinpath(camera_file).as_posix())  # type: np.ndarray
        # format see https://github.com/Fyusion/LLFF Nx17 --> Nx[3x5+2] 3x5: [Tv2w, [H, W, focal]], 2: [near, far]
        near_fars = torch.from_numpy(poses[:, -2:].astype(np.float32))
        poses = poses[:, :-2].reshape([-1, 3, 5])
        assert np.all(poses[:, :3, 4] == poses[0:1, :3, 4]), "hwf must be same for all poses"
        fovy = torch.from_numpy(ops_3d.focal_to_fov(poses[:, 2, 4], poses[:, 0, 4])).float()
        FoV = torch.stack([ops_3d.fovx_to_fovy(fovy, 1. / self.aspect), fovy], dim=1)
        # poses = np.concatenate([poses[..., 1:2], -poses[..., 0:1], poses[..., 2:4]], axis=-1)  # llff to opengl
        near_original = near_fars.min()
        scale_factor = near_original * bd_factor
        near_fars /= scale_factor
        poses[..., 3] /= scale_factor
        poses, _ = center_poses(poses[:, :, :4], np.eye(4))

        Tv2w = ops_3d.to_4x4(torch.from_numpy(poses[..., :3, :4]).float())
        Tv2w = ops_3d.convert_coord_system(Tv2w, self.coord_src, self.coord_dst, inverse=True)
        Tv2w = ops_3d.camera_translate_scale(Tv2w, scale=pose_scale_offset[0], translate=pose_scale_offset[1])
        cameras = Cameras(Tv2w=Tv2w, FoV=FoV)
        return cameras, near_fars

    def __getitem__(self, index=None):
        if isinstance(index, int):
            return self.camera_ray(index)
        elif isinstance(index, tuple):
            if self.batch_mode:
                return self.camera_ray(None, index[1])
            else:
                return self.random_ray(None, index[1])
        else:
            raise RuntimeError()

    def extra_repr(self):
        s = [
            f"images: cam*/{self.img_dir}/*{self.img_suffix}",
            f"mask_dir: {self.mask_dir}, mask_suffix: {self.mask_suffix}" if self.mask_suffix else None,
            f"camera_file: {self.camera_file}, coord system: {self.coord_src}→{self.coord_dst}",
            f"cameras: {self.camera_indices}",
            f"image size{'' if self.downscale is None else f'↓{self.downscale}'}="
            f"{self.image_size[0]} x {self.image_size[1]}, focal={utils.float2str(self.cameras.focal.mean().item())}",
            f"background={self.background_type}",
            f"num_frames: {self.num_frames}, num_cameras: {self.num_cameras}",
            # f"camera_radius_scale={self.camera_radius_scale}" if self.camera_radius_scale != 1.0 else None,
            # f"camera noise: {self.camera_noise}" if self.camera_noise > 0 else None,
        ]
        return super().extra_repr() + s


def test():
    import matplotlib.pyplot as plt
    utils.set_printoptions()
    cfg = {**NERF_DATASETS['DyNeRF']['common'], **NERF_DATASETS['DyNeRF']['train']}
    cfg['root'] = Path('~/data', cfg['root']).expanduser()
    cfg['scene'] = 'cut_roasted_beef'
    # cfg['scene'] = 'coffee_martini'
    cfg['with_rays'] = True
    cfg['max_num_frame'] = 10
    cfg['coord_src'] = 'llff'
    cfg['coord_dst'] = 'opengl'
    db = DyNeRFDataset(**cfg)
    print(db)
    print('camera_id:', torch.bincount(db.camera_ids))
    print(utils.show_shape(db.cameras.Tv2w, db.times, db.camera_ids, db.time_ids))
    print(utils.show_shape(db.camera_ray(0)))

    inputs, targets, infos = db.camera_ray(0)
    rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
    inputs, targets, infos = db.camera_ray(1)
    print(inputs['time_id'])
    print(infos['campos'])
    print(infos['cam_id'])

    last_cam_idx = (db.camera_ids == db.camera_ids.max()).nonzero().min().item()
    plt.subplot(131)
    plt.imshow(targets['images'][..., :3].numpy())
    plt.subplot(132)
    plt.imshow(inputs['background'].expand_as(targets['images'][..., :3]).numpy())
    plt.subplot(133)
    print(last_cam_idx, db.camera_ids[last_cam_idx])
    plt.imshow(db.images[last_cam_idx, ..., :3].numpy())
    # plt.imshow(torch.lerp(inputs['background'], targets['images'][..., :3], targets['images'][..., 3:]))
    plt.show()

    with utils.vis3d:
        utils.vis3d.add_camera_poses(db.cameras.Tv2w, fovy=np.rad2deg(db.cameras.FoV[0, 1].item()), aspect=db.aspect,
                                     color=(1, 0, 0))
        utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2)[40::80, 40::80])
        inputs = db.random_ray(None, 512)[0]
        rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
        utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2), color=(0.1, 0.1, 0.1))
        inputs = db.random_ray(last_cam_idx, 10)[0]
        rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
        utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2), color=(0.3, 0.3, 0.3))
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
    db.batch_mode = False
    print('batch_mode=False', utils.show_shape(db[0, 5]))
    # db.batch_mode = True
    # print('batch_mode=True', utils.show_shape(db[0, 5]))


if __name__ == '__main__':
    test()
