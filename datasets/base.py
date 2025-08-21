import functools
import logging
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import List, Union
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor
from torch.utils.data import Dataset as _Dataset

import utils
from utils import Registry, srgb_to_rgb, load_image, ops_3d, Cameras

NERF_DATASETS = Registry(ignore_case=True)
NERF_DATASET_STYLE = Registry(ignore_case=True)


class Dataset(_Dataset, ABC):

    def __init__(
            self,
            root: Path,
            samples=None,
            class_names: List[str] = None,
            transforms=None,
            cache_in_memory=False,
            **kwargs
    ):
        """ Dataset基础类
        使用new_list, new_dict来避免内存泄露问题(见 https://github.com/pytorch/pytorch/issues/13246)

        Args:
            root:
            samples:
            class_names:
            transforms:
            cache_in_memory: 将加载的数据数据放入内存中，避免重复读取文件
            **kwargs:
        """
        self.samples = samples
        self.transforms = transforms  # 数据增强操作
        self.root = root  # 数据集根目录

        if class_names is not None:
            self._class_names = class_names
            self.cls2id = {name: idx for idx, name in enumerate(class_names)}
            self.id2cls = {idx: name for idx, name in enumerate(class_names)}
            self._num_classes = len(class_names)
        else:
            self._class_names = None
            self.cls2id = None
            self.id2cls = None
            self._num_classes = None

        if 'split_ratio' in kwargs:
            self._split_dataset(kwargs.pop('split_ratio'), kwargs.pop('split_id', 0), kwargs.pop('split_seed', 42))

        self._cache_dir = kwargs.pop('cache_dir', '')  # 缓存目录
        self._cache_dir = Path(self._cache_dir).expanduser() if self._cache_dir else None
        self._cache_disk_force = kwargs.pop('cache_disk_force', False)  # 强制重新缓存
        if self._cache_dir is not None:
            self._cache_dir.mkdir(exist_ok=True)

        self._cache = self.manager.dict() if cache_in_memory else None
        if cache_in_memory:
            self._preload()
        if len(kwargs) > 0:
            logging.info(f"{self.__class__.__name__} got unused parameters: {list(kwargs.keys())}")

    @property
    def manager(self):
        if not hasattr(self, '_manager'):
            self._manager = mp.Manager()
        return self._manager

    def _preload(self):
        pass

    @property
    def class_names(self):
        return self._class_names

    @property
    def num_classes(self):
        return self._num_classes

    def set_transforms(self, transforms=None):
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def _split_dataset(self, split_ratio=(1,), split_id=0, split_seed=42):
        """根据比例<split_ratio>将数据集随机划分为若干部分，返回第<split_id>部分"""
        if isinstance(split_seed, int) and split_seed > 0:
            rng = np.random.RandomState(split_seed)
            rng.shuffle(self.samples)  # 随机打乱数据集

        assert 0 <= split_id < len(split_ratio)
        num = len(self.samples)
        sum_ratio = sum(split_ratio)
        start_index = int(num * sum(split_ratio[:split_id]) / sum_ratio)
        end_index = int(num * sum(split_ratio[:split_id + 1]) / sum_ratio)
        self.samples = self.samples[start_index:end_index]

    def __repr__(self):
        s = f"{self.__class__.__name__}{'' if self._cache is None else '[RAM]'}:\n"
        if self._cache_dir is not None:
            s += f"  Cache data in: {self._cache_dir}"
            if self._cache_disk_force:
                s += "[force]"
            s += "\n"
        s += f"  Num Samples: {len(self.samples)}\n"
        if self.num_classes is not None:
            s += f"  Num Categories: {self.num_classes}\n"
        s += f"  Root Location: {self.root}\n"
        es = self.extra_repr()
        if isinstance(es, str):
            es = [es]
        for ss in es:
            if ss:
                s += '  ' + ss + '\n'
        s += "  Transforms: {}\n".format(repr(self.transforms).replace('\n', '\n' + ' ' * 4))
        return s

    def extra_repr(self) -> Union[str, List[str]]:
        return ''

    @staticmethod
    def cache_in_memory(func):
        """缓存得到数据，要求数据是int, float, bool, str, np.ndarray, torch.Tensor,及其用tuple,list, dict形成的组合"""
        func_id = func.__hash__()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cls = args[0]  # type: Dataset
            index = args[1]  # type: int
            # print('get', index, func.__name__)
            if cls._cache is None:
                return func(*args, **kwargs)
            key = (func_id, index)
            # print('key:', key)
            if key not in cls._cache:
                cls._cache[key] = func(*args, **kwargs)
            return deepcopy(cls._cache[key])

        return wrapper

    @staticmethod
    def cache_in_disk(func):
        """缓存数据到磁盘
        要求数据可以使用torch.save存取
        """

        @functools.wraps(func)
        def wrapper(cls: Dataset, index: int, *args, **kwargs):
            if cls._cache_dir is None:
                return func(cls, index, *args, **kwargs)
            filename = f"{cls.__class__.__name__}_{func.__name__}_{index}.cache_data"
            if cls._cache_disk_force or not cls._cache_dir.joinpath(filename).exists():
                data = func(cls, index, *args, **kwargs)
                torch.save(data, cls._cache_dir.joinpath(filename))
            else:
                data = torch.load(cls._cache_dir.joinpath(filename), map_location='cpu')
            return data

        return wrapper

    def new_list(self, *args, **kwargs) -> list:
        return self.manager.list(*args, **kwargs)

    def new_n_list(self, n=0, default=None) -> list:
        return self.manager.list([default for _ in range(n)])

    def new_dict(self, *args, **kwargs) -> dict:
        return self.manager.dict(*args, **kwargs)


class NERF_Base_Dataset(Dataset):
    num_cameras: int = -1
    """ < 0 表示每张图片都有一个相机位姿, > 0 表示相机的数量"""
    cameras: Cameras
    """cameras, shape: [M]"""
    camera_ids: Tensor
    """The index of the camera corresponding to each image, shape: [N], value range: [0, M-1]"""
    num_frames: int
    """How many time-steps in the dataset, ie T"""
    times: Tensor = None
    """The sorted (normalized) times in this dataset, shape: [T]"""
    time_ids: Tensor = None
    """The index of the time-step corresponding to each image, shape: [N], value range: [0, T-1]"""
    scene: str
    images: Tensor
    """images, shape: [N, H, W, 3/4], dtype: uint8 or fp32"""
    image_size: Tuple[int, int]
    """(width, height)"""
    background_type = 'none'
    background: Optional[Tensor] = None
    coord_src: str = 'opengl'
    """The source coordinates system, defalut: opengl"""
    coord_dst: str = 'opengl'
    """The destination coordinate system, defalut: opengl"""

    def __init__(
            self,
            root: Path,
            cameras: Cameras,
            samples=None,
            near=0.1, far=1000., perspective_z01=True,
            batch_mode=False, random_camera=False, with_rays=True,
            **kwargs
    ):
        self.near_far = (near, far)
        self.perspective_z01 = perspective_z01
        self.batch_mode = batch_mode
        self.random_camera = random_camera
        self.with_rays = with_rays
        self.cameras = cameras
        self.cameras.complete(self.near_far, z01=perspective_z01)
        self.cameras.expand(len(samples) if self.num_cameras <= 0 else self.num_cameras)
        super().__init__(root, samples, **kwargs)

    @property
    def near(self):
        return self.near_far[0]

    @property
    def far(self):
        return self.near_far[1]

    def load_images(self, paths: list, image_size=None, downscale: int = None, srgb=False, fp32: bool = True):
        images = []
        for img_path in paths:  # do not sort paths here!!
            img = load_image(img_path)
            if image_size is None and downscale is not None:
                image_size = (img.shape[1] // downscale, img.shape[0] // downscale)
            if image_size is not None and (img.shape[1], img.shape[0]) != image_size:
                img = cv2.resize(img, list(map(int, image_size)), interpolation=cv2.INTER_AREA)  # down scale
            if fp32:
                if img.dtype != np.float32:  # LDR image
                    img = torch.from_numpy(img.astype(np.float32) / 255.)
                    if srgb:
                        img = srgb_to_rgb(img)
                else:  # HDR image
                    img = torch.from_numpy(img.astype(np.float32))
            else:
                if img.dtype != np.uint8:
                    img = np.clip(img * 255., 0, 255).astype(np.uint8)
                img = torch.from_numpy(img)
            images.append(img)
        return torch.stack(images, dim=0)

    def extra_repr(self):
        # theta_range = np.round(np.rad2deg(self.theta_range), 3)
        # phi_range = np.round(np.rad2deg(self.phi_range), 3)
        return [
            f"near={self.near}, far={self.far}",
            # f"camera range: raduis={self.radius_range}, theta={theta_range}, phi={phi_range}",
        ]

    def get_background(self, pixels: Tensor, x_ind=None, y_ind=None) -> Optional[Tensor]:
        if self.background_type == 'none':
            return None
        elif self.background_type == 'black':
            bg = pixels.new_zeros(1)
        elif self.background_type == 'white':
            bg = pixels.new_tensor(255 if pixels.dtype == torch.uint8 else 1.)
        elif self.background_type == 'reference':
            bg = pixels[..., :3].clone()
        elif self.background_type == 'random':
            if pixels.dtype == torch.uint8:
                bg = torch.randint(0, 255, pixels[..., :3].shape, dtype=pixels.dtype, device=pixels.device)
            else:
                bg = torch.rand_like(pixels[..., :3])
        elif self.background_type == 'random2':
            if pixels.dtype == torch.uint8:
                bg = torch.randint(0, 255, (3,), dtype=pixels.dtype, device=pixels.device)
            else:
                bg = torch.rand((3,), dtype=pixels.dtype, device=pixels.device)
        elif self.background_type == 'checker':
            bg = self.background[y_ind, x_ind, :3].expand_as(pixels[..., :3])
        else:
            raise ValueError()
        if bg.ndim != pixels.ndim:
            bg = bg.view(*[1] * (pixels.ndim - bg.ndim), *bg.shape)
        assert bg.ndim == pixels.ndim
        return bg

    def init_background(self, images: Tensor):
        if self.background_type == 'white':
            self.background = images.new_tensor(255 if images.dtype == torch.uint8 else 1)
        elif self.background_type == 'black':
            self.background = images.new_tensor(0)
        elif self.background_type == 'reference':
            self.background = images[..., :3]
        elif self.background_type == 'random':
            if images.dtype == torch.uint8:
                self.background = torch.randint_like(images[0, ..., :3], 0, 255)
            else:
                self.background = torch.rand_like(images[0, ..., :3])
        elif self.background_type == 'random2':
            if images.dtype == torch.uint8:
                self.background = torch.randint(0, 255, (1, 1, 3), dtype=images.dtype)  # .expand_as(images)
            else:
                self.background = torch.rand(1, 1, 3, dtype=images.dtype)  # .expand_as(images)

        elif self.background_type == 'checker':
            N, H, W, C = images.shape
            self.background = torch.from_numpy(utils.image_checkerboard((H, W), 8)).to(images)
        elif self.background_type == 'none':
            self.background = None
        else:
            raise NotImplementedError(f"background type \"{self.background_type}\" is not support")

    def get_image(self, index: Union[int, Tensor]):
        image = self.images[index]
        if image.dtype == torch.uint8:
            image = image.float() / 255.
        if self.background_type in ['random', 'random2'] and image.shape[-1] == 4:
            background = self.get_background(image)
            image = image.clone()
            torch.lerp(background, image[..., :3], image[..., -1:], out=image[..., :3])
        return image

    def camera_ray(self, index, batch_size=None):
        if batch_size is not None:
            index = torch.randint(0, len(self.images), (batch_size,))
        cam_ind = index if self.num_cameras < 0 else self.camera_ids[index]
        Tv2s = self.cameras.Tv2s[cam_ind]
        Tv2w = self.cameras.Tv2w[cam_ind]
        Tw2v = self.cameras.Tw2v[cam_ind]
        Tv2c = self.cameras.Tv2c[cam_ind]
        inputs = {}
        image = self.get_image(index)
        infos = {
            'Tw2v': Tw2v,
            'Tv2s': Tv2s,
            'Tv2c': Tv2c,
            'size': torch.tensor(self.image_size),
            'index': index,
            'campos': Tv2w[..., :3, 3],
            'cam_id': cam_ind,
            # 'focals': self.cameras.focal[cam_ind],
            'FoV': self.cameras.FoV[cam_ind],
        }
        inputs['background'] = self.get_background(image)
        if image.ndim == 4:
            inputs['background'] = inputs['background'].expand(len(image), -1, -1, -1)
        if self.background_type == 'random' and image.shape[-1] == 4:
            torch.lerp(inputs['background'], image[..., :3], image[..., -1:], out=image[..., :3])
        targets = {'images': image}
        if self.times is not None:
            inputs['t'] = self.times[index]
            inputs['time_id'] = self.time_ids[index]
        return inputs, targets, infos

    def __getitem__(self, index=None):
        if isinstance(index, (tuple, list)):
            return self.camera_ray(None, index[1])
        else:
            return self.camera_ray(index)


def find_best_img_dir(root: Path, img_dir: str, goal: float):
    """选择最接近目标下采样倍数的图片目录"""
    downscales = []
    if root.joinpath(img_dir).exists():
        downscales.append((1., img_dir))
    for img_dir_s in root.glob(f"{img_dir}_x*"):
        if img_dir_s.is_dir():
            try:
                scale = float(img_dir_s.name[len(f"{img_dir}_x"):])
                downscales.append((scale, img_dir_s.name))
            except ValueError:
                pass
    assert len(downscales) > 0
    downscales = sorted(downscales)
    best = None
    best_img_dir = img_dir
    for scale, img_dir in downscales:
        if scale == goal:
            best, best_img_dir = scale, img_dir
            break
        if scale > goal:
            if best is None:
                best, best_img_dir = scale, img_dir
            break
        best, best_img_dir = scale, img_dir
    return goal / best, best_img_dir


class DynamceSceneDataset(NERF_Base_Dataset):
    times: Tensor
    camera_ids: Tensor
    time_ids: Tensor
    num_cameras: int
    """ < 0 表示每张图片都有一个相机位姿, > 0 表示相机的数量"""
    num_frames: int
    scene: str

    def get_fovy(self, index: int):
        if self.num_cameras > 0:
            return self.FoV[self.camera_ids[index], 1]
        else:
            return self.FoV[1] if self.FoV.ndim == 1 else self.FoV[index, 1]


if __name__ == '__main__':
    db = NERF_Base_Dataset(Path('.'))
    print(db)
