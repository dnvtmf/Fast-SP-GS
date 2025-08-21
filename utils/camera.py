from typing import Union, Sequence, Tuple

import torch
from torch import Tensor


class Cameras:
    def __init__(
            self,
            Tv2w: Tensor = None,
            Tw2v: Tensor = None,
            Tv2s: Tensor = None,
            Ts2v: Tensor = None,
            Tv2c: Tensor = None,
            FoV: Tensor = None,
            image_size: Tuple[int, int] = None,  # WxH
            focal: Union[float, Tensor] = None,
            pp: Tensor = None,
            near_far: Tuple[float, float] = (0.1, 1000.),
            z01: bool = False,
    ):
        self.Tv2w = Tv2w
        """Transform matrix for view space to world space, shape: [4, 4] or [N, 4, 4]"""
        self.Tw2v = Tw2v
        """Transform matrix for world space to view space, shape: [4, 4] or [N, 4, 4]"""
        self.Tv2s = Tv2s
        """Transform matrix for view space to screen space, shape: [3, 3] or [N, 3, 3]"""
        self.Ts2v = Ts2v
        """Transform matrix for screen space to view space, shape: [3, 3] or [N, 3, 3]"""
        self.Tv2c = Tv2c
        """Transform matrix for view space to clip space, shape: [4, 4] or [N, 4, 4]"""
        self.FoV = FoV
        """ field of view: radians, e.g. 0.25*pi, shape: [2] or [N, 2]"""
        self.focal = focal
        """ focal length, shape: [1], [2], [N, 1] or [N, 2] """
        self.image_size = image_size
        """ image size (W, H)"""
        self.aspect = image_size[0] / image_size[1] if image_size is not None else None
        """aspect of images, ie W/H"""
        self.pp = pp
        """principal points  """
        self.near_far = near_far
        """the value of near and far plane for clip space"""
        self.z01 = z01
        """In clip space, the range of z is [0, 1] or [-1, 1]"""

    def complete(self, near_far: Sequence[float] = None, z01: bool = None):
        from utils import ops_3d
        # Camera Extrinsic
        if self.Tv2w is None:
            assert self.Tw2v is not None
            self.Tv2w = torch.inverse(self.Tw2v)

        if self.Tw2v is None:
            assert self.Tv2w is not None
            self.Tw2v = torch.inverse(self.Tv2w)
        self.Tv2w = self.Tv2w.float()
        self.Tw2v = self.Tw2v.float()
        # Camera Intrinsics
        assert self.image_size is not None
        self.aspect = self.image_size[0] / self.image_size[1]
        if self.FoV is None:
            assert self.focal is not None and self.focal.shape[-1] in [1, 2], f"Can not calculate fovy"
            fovx = ops_3d.focal_to_fov(self.focal[..., 0], self.image_size[0])
            fovy = ops_3d.focal_to_fov(self.focal[..., 0 if self.focal.shape[-1] == 1 else 1], self.image_size[1])
            if isinstance(fovx, Tensor):
                self.FoV = torch.stack([fovx, fovy], dim=-1)
            else:
                self.FoV = torch.tensor([fovx, fovy], dtype=torch.float)
        if self.focal is None:
            assert self.FoV.shape[-1] in [1, 2]
            self.focal = ops_3d.fov_to_focal(self.FoV, self.FoV.new_tensor(self.image_size))

        if self.pp is None:
            self.pp = self.Tw2v.new_tensor(self.image_size) * 0.5
        if self.Tv2s is None and self.Ts2v is None:
            self.Tv2s = ops_3d.camera_intrinsics(self.focal, cx_cy=self.pp, size=self.image_size)
            self.Ts2v = ops_3d.camera_intrinsics(self.focal, cx_cy=self.pp, size=self.image_size, inv=True)
        elif self.Tv2s is None:
            self.Tv2s = torch.inverse(self.Ts2v)
        elif self.Ts2v is None:
            self.Ts2v = torch.inverse(self.Tv2s)  # noqa
        self.Tv2s = self.Tv2s.float()
        self.Ts2v = self.Ts2v.float()
        if near_far is not None:
            self.near_far = near_far
        if z01 is not None:
            self.z01 = z01
        if self.Tv2c is None or near_far is not None or z01 is not None:
            # self.Tv2c = ops_3d.perspective(self.FoV[..., 1], n=near, f=far, size=self.image_size, z01=z01)
            self.Tv2c = ops_3d.perspective2(self.image_size, self.focal, None, self.pp, self.near_far,
                                            self.Tw2v.device, self.z01)
        self.Tv2c = self.Tv2c.float()

    def expand(self, *shape):
        self.Tw2v = self.Tw2v.expand(*shape, 4, 4)
        self.Tv2w = self.Tv2w.expand(*shape, 4, 4)
        self.Tv2s = self.Tv2s.expand(*shape, 3, 3)
        self.Ts2v = self.Ts2v.expand(*shape, 3, 3)
        self.Tv2c = self.Tv2c.expand(*shape, 4, 4)

        self.focal = self.focal.expand(*shape, 2)
        self.FoV = self.FoV.expand(*shape, 2)
        if self.pp is not None:
            self.pp = self.pp.expand(*shape, 2)

    @property
    def shape(self):
        return self.Tw2v.shape[:-2]

    @property
    def device(self):
        return self.Tw2v.device

    @property
    def principal_points(self):
        if self.pp is None:
            return self.Tw2v.new_tensor([self.image_size[0] * 0.5, self.image_size[1] * 0.5])
        else:
            return self.pp

    def __len__(self):
        return 1 if self.Tw2v.ndim == 2 else len(self.Tw2v)

    def __repr__(self):
        return f"Cameras{list(self.Tw2v.shape[:-2])}(image_size={self.image_size})"

    def __getitem__(self, item):
        kwargs = {}
        for name in ['Tv2w', 'Tw2v', 'Tv2s', 'Ts2v', 'Tv2c']:
            v = getattr(self, name)
            kwargs[name] = v[item] if isinstance(v, Tensor) and v.ndim >= 3 else v
        for name in ['pp', 'FoV', 'focal']:
            v = getattr(self, name)
            kwargs[name] = v[item] if isinstance(v, Tensor) and v.ndim >= 2 else v
        return Cameras(image_size=self.image_size, **kwargs)

    def resize_(self, image_size):
        from utils import ops_3d
        sclae = (image_size[0] / self.image_size[0], image_size[1] / self.image_size[1])
        self.image_size = image_size
        self.aspect = image_size[0] / image_size[1]
        self.pp = self.pp * self.pp.new_tensor(sclae)
        self.focal = ops_3d.fov_to_focal(self.FoV, self.FoV.new_tensor(self.image_size))
        self.Tv2s = ops_3d.camera_intrinsics(self.focal, cx_cy=self.pp, size=self.image_size)
        self.Ts2v = ops_3d.camera_intrinsics(self.focal, cx_cy=self.pp, size=self.image_size, inv=True)
        return self

    def pad_(self, padding: Sequence[int], *args, **kwargs):
        """

        Args:
            padding: [left, top, right, bottom] or [-x, -y, +x, +y]
            *args:
            **kwargs:

        Returns:

        """
        from utils import ops_3d
        assert len(padding) == 4
        if self.pp is None:
            self.pp = self.Tw2v.new_tensor(self.image_size) * 0.5
        self.pp += self.pp.new_tensor(padding[:2])
        if self.focal is None:
            self.focal = ops_3d.fov_to_focal(self.FoV, self.FoV.new_tensor(self.image_size))
        self.image_size = (self.image_size[0] + padding[0] + padding[2], self.image_size[1] + padding[1] + padding[3])
        if self.FoV is not None:
            self.FoV = ops_3d.focal_to_fov(self.focal, self.image_size)
        if self.Tv2c is not None:
            self.Tv2c = ops_3d.perspective2(self.image_size, self.focal, None, self.pp, self.near_far,
                                            self.Tw2v.device, self.z01)
        if self.Tv2s is not None:
            self.Tv2s = ops_3d.camera_intrinsics(self.focal, cx_cy=self.pp, size=self.image_size)
            self.Ts2v = ops_3d.camera_intrinsics(self.focal, cx_cy=self.pp, size=self.image_size, inv=True)
        return self

    def to(self, device, dtype=None, *args, **kwargs):
        for attr in self.__dir__():
            if attr == 'principal_points':
                continue
            if isinstance(getattr(self, attr), Tensor):
                setattr(self, attr, getattr(self, attr).to(device, dtype, *args, **kwargs))
        return self
