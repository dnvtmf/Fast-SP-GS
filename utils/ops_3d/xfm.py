"""转变点或向量"""
from typing import Union, Any

import numpy as np
import torch
from torch import Tensor
import torch.utils.cpp_extension
import pytest
from torch.amp import custom_bwd, custom_fwd

from fast_2d_gs._C import get_C_function, try_use_C_extension, get_python_function

__all__ = ['xfm', 'xfm_vectors', 'apply', 'pixel2points']


class _xfm_func(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, points, matrix, isPoints=True, to_homo=False):  # noqa
        ctx.save_for_backward(points, matrix)
        ctx.isPoints = isPoints
        ctx.to_homo = to_homo
        return get_C_function('xfm_fwd')(points, matrix, isPoints, to_homo)

    @staticmethod
    @torch.autograd.function.once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        points, matrix = ctx.saved_variables
        grad_points, grad_matrix = get_C_function('xfm_bwd')(
            dout, points, matrix, ctx.isPoints, ctx.to_homo, ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        )
        return grad_points, grad_matrix, None, None


@try_use_C_extension(_xfm_func.apply, "xfm_fwd", 'xfm_bwd')
def _xfm(points: Tensor, matrix: Tensor, is_points=True, to_homo=False) -> Tensor:
    dim = points.shape[-1]
    if dim + 1 == matrix.shape[-1]:
        points = torch.constant_pad_nd(points, (0, 1), 1.0 if is_points else 0.0)
    else:
        to_homo = False
    if is_points:
        out = torch.matmul(points, torch.transpose(matrix, -1, -2))
    else:
        out = torch.matmul(points, torch.transpose(matrix, -1, -2))
    if not to_homo:
        out = out[..., :dim]
    return out


def xfm(points: Tensor, matrix: Tensor, homo=False) -> Tensor:
    """Transform points.
    Args:
        points: Tensor containing 3D points with shape [..., num_vertices, 3] or [..., num_vertices, 4]
        matrix: A 4x4 transform matrix with shape [..., 4, 4] or [4, 4]
        homo: convert output to homogeneous
    Returns:
        Tensor: Transformed points in homogeneous 4D with shape [..., num_vertices, 3/4]
    """
    return _xfm(points, matrix, True, homo)


def apply(points: Tensor, matrix: Tensor, homo=False, is_points=True) -> Tensor:
    """Transform points p' = M @ p.
    Args:
       points: Tensor containing 3D points with shape [...,  C]
       matrix: A transform matrix with shape [..., C, C] or [..., C+1, C+1]
       homo: convert output to homogeneous
       is_points: is points or vector?
    Returns:
       Tensor: Transformed points with shape [..., C] or [..., C+1]
    """
    dim = points.shape[-1]
    if dim + 1 == matrix.shape[-1]:
        points = torch.constant_pad_nd(points, (0, 1), 1.0 if is_points else 0.0)
    else:
        homo = False
    # out = torch.einsum('...ij,...j->...i', matrix, points)
    out = torch.sum(matrix * points[..., None, :], dim=-1)
    if not homo:
        out = out[..., :dim]
    return out


def xfm_vectors(vectors: Tensor, matrix: Tensor, to_homo=True) -> Tensor:
    """Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [..., num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [..., 4, 4] or [4, 4]

    Returns:
        Tensor: Transformed vectors in homogeneous 4D with shape [..., num_vertices, 3/4].
    """
    return _xfm(vectors, matrix, False, to_homo)


class _pixel2points_cu(torch.autograd.Function):
    _forward_func = get_C_function('pixel2points_forward')
    _backward_func = get_C_function('pixel2points_backward')

    @staticmethod
    def forward(ctx, *inputs):
        depths, pixels, Ts2v, Tv2w = inputs
        points = _pixel2points_cu._forward_func(depths, pixels, Ts2v, Tv2w)
        ctx.save_for_backward(depths, pixels, Ts2v, Tv2w)
        return points

    @staticmethod
    def backward(ctx, *grad_outputs):
        depths, pixels, Ts2v, Tv2w = ctx.saved_tensors
        grad_depths = torch.zeros_like(depths) if ctx.needs_input_grad[0] else None
        grad_pixels = torch.zeros_like(pixels) if ctx.needs_input_grad[1] else None
        grad_Ts2v = torch.zeros_like(Ts2v) if ctx.needs_input_grad[2] else None
        grad_Tv2w = torch.zeros_like(Tv2w) if ctx.needs_input_grad[3] else None
        _pixel2points_cu._backward_func(
            depths, pixels, Ts2v, Tv2w, grad_outputs[0],
            grad_depths, grad_pixels, grad_Ts2v, grad_Tv2w
        )
        return grad_depths, grad_pixels, grad_Ts2v, grad_Tv2w


def _pixel2points_py(
        depth: Tensor, Tv2s: Tensor = None, Ts2v: Tensor = None, Tw2v: Tensor = None, Tv2w: Tensor = None,
        pixel: Tensor = None
) -> Tensor:
    """convert <pixel, depth> to 3D point"""
    if pixel is None:
        H, W = depth.shape[-2:]
        pixel = torch.stack(
            torch.meshgrid(
                torch.arange(W, device=depth.device, dtype=depth.dtype),
                torch.arange(H, device=depth.device, dtype=depth.dtype),
                indexing='xy'
            ), dim=-1
        )  # shape: [H, W, 2]
    if Tv2s is not None:
        Ts2v = Tv2s.inverse()
    xyz = torch.cat([pixel, torch.ones_like(pixel[..., :1])], dim=-1) * depth[..., None]  # [..., H, W, 3]
    xyz = apply(xyz, Ts2v[..., None, None, :, :])
    if Tv2w is not None:
        xyz = apply(xyz, Tv2w[..., None, None, :, :])
    elif Tw2v is not None:
        xyz = apply(xyz, Tw2v.inverse()[..., None, None, :, :])
    return xyz


def pixel2points(
        depth: Tensor, Tv2s: Tensor = None, Ts2v: Tensor = None, Tw2v: Tensor = None, Tv2w: Tensor = None,
        pixel: Tensor = None
) -> Tensor:
    """convert <pixel, depth> to 3D point
    Args:
        depth: shape [..., H, W]
        pixel: shape [..., H, W, 2]
        Tv2s: shape [..., 3, 3]
        Ts2v: shape [..., 3, 3]
        Tw2v: shape [..., 4, 4]
        Tv2w: shape [..., 4, 4]
    Returns:
        Tensor: 3D points, shape:[..., H, W, 3]
    """
    if depth.is_cuda:
        if Ts2v is None:
            Ts2v = Tv2s.inverse()
        if Tv2w is None and Tw2v is not None:
            Tv2w = Tw2v.inverse()
        shapes = [depth.shape[:-2], Ts2v.shape[:-2]]
        if pixel is not None:
            shapes.append(pixel.shape[:-3])
        if Tv2w is not None:
            shapes.append(Tv2w.shape[:-3])
        shapes = torch.broadcast_shapes(*shapes)
        H, W = depth.shape[-2:]
        depth = depth.expand(*shapes, H, W).view(-1, H, W).float().contiguous()
        if pixel is not None:
            pixel = pixel.expand(*shapes, H, W, 2).view(-1, H, W, 2).float().contiguous()
        Ts2v = Ts2v.expand(*shapes, 3, 3).view(-1, 3, 3).contiguous()
        if Tv2w is not None:
            Tv2w = Tv2w.expand(*shapes, 4, 4).view(-1, 4, 4).contiguous()
        points = _pixel2points_cu.apply(depth, pixel, Ts2v, Tv2w)
        return points.view(*shapes, H, W, 3)
    else:
        return _pixel2points_py(depth, Tv2s, Ts2v, Tw2v, Tv2w, pixel)


def camera_distort(pixels: Union[np.ndarray, Tensor], k1=0., k2=0., k3=0., p1=0., p2=0.):
    """
    deal camera distortion, include radial_distortion (k1, k2, k3) and tangential_distortion (p1, p2)
    pixels should in normailize space, ie z=1
    """
    x, y = pixels.unbind(-1)
    r2 = x * x + y * y
    c = 1 + (k1 + (k2 + k3 * r2) * r2) * r2
    xd = x * c + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * c + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    if isinstance(pixels, np.ndarray):
        return np.stack([xd, yd], axis=-1)
    else:
        return torch.stack([xd, yd], dim=-1)


def camera_undistort(pixels_d: Tensor, k1=0., k2=0., k3=0., p1=0., p2=0., eps: float = 1e-9, max_iterations=10):
    def _compute_residual_and_jacobian(x, y, xd, yd):
        """Auxiliary function of radial_and_tangential_undistort()."""

        r = x * x + y * y
        d = 1.0 + r * (k1 + r * (k2 + k3 * r))

        fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
        fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

        # Compute derivative of d over [x, y]
        d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
        d_x = 2.0 * x * d_r
        d_y = 2.0 * y * d_r

        # Compute derivative of fx over x and y.
        fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
        fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

        # Compute derivative of fy over x and y.
        fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
        fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

        return fx, fy, fx_x, fx_y, fy_x, fy_y

    step_x, step_y = 0, 0
    xd, yd = pixels_d.unbind(-1)
    x, y = xd.clone(), yd.clone()
    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(x=x, y=y, xd=xd, yd=yd)
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(torch.abs(denominator) > eps, x_numerator / denominator, torch.zeros_like(denominator))
        step_y = torch.where(torch.abs(denominator) > eps, y_numerator / denominator, torch.zeros_like(denominator))
        x = x + step_x
        y = y + step_y

    return torch.stack([x, y], dim=-1)
