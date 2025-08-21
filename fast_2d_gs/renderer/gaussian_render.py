"""
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code: https://github.com/graphdeco-inria/gaussian-splatting
"""
import math
from typing import NamedTuple, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.amp import custom_bwd, custom_fwd

from fast_2d_gs._C import try_use_C_extension, get_python_function, get_C_function
from utils import ops_3d

__all__ = [
    'compute_cov2D', 'compute_cov3D', 'build_rotation', 'build_scaling_rotation', 'preprocess', 'render',
    'RasterizeBuffer', 'topk_weights',
]

_BLOCK_X = 16
_BLOCK_Y = 16


def _cpu_deep_copy_tuple(input_tuple):
    copied_tensors = []
    for item in input_tuple:
        if isinstance(item, Tensor):
            copied_tensors.append(item.to(torch.device('cpu'), non_blocking=True).clone())
        else:
            copied_tensors.append(item)
    return tuple(copied_tensors)


class RasterizeBuffer(NamedTuple):
    W: int
    """output image width"""
    H: int
    """output image width"""
    P: int
    """number of gaussian"""
    R: int
    """number of rendered"""
    tile_ranges: Tensor
    point_list: Tensor
    n_contribed: Tensor
    conic_opacity: Tensor
    means2D: Tensor
    opacity: Tensor


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(
            ctx,
            means3D,
            means2D,
            sh,
            sh_rest,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            cov2Ds_precomp,
            extras,
            image_size,
            sh_degree,
            Tw2v,
            Tv2c,
            cam_pos,
            tanFoV,
            is_opengl,
            culling, accum_max_count, accum_weights_p, accum_weights_count,
            *other_extras
    ):
        # Restructure arguments the way that the C++ lib expects them
        inputs = (
            image_size[1], image_size[0], sh_degree, False, is_opengl,
            means3D, opacities, scales, rotations, sh, sh_rest, extras,
            Tw2v, Tv2c, cam_pos, tanFoV,
            cov3Ds_precomp, cov2Ds_precomp, means2D,
        )

        # Invoke C++/CUDA rasterizer
        # if False:
        #     torch.cuda.synchronize()
        #     cpu_args = _cpu_deep_copy_tuple(inputs)  # Copy them before they can be corrupted
        #     try:
        #         (num_rendered, pixels, out_opaticy, out_extra, cov3Ds, colors, conic_opacity, radii, tile_ranges,
        #         point_list, n_contrib) = get_C_function('rasterize_gaussians')(*inputs)
        #         torch.cuda.synchronize()
        #     except Exception as ex:
        #         torch.save(cpu_args, "snapshot_fw.dump")
        #         print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
        #         raise ex
        # else:
        (num_rendered, pixels, out_opaticy, out_extra, cov3Ds, colors, conic_opacity, radii, tile_ranges,
         point_list, n_contrib) = get_C_function('rasterize_gaussians')(*inputs)

        buffer = RasterizeBuffer(
            image_size[0], image_size[1], len(opacities), num_rendered,
            tile_ranges, point_list, n_contrib, conic_opacity, pixels, out_opaticy
        )
        pixel_extras = []
        for extra_i in other_extras:
            pixel_extras.append(
                get_C_function('gaussian_rasterize_extra_forward')(
                    buffer.W, buffer.H, extra_i, means2D, conic_opacity, tile_ranges, point_list, n_contrib
                )
            )
        # Keep relevant tensors for backward
        ctx._settings = (image_size, tanFoV, sh_degree, is_opengl, cov3Ds_precomp is None)
        ctx.save_for_backward(
            means3D, scales, rotations, opacities, sh, sh_rest, extras,
            cov3Ds if cov3Ds_precomp is None else cov3Ds_precomp,
            Tw2v, Tv2c, cam_pos,
            tile_ranges, point_list, n_contrib,
            means2D, conic_opacity, radii, out_opaticy,
            *other_extras
        )
        return pixels, out_opaticy, out_extra, radii, buffer, *pixel_extras

    @staticmethod
    @custom_bwd(device_type='cuda')
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        grad_out_color, grad_out_opacity, grad_out_extra = grad_outputs[:3]
        grad_out_extras = grad_outputs[5:]
        # Restore necessary values from context
        image_size, tan_fov, sh_degree, is_opengl, no_cov3Ds = ctx._settings
        means3D, scales, rotations, opacities, sh, sh_rest, extras, cov3Ds = ctx.saved_tensors[:8]
        Tw2v, Tv2c, cam_pos, = ctx.saved_tensors[8:11]
        tile_ranges, point_list, n_contrib, = ctx.saved_tensors[11:14]
        means2D, conic_opacity, radii, opacity = ctx.saved_tensors[14:18]
        other_extras = ctx.saved_tensors[18:]

        grad_means3D = torch.zeros_like(means3D)
        grad_scales = None
        grad_rotations = None
        # if cov2D is not None:
        #     grad_cov = torch.zeros_like(cov2D) if ctx.needs_input_grad[-2] else None
        if no_cov3Ds:
            grad_scales = torch.zeros_like(scales) if ctx.needs_input_grad[5] else None
            grad_rotations = torch.zeros_like(rotations) if ctx.needs_input_grad[6] else None
            grad_cov = torch.zeros_like(cov3Ds)
        else:
            grad_cov = torch.zeros_like(cov3Ds) if ctx.needs_input_grad[7] else None
        grad_opacities = torch.zeros_like(opacities)
        grad_colors, grad_shs = None, None
        if sh_degree >= 0:
            if ctx.needs_input_grad[3]:
                grad_colors = torch.zeros_like(sh)
        elif ctx.needs_input_grad[2]:
            grad_shs = torch.zeros_like(sh)
        grad_Tw2v = torch.zeros_like(Tw2v) if ctx.needs_input_grad[11] else None
        grad_campos = torch.zeros_like(cam_pos) if ctx.needs_input_grad[13] else None
        grad_extra = torch.zeros_like(extras) if extras is not None and ctx.needs_input_grad[8] else None

        grad_means2D = torch.zeros_like(means2D) if ctx.needs_input_grad[1] else None
        grad_conic = None
        grad_extras = []
        assert len(grad_out_extras) == len(other_extras)
        for i, extra_i in enumerate(other_extras):
            if grad_out_extras[i] is None:
                grad_extras.append(None)
                continue
            grad_extra_i, grad_means2D, grad_conic = get_C_function('gaussian_rasterize_extra_backward')(
                image_size[0], image_size[1],
                extra_i, opacity,  # inputs, outputs
                grad_out_extras[i],  # grad_outputs
                means2D, conic_opacity, tile_ranges, point_list, n_contrib,  # buffer
                grad_means2D, grad_conic  # grad_inputs
            )
            grad_extras.append(grad_extra_i)
        # if ctx.raster_settings.detach_other_extra:
        #     grad_means2D, grad_conic, grad_opacity = None, None, None
        inputs = (
            sh_degree, False, is_opengl,
            Tw2v, Tv2c, cam_pos, tan_fov[0], tan_fov[1],
            means3D, scales, rotations, sh, sh_rest, extras, cov3Ds, None,  # inputs
            tile_ranges, point_list, n_contrib,
            means2D, conic_opacity, radii, opacity,  # outputs
            grad_out_color, grad_out_opacity, grad_out_extra,  # grad_outputs
            grad_means2D, grad_conic,  # grad_internal
            grad_means3D, grad_scales, grad_rotations, grad_opacities, grad_shs, grad_extra,  # grad inputs
            grad_Tw2v, grad_campos, grad_colors, grad_cov,  # grad camera; grad pre-computed
        )
        # Compute gradients for relevant tensors by invoking backward method
        if False:
            for x in inputs:
                if isinstance(x, Tensor):
                    x.sum()
            torch.cuda.synchronize()
            cpu_args = _cpu_deep_copy_tuple(inputs)  # Copy them before they can be corrupted
            try:
                assert 0 <= point_list.amin() and point_list.amax() < means3D.shape[0]
                assert 0 <= tile_ranges.amin() and tile_ranges.amax() <= point_list.shape[0]
                get_C_function("rasterize_gaussians_backward")(*inputs)
                torch.cuda.synchronize()
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occurred in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            get_C_function("rasterize_gaussians_backward")(*inputs)
        grads = (
            grad_means3D,
            grad_means2D,
            grad_shs,
            grad_colors,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov if no_cov3Ds and ctx.needs_input_grad[7] else None,
            grad_extra,
            None, None,
            grad_Tw2v, None, grad_campos, None, None,
            *grad_extras
        )
        return grads


def RasterizeGaussians(
        means3D: Tensor,
        means2D: Optional[Tensor],
        colors: Tensor,
        sh_rest: Optional[Tensor],
        opacities: Tensor,
        scales: Optional[Tensor],
        rotations: Optional[Tensor],
        cov3Ds_precomp: Optional[Tensor],
        cov2Ds_precomp: Optional[Tensor],
        extras: Optional[Tensor],
        image_size: Tuple[int, int],
        sh_degree: int,
        Tw2v: Tensor,
        Tv2c: Tensor,
        cam_pos: Tensor,
        tanFoV: Tensor,
        is_opengl: bool,
        culling: Optional[Tensor],
        accum_max_count: Optional[Tensor],
        accum_weights_p: Optional[Tensor],
        accum_weights_count: Optional[Tensor],
        *other_extras: Tensor,
):
    W, H = image_size
    means2D, depths, colors, radii, tiles_touched, conic_opacity = _preprocess.apply(
        means3D, scales, rotations, opacities, colors, sh_rest,
        Tw2v, Tv2c, cam_pos, tanFoV,
        image_size[0], image_size[1], sh_degree, is_opengl,
        cov3Ds_precomp, cov2Ds_precomp, means2D, culling
    )
    tile_ranges, point_list = GS_prepare(W, H, means2D, depths, radii, tiles_touched, False)
    # tile_ranges, point_list = get_C_function("GS_prepare_v2")(W, H, means2D, depths, radii, tiles_touched, False)
    out_pixels, out_opacities, out_extras, n_contrib = GS_render(
        W, H, means2D, colors, conic_opacity, tile_ranges, point_list, extras,
        accum_max_count, accum_weights_p, accum_weights_count
    )
    buffer = RasterizeBuffer(
        image_size[0], image_size[1], len(opacities), point_list.shape[0],
        tile_ranges, point_list, n_contrib, conic_opacity, means2D, out_opacities
    )
    out_other_extras = []
    for extras_ in other_extras:
        out_other_extras.append(
            _render_extra.apply(
                W, H, extras_, means2D, conic_opacity, tile_ranges, point_list, n_contrib, out_opacities
            )
        )
    return out_pixels, out_opacities, out_extras, radii, buffer, *out_other_extras


def render(
        Tw2v: Tensor,
        Tv2c: Tensor,
        size: Tuple[int, int],
        points: Tensor,
        opacity: Tensor,
        scales: Tensor = None,
        rotations: Tensor = None,
        campos: Tensor = None,
        FoV: Tensor = None,
        focals: Tensor = None,
        cov3Ds: Tensor = None,
        cov2Ds: Tensor = None,
        sh_features: Tensor = None,
        sh_features_rest: Tensor = None,
        colors: Tensor = None,
        sh_degree=0,
        is_opengl=False,
        culling: Tensor = None,
        accum_max_count: Tensor = None,
        accum_weights_p: Tensor = None,
        accum_weights_count: Tensor = None,
        **kwargs
):
    """ Render the scene."""
    means3D = points
    means2D = torch.zeros_like(points[:, :2]).contiguous().requires_grad_()
    if campos is None:
        campos = Tw2v.inverse()[:3, 3].contiguous()
    if FoV is None:
        tanFoV = 0.5 * torch.as_tensor(size, device=Tw2v.device) / focals
    else:
        tanFoV = torch.tan(FoV * 0.5)
    extras = None
    extra_key = None
    for k, v in kwargs.items():
        if v.shape[-1] <= 4:
            extras = v
            extra_key = k
            kwargs.pop(k)
            break
    # outputs = _RasterizeGaussians.apply(
    outputs = RasterizeGaussians(
        means3D.contiguous(),
        means2D,
        sh_features if colors is None else colors.contiguous(),
        sh_features_rest.contiguous() if sh_features_rest is not None else None,
        opacity.contiguous(),
        scales.contiguous() if scales is not None else None,
        rotations.contiguous() if rotations is not None else None,
        cov3Ds.contiguous() if cov3Ds is not None else None,
        cov2Ds.contiguous() if cov2Ds is not None else None,
        extras.contiguous() if extras is not None else None,
        size,
        sh_degree if colors is None else -1,
        Tw2v.view(4, 4).contiguous(),
        Tv2c.view(4, 4).contiguous(),
        campos.view(3).contiguous(),
        tanFoV.view(2),
        is_opengl,
        culling,
        accum_max_count,
        accum_weights_p,
        accum_weights_count,
        *kwargs.values()
    )
    rendered_image, rendered_opacity, out_extra, radii, buffer = outputs[:5]
    output_extras = {k: v for k, v in zip(kwargs.keys(), outputs[5:])}
    if extra_key is not None:
        output_extras[extra_key] = out_extra
    # return color, opaticy, radii, buffer, output_extras

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "images": rendered_image,
        "opacity": rendered_opacity,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        'buffer': buffer,
        **output_extras
    }


def topk_weights(topk: int, buffer: RasterizeBuffer) -> Tuple[Tensor, Tensor]:
    """return topk indices and weights """
    return get_C_function('gaussian_topk_weights')(
        topk, buffer.W, buffer.H, buffer.P, buffer.R, buffer.means2D,
        buffer.conic_opacity, buffer.tile_ranges, buffer.point_list
    )


class _compute_cov3D(torch.autograd.Function):
    _forward = get_C_function('gs_compute_cov3D_forward')
    _backward = get_C_function('gs_compute_cov3D_backward')

    @staticmethod
    def forward(ctx, *inputs):
        scaling, rotation = inputs
        cov3D = _compute_cov3D._forward(rotation, scaling)
        ctx.save_for_backward(rotation, scaling)
        return cov3D

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        rotation, scaling = ctx.saved_tensors
        grad_rotation, grad_scaling = _compute_cov3D._backward(rotation, scaling, grad_outputs[0])
        return grad_scaling, grad_rotation


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r  # / norm[:, None] ## when apply norm, the grad of r is very small
    R = torch.zeros((q.size(0), 3, 3), device=q.device)
    x, y, z, r = q.unbind(-1)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


@try_use_C_extension(_compute_cov3D.apply, "gs_compute_cov3D_forward", "gs_compute_cov3D_backward")
def compute_cov3D(scaling: Tensor, rotation: Tensor) -> Tensor:
    L = build_scaling_rotation(scaling, rotation)
    L = L @ L.transpose(1, 2)
    symm = torch.stack([L[:, 0, 0], L[:, 0, 1], L[:, 0, 2], L[:, 1, 1], L[:, 1, 2], L[:, 2, 2]], dim=-1)
    return symm


class _compute_cov2D(torch.autograd.Function):
    _forward = get_C_function('gs_compute_cov2D_forward')
    _backward = get_C_function('gs_compute_cov2D_backward')

    @staticmethod
    def forward(ctx, *inputs, **kwargs):
        means, cov3D, Tw2v, fx, fy, tx, ty = inputs
        ctx.focal_fov = (fx, fy, tx, ty)
        means, cov3D, Tw2v, = means.contiguous(), cov3D.contiguous(), Tw2v.contiguous()
        cov2D = _compute_cov2D._forward(cov3D, means, Tw2v, fx, fy, tx, ty)
        ctx.save_for_backward(cov3D, means, Tw2v)
        return cov2D

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        cov3D, means, Tw2v = ctx.saved_tensors
        grad_Tw2v = torch.zeros_like(Tw2v) if ctx.needs_input_grad[2] else None
        grad_cov3D, grad_mean = _compute_cov2D._backward(cov3D, means, Tw2v, *ctx.focal_fov, grad_outputs[0], grad_Tw2v)
        return grad_mean, grad_cov3D, grad_Tw2v, None, None, None, None


@try_use_C_extension(_compute_cov2D.apply, "gs_compute_cov2D_forward", "gs_compute_cov2D_backward")
def compute_cov2D(
        means: Tensor, cov3D: Tensor, Tw2v: Tensor,
        focal_x: float, focal_y: float, tan_fovx: float, tan_fovy: float
):
    def show_grad(name):
        def _show(grad):
            print(f"{name}: {grad[49]=}")

        return _show

    # means.register_hook(show_grad('means'))
    P = means.shape[0]
    R = Tw2v[..., :3, :3]
    # print(f'{R=}')
    p = torch.einsum('pi,ji->pj', means, R) + Tw2v[..., :3, 3]
    # p.register_hook(show_grad('t'))

    x, y, z = p.unbind(-1)
    limit_x = z.detach() * 1.3 * tan_fovx
    limit_y = z.detach() * 1.3 * tan_fovy
    x = x.clamp(-limit_x, limit_x)
    y = y.clamp(-limit_y, limit_y)
    v0 = torch.zeros_like(z)

    J = torch.stack(
        [
            focal_x / z, v0, -(focal_x * x) / (z * z),
            v0, focal_y / z, -(focal_y * y) / (z * z),
        ], dim=-1
    ).view(P, 2, 3)
    # J.register_hook(show_grad('J'))
    T = J @ R
    # T.register_hook(show_grad('T'))
    V = torch.stack(
        [
            cov3D[:, 0], cov3D[:, 1], cov3D[:, 2],
            cov3D[:, 1], cov3D[:, 3], cov3D[:, 4],
            cov3D[:, 2], cov3D[:, 4], cov3D[:, 5],
        ], dim=-1
    ).view(-1, 3, 3)
    V_ = T @ V @ T.transpose(-1, -2)
    cov2D = torch.stack([V_[:, 0, 0] + 0.3, V_[:, 0, 1], V_[:, 1, 1] + 0.3], dim=-1)
    return cov2D


class _preprocess(torch.autograd.Function):
    _forward = get_C_function('gs_preprocess_forward')
    _backward = get_C_function('gs_preprocess_backward')

    @staticmethod
    def forward(ctx: Any, *args):
        (
            means3D, scales, rotations, opacities, sh_or_colors, sh_rest,  # Gaussians
            Tw2v, Tv2c, campos, tanFoV,  # camera information
            W, H, sh_degree, is_opengl,  # const parameters
            cov3D, cov2D, means2D, culling
        ) = args
        ctx._args = (W, H, sh_degree, is_opengl, cov3D is not None)
        means2D, depths, colors, radii, tiles_touched, cov3D, conic_opacity = _preprocess._forward(
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, sh_or_colors, sh_rest,
            Tw2v, Tv2c, campos, tanFoV,
            cov3D, cov2D, means2D, culling
        )
        ctx.save_for_backward(
            means3D, scales, rotations, opacities, sh_or_colors, sh_rest,  # Gaussians
            Tw2v, Tv2c, campos, tanFoV,  # camera information
            cov3D, cov2D, radii, colors
        )
        return means2D, depths, colors, radii, tiles_touched, conic_opacity

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        (
            means3D, scales, rotations, opacities, sh_or_colors, sh_rest,
            Tw2v, Tv2c, campos, tanFoV,
            cov3D, cov2D, radii, colors
        ) = ctx.saved_tensors
        W, H, sh_degree, is_opengl, has_cov3D = ctx._args

        grad_means3D = torch.zeros_like(means3D)
        grad_scales = None
        grad_rotations = None
        if cov2D is not None:
            grad_cov = torch.zeros_like(cov2D) if ctx.needs_input_grad[-3] else None
        elif has_cov3D:
            grad_cov = torch.zeros_like(cov3D) if ctx.needs_input_grad[-4] else None
        else:
            grad_scales = torch.zeros_like(scales) if ctx.needs_input_grad[1] else None
            grad_rotations = torch.zeros_like(rotations) if ctx.needs_input_grad[2] else None
            grad_cov = torch.zeros_like(cov3D)
        grad_opacities = torch.zeros_like(opacities)
        grad_shs = torch.zeros_like(sh_or_colors) if sh_degree >= 0 and ctx.needs_input_grad[4] else None
        grad_sh_rest = torch.zeros_like(sh_rest) if sh_degree >= 0 and ctx.needs_input_grad[5] else None

        grad_Tw2v = torch.zeros_like(Tw2v) if ctx.needs_input_grad[6] else None
        grad_campos = torch.zeros_like(campos) if ctx.needs_input_grad[8] else None
        _preprocess._backward(
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, sh_or_colors, sh_rest,
            Tw2v, Tv2c, campos, tanFoV,
            cov3D, cov2D, radii, colors,
            grad_outputs[0], grad_outputs[1], grad_outputs[2], grad_outputs[5],
            grad_means3D, grad_scales, grad_rotations, grad_opacities, grad_shs, grad_sh_rest, grad_cov,
            grad_Tw2v, grad_campos,
        )
        grad_inputs = [grad_means3D, grad_scales, grad_rotations, grad_opacities, grad_shs, grad_sh_rest] + \
                      [grad_Tw2v, None, grad_campos, None] + [None] * 8
        if sh_degree < 0:
            grad_inputs[4] = grad_outputs[2] if sh_rest >= 0 else None
        elif has_cov3D and ctx.needs_input_grad[-4]:
            grad_inputs[-4] = grad_cov
        if cov2D is not None and ctx.needs_input_grad[-3]:
            grad_inputs[-3] = grad_cov
        grad_inputs[-2] = grad_outputs[0] if ctx.needs_input_grad[-2] else None
        return tuple(grad_inputs)


def preprocess(
        means3D: Tensor, scales: Tensor, rotations: Tensor, opacities: Tensor, sh_or_colors: Tensor,
        sh_rest: Optional[Tensor],  # Gaussians
        Tw2v: Tensor, Tv2c: Tensor, campos: Tensor, tanFoV: Tensor,  # camera information
        W: int, H: int, sh_degree=0, is_opengl=True,  # const parameters
        cov3D: Tensor = None, cov2D: Tensor = None, means2D: Tensor = None, culling: Tensor = None,
        # pre-computed Tensors
):
    """预处理Gaussians
    Args:
        means3D: the postions of Gaussians, shape: [P, 3]
        scales: the scales of Gaussians, shape: [P, 3]
        rotations: the rotations of Gaussians, shape: [P, 4]
        opacities: the opacities of Gaussians, shape: [P, 1]
        sh_or_colors: percompute colors or SH features of Gaussians or SH DC features , shape: [P, 3] or [P, F/1, 3]
        sh_rest: SH rest features of Gaussians, shape: [P, F]
        Tw2v: camera transform maxtrix, world to view space, shape: [4, 4]
        Tv2c: perspective matrix, covert view space to clip space, shape: [4, 4]
        campos: the postion of camera, shape: [3]
        tanFoV: the focal length of camera, shape: [2]
        W: the width of output image
        H: the height of output image
        sh_degree: the used degree of SH
        is_opengl: Is clip space a left-handed coordinate system, like OpenGL?
        cov3D: pre-compute 3D covariance matrix, None or [P, 6]
        cov2D: pre-compute 2D covariance matrix, None or [P, 3]
        means2D: the postions of Gaussians on the image plane, shape: [P, 2]
        culling: non-visible masks
    Returns:
        (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)

        - means2D: the pixel coordinates on the image plane, shape: [P, 2]
        - depth: the depth of each Gaussian to image plane, shape: [P]
        - colors: RGB clors, [P, 3]
        - radii: the radii of project 2D Gaussian, =0 mean not shown on image plane, shape: [P]
        - tiles_touched: how many tiles may touch by each Gaussian, [P]
        - conic_opacity: the inverse cov2D and opacity of each Gaussian, [P, 4]
    """
    from utils.ops_3d.spherical_harmonics import SH_to_RGB

    def show_grad(name):
        def _show(grad):
            print(f"{name}: {grad[6116449]=}")

        return _show

    N, _ = means3D.shape
    size = means3D.new_tensor([W, H])
    p_hom = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    # means3D.register_hook(show_grad('means3D'))
    # p_hom.register_hook(show_grad('p_hom'))
    p_view = torch.einsum('ij,pj->pi', Tw2v, p_hom)  # Tw2v @ points
    # p_view.register_hook(show_grad('p_view'))
    p_clip = torch.einsum('ij,pj->pi', Tv2c, p_view)  # Tv2c @ points
    # p_clip.register_hook(show_grad('p_clip'))
    p_w = 1 / (p_clip[:, 3:4] + 1e-7)
    p_proj = p_clip[:, :3] * p_w

    def scale_grad(grad):
        return 0.5 * size * grad

    sign = means3D.new_tensor([1., -1. if is_opengl else 1.])
    offset = 0 if is_opengl else -1.0
    if means2D is not None:
        means2D.register_hook(scale_grad)
        means2D.data = ((p_proj[:, :2] * sign + 1.0) * size + offset) * 0.5
    else:
        means2D = ((p_proj[:, :2] * sign + 1.0) * size + offset) * 0.5

    if is_opengl:
        depth = p_proj[:, 2]
        mask = (-1 <= depth) & (depth <= 1)
    else:
        depth = p_view[:, 2]
        mask = p_view[:, 2] > 0.2
    if culling is not None:
        mask = mask | ~culling
    # print('\033[34m')
    # print('num mask:', mask.sum().item())
    # print('p_view:', p_view[6116449])
    # print('p_proj:', p_proj[6116449])
    tan_fovx, tan_fovy = tanFoV.tolist()
    focal_x, focal_y = 0.5 * W / tan_fovx, 0.5 * H / tan_fovy
    # print('focal:', focal_x, focal_y)
    if cov2D is None:
        if cov3D is None:
            cov3D = compute_cov3D(scales, rotations)
        cov2D = compute_cov2D(means3D, cov3D, Tw2v, focal_x, focal_y, tan_fovx, tan_fovy)
        # print('cov3D:', cov3D[6116449])
        # print('cov2D:', cov2D[6116449])
    # Invert covariance (EWA algorithm)
    det = cov2D[:, 0] * cov2D[:, 2] - cov2D[:, 1] * cov2D[:, 1]
    mask = mask & (det > 0)
    # compute colors
    if sh_degree >= 0:
        if sh_rest is not None:
            sh_or_colors = torch.cat([sh_or_colors, sh_rest], dim=1)
        colors = SH_to_RGB(sh_or_colors, means3D, campos, sh_degree)
        colors = colors.clamp_min(0)
    else:
        colors = sh_or_colors

    # Compute extent in screen space
    with torch.no_grad():
        mid = 0.5 * (cov2D[:, 0] + cov2D[:, 2])
        la1 = mid + (mid * mid - det).clamp_min(0.1).sqrt()
        la2 = mid - (mid * mid - det).clamp_min(0.1).sqrt()
        radii = torch.ceil(3. * torch.maximum(la1, la2).sqrt()).to(torch.int32)

        grid_x = (W + _BLOCK_X - 1) // _BLOCK_X
        grid_y = (H + _BLOCK_Y - 1) // _BLOCK_Y
        rect = torch.stack(
            [
                ((means2D[:, 0] - radii).int() // _BLOCK_X).clamp(0, grid_x),
                ((means2D[:, 1] - radii).int() // _BLOCK_Y).clamp(0, grid_y),
                ((means2D[:, 0] + radii + _BLOCK_X - 1).int() // _BLOCK_X).clamp(0, grid_x),
                ((means2D[:, 1] + radii + _BLOCK_Y - 1).int() // _BLOCK_Y).clamp(0, grid_y),
            ], dim=-1
        )
        tiles_touched = ((rect[:, 2] - rect[:, 0]) * (rect[:, 3] - rect[:, 1]))
        mask = mask & (tiles_touched > 0)
    inv_dev = 1. / det.clamp_min(1e-10)
    conic_opacity = torch.stack(
        [cov2D[:, 2] * inv_dev, -cov2D[:, 1] * inv_dev, cov2D[:, 0] * inv_dev, opacities[:, 0]], dim=-1
    )
    # print('means2D:', means2D[6116449])
    # print('radii:', radii[6116449])
    # print('conic_opacity:', conic_opacity[6116449])
    # print('mask:', mask[6116449])
    # print('rect:', rect[6116449])
    mask = mask.detach().float()
    radii, tiles_touched = radii * mask.to(radii), tiles_touched * mask.to(tiles_touched)
    colors = colors * mask[:, None].detach().float()
    # print('\033[0m')
    return means2D * mask[:, None], depth * mask, colors, radii, tiles_touched, conic_opacity * mask[:, None]


@torch.no_grad()
def prepare_v0(W: int, H: int, means2D: Tensor, depths: Tensor, radii: Tensor, tiles_touched: Tensor, debug: bool):
    """确定每个tile需要哪些Gaussian, 为渲染做准备"""
    N = tiles_touched.sum().item()
    ## For each instance to be rendered, produce adequate [ tile | depth ] key and sort
    grid_x = (W + _BLOCK_X - 1) // _BLOCK_X
    grid_y = (H + _BLOCK_Y - 1) // _BLOCK_Y
    rect = torch.stack(
        [
            ((means2D[:, 0] - radii).int() // _BLOCK_X).clamp(0, grid_x),
            ((means2D[:, 1] - radii).int() // _BLOCK_Y).clamp(0, grid_y),
            ((means2D[:, 0] + radii + _BLOCK_X - 1).int() // _BLOCK_X).clamp(0, grid_x),
            ((means2D[:, 1] + radii + _BLOCK_Y - 1).int() // _BLOCK_Y).clamp(0, grid_y),
        ], dim=-1
    )
    sorted_key = torch.zeros((N,), dtype=torch.double, device=means2D.device)
    sorted_idx = torch.zeros((N,), dtype=torch.int, device=means2D.device)
    k = 0
    min_depth, max_dpeth = depths.aminmax()
    depths = (depths - min_depth) / (max_dpeth - min_depth) * 0.5 + 0.25  # [0.25, 0.75]
    for i in range(rect.shape[0]):
        min_x, min_y, max_x, max_y = rect[i].tolist()
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                sorted_key[k] = (y * grid_x + x) + depths[i].item()
                sorted_idx[k] = i
                k += 1
    order = torch.argsort(sorted_key)
    num_gs_on_pixels = torch.bincount(sorted_key.floor().long(), minlength=grid_x * grid_y)
    temp = num_gs_on_pixels.cumsum(dim=0)
    tile_range = torch.stack([temp - num_gs_on_pixels, temp], dim=-1)
    tile_range = tile_range * (tile_range[:, 0:1] != tile_range[:, 1:2])
    return tile_range, sorted_idx[order]


@try_use_C_extension
def GS_prepare(W: int, H: int, means2D: Tensor, depths: Tensor, radii: Tensor, tiles_touched: Tensor, debug: bool):
    """确定每个tile需要哪些Gaussian, 为渲染做准备"""
    with torch.no_grad():
        grid_x = (W + _BLOCK_X - 1) // _BLOCK_X
        grid_y = (H + _BLOCK_Y - 1) // _BLOCK_Y
        rect = torch.stack(
            [
                ((means2D[:, 0] - radii).int() // _BLOCK_X).clamp(0, grid_x),
                ((means2D[:, 1] - radii).int() // _BLOCK_Y).clamp(0, grid_y),
                ((means2D[:, 0] + radii + _BLOCK_X - 1).int() // _BLOCK_X).clamp(0, grid_x),
                ((means2D[:, 1] + radii + _BLOCK_Y - 1).int() // _BLOCK_Y).clamp(0, grid_y),
            ], dim=-1
        )
        depths, order = torch.sort(depths)  # in OpenGL, depth < 0
        rect = rect[order]
        tile_range = []
        point_list = []
        num_total = 0
        for y in range(grid_y):
            for x in range(grid_x):
                mask = (rect[:, 0] <= x) & (rect[:, 1] <= y) & (rect[:, 2] > x) & (rect[:, 3] > y)
                num_in_this_tile = mask.sum()
                if num_in_this_tile == 0:
                    tile_range.append((0, 0))
                else:
                    tile_range.append((num_total, num_total + num_in_this_tile))
                    num_total = num_total + num_in_this_tile
                point_list.append(order[mask])
        tile_range = torch.tensor(tile_range, dtype=torch.long, device=depths.device)
        point_list = torch.cat(point_list, dim=0).int()
    return tile_range, point_list


class _GS_render(torch.autograd.Function):
    _forward = get_C_function('gaussian_rasterize_forward')
    _backward = get_C_function('gaussian_rasterize_backward')

    @staticmethod
    def forward(ctx, *inputs):
        (W, H, mean2D, colors, conic_opacity, tile_range, point_list, extras,
         accum_max_count, accum_weights_p, accum_weights_count) = inputs
        images, out_opacity, out_extras, n_contrib = _GS_render._forward(
            W, H, mean2D, conic_opacity, colors, extras, point_list, tile_range,
            accum_max_count, accum_weights_p, accum_weights_count, None
        )
        ctx.save_for_backward(mean2D, colors, conic_opacity, extras, tile_range, point_list, out_opacity, n_contrib)
        return images, out_opacity, out_extras, n_contrib

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        grad_images, grad_alpha, grad_out_extra, _ = grad_outputs
        mean2D, colors, conic_opacity, extras, tile_range, point_list, out_opacity, n_contrib = ctx.saved_tensors
        grad_colors, grad_extra, grad_means2D, grad_dconic_opacity = _GS_render._backward(
            mean2D, conic_opacity, colors, extras, out_opacity,
            grad_images, grad_alpha, grad_out_extra,
            tile_range, point_list, n_contrib
        )
        return None, None, grad_means2D, grad_colors, grad_dconic_opacity, None, None, grad_extra, None, None, None


class _GS_render_fast(torch.autograd.Function):
    _forward = get_C_function('gaussian_rasterize_forward')
    _backward = get_C_function('gaussian_rasterize_fast_backward')

    @staticmethod
    def forward(ctx, *inputs):
        (W, H, mean2D, colors, conic_opacity, tile_range, point_list, extras,
         accum_max_count, accum_weights_p, accum_weights_count) = inputs
        per_tile_bucket_offset = (tile_range[..., 1] - tile_range[..., 0] + 31) // 32
        per_tile_bucket_offset = torch.cumsum(per_tile_bucket_offset, dim=-1).contiguous()

        images, out_opacity, out_extras, n_contrib, max_contrib, bucket_to_tile, sampled_T, sampled_ar = _GS_render_fast._forward(
            W, H, mean2D, conic_opacity, colors, extras, point_list, tile_range,
            accum_max_count, accum_weights_p, accum_weights_count, per_tile_bucket_offset
        )
        ctx.save_for_backward(mean2D, colors, conic_opacity, extras, tile_range, point_list,
                              images, out_opacity, out_extras, n_contrib,
                              per_tile_bucket_offset, max_contrib, bucket_to_tile, sampled_T, sampled_ar)
        return images, out_opacity, out_extras, n_contrib

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        grad_images, grad_alpha, grad_out_extra, _ = grad_outputs
        (mean2D, colors, conic_opacity, extras, tile_range, point_list,
         images, out_opacity, out_extras, n_contrib,
         per_tile_bucket_offset, max_contrib, bucket_to_tile, sampled_T, sampled_ar) = ctx.saved_tensors
        grad_colors, grad_extra, grad_means2D, grad_dconic_opacity = _GS_render_fast._backward(
            mean2D, conic_opacity, colors, extras, images, out_extras, out_opacity,
            grad_images, grad_alpha, grad_out_extra,
            tile_range, point_list, n_contrib,
            max_contrib, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar
        )
        return None, None, grad_means2D, grad_colors, grad_dconic_opacity, None, None, grad_extra, None, None, None


@try_use_C_extension(_GS_render.apply, 'gaussian_rasterize_forward', 'gaussian_rasterize_backward')
def GS_render(
        W: int, H: int, mean2D: Tensor, colors: Tensor, conic_opacity: Tensor, tile_range: Tensor,
        point_list: Tensor, extra: Tensor = None,
        accum_max_count=None, accum_weights_p=None, accum_weights_count=None,
):
    grid_x = (W + _BLOCK_X - 1) // _BLOCK_X
    grid_y = (H + _BLOCK_Y - 1) // _BLOCK_Y
    images = mean2D.new_zeros(colors.shape[-1], H, W)
    n_contrib = conic_opacity.new_zeros(H, W, dtype=torch.int)
    opacities = conic_opacity.new_zeros(H, W)
    out_extra = None if extra is None else extra.new_zeros(extra.shape[-1], H, W)

    pix_id = 57080
    debug_x, debug_y = pix_id % W, pix_id // W
    debug_tx, debug_ty = debug_x // _BLOCK_X, debug_y // _BLOCK_Y
    debug_i = (debug_y % _BLOCK_Y) * _BLOCK_X + debug_x % _BLOCK_X
    for y in range(grid_y):
        sy = y * _BLOCK_Y
        ey = min(H, sy + _BLOCK_Y)
        H_ = ey - sy
        for x in range(grid_x):
            s, e = tile_range[y * grid_x + x].tolist()
            sx = x * _BLOCK_X
            ex = min(W, sx + _BLOCK_X)
            W_ = ex - sx
            if s == e:
                continue
            tile_xy = torch.stack(
                torch.meshgrid(
                    torch.arange(W_, device=mean2D.device, dtype=mean2D.dtype),
                    torch.arange(H_, device=mean2D.device, dtype=mean2D.dtype),
                    indexing='xy'
                ), dim=-1
            )
            xy = tile_xy.reshape(-1, 2) + tile_xy.new_tensor([x * _BLOCK_X, y * _BLOCK_Y])  # [T_W*T_H, 2]
            index = point_list[s:e]
            gs_xy = mean2D[index]  # [P, 2]
            gs_cov_inv_opactiy = conic_opacity[index]  # [P, 4]
            gs_color = colors[index]
            gs_extra = extra[index] if extra is not None else None
            d_xy = xy[:, None, :] - gs_xy[None, :, :]

            power = (-0.5 * (gs_cov_inv_opactiy[:, 0] * d_xy[:, :, 0] * d_xy[:, :, 0]
                             + gs_cov_inv_opactiy[:, 2] * d_xy[:, :, 1] * d_xy[:, :, 1])
                     - gs_cov_inv_opactiy[:, 1] * d_xy[:, :, 0] * d_xy[:, :, 1])
            mask = power <= 0
            alpha = (gs_cov_inv_opactiy[:, 3] * power.exp()) * mask  # [T_W * T_H, P]
            alpha = alpha + (alpha.clamp_max(0.99) - alpha).detach()
            mask = mask & (alpha >= 1. / 255)
            alpha = torch.where(alpha < 1. / 255, torch.zeros_like(alpha), alpha)
            sigma = (1 - alpha).cumprod(dim=1)
            idx_ = torch.arange(W_ * H_, device=index.device, dtype=torch.int)
            with torch.no_grad():
                mask = torch.logical_and(mask, sigma >= 0.0001)
                num_used = (mask.flip(1).cumsum(dim=1) > 0).sum(dim=1)
            # if torch.any(index == 3968):
            #     for i in range(W_ * H_):
            #         print(xy[i, 1] * W + xy[i, 0], num_used[i])
            # if x == debug_tx and y == debug_ty:
            #     print('\033[33m')
            #     print(f"{num_used[debug_i]=}")
            #     for i in range(len(index)):
            #         if mask[debug_i, i]:
            #             print(f"gs={index[i].item():}, xy={gs_xy[i, 0]:.6f}, {gs_xy[i, 1]:.6f}, "
            #                   f"power={power[debug_i, i].item():.6f}, "
            #                   f"alpha={alpha[debug_i, i].item():.6f}, sigma={sigma[debug_i, i]:.6f}")
            #     print('\033[0m')
            #
            #     def _show(mask_):
            #         def show_grad(grad):
            #             print(grad.shape, debug_i, mask_.shape)
            #             print('alpha grad:', grad[debug_i][mask_[debug_i]])
            #
            #         return show_grad
            #
            #     alpha.register_hook(_show(mask))
            n_contrib[sy:ey, sx:ex] = num_used.reshape(H_, W_)
            opacities[sy:ey, sx:ex] = (1 - sigma[idx_, num_used - 1]).reshape(H_, W_)  # * mask.any(dim=1)
            sigma = torch.cat([torch.ones_like(sigma[:, :1]), sigma[:, :-1]], dim=-1) * alpha * mask
            images[:, sy:ey, sx:ex] = torch.einsum('pi,ij->jp', sigma, gs_color).reshape(-1, H_, W_)
            if extra is not None:
                out_extra[:, sy:ey, sx:sy] = torch.einsum('pi,ij->jp', sigma, gs_extra).reshape(-1, H_, W_)
    return images, opacities, out_extra, n_contrib


class _render_extra(torch.autograd.Function):
    _forward = get_C_function('gaussian_rasterize_extra_forward')
    _backward = get_C_function('gaussian_rasterize_extra_backward')

    @staticmethod
    def forward(ctx, *args):
        W, H, extras, means2D, conic_opacity, ranges, point_list, n_contrib, out_opactiy, = args
        extras = extras.contiguous()
        out_pixels = _render_extra._forward(W, H, extras, means2D, conic_opacity, ranges, point_list, n_contrib)
        ctx.save_for_backward(out_opactiy, extras, means2D, conic_opacity, ranges, point_list, n_contrib)
        return out_pixels

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outpus):
        out_opactiy, extras, means2D, conic_opacity, ranges, point_list, n_contrib = ctx.saved_tensors
        H, W = out_opactiy.shape
        grad_extra = torch.zeros_like(extras) if ctx.needs_input_grad[2] else None
        grad_means2D = torch.zeros_like(means2D) if ctx.needs_input_grad[3] else None
        grad_conic = torch.zeros_like(conic_opacity) if ctx.needs_input_grad[4] else None
        _render_extra._backward(
            W, H, extras, out_opactiy, grad_outpus[0], means2D, conic_opacity, ranges, point_list, n_contrib,
            grad_extra, grad_means2D, grad_conic
        )
        return None, None, grad_extra, grad_means2D, grad_conic, None, None, None, None


def render_extra(buffer: RasterizeBuffer, *args: Tensor, detach=False):
    outputs = []
    if detach:
        means2D, conic_opacity = buffer.means2D.detach(), buffer.conic_opacity.detach()
    else:
        means2D, conic_opacity = buffer.means2D, buffer.conic_opacity
    for x in args:
        outputs.append(_render_extra.apply(
            buffer.W, buffer.H, x, means2D, conic_opacity, buffer.tile_ranges,
            buffer.point_list, buffer.n_contribed, buffer.opacity))
    return outputs[0] if len(args) == 1 else outputs


@torch.no_grad()
def _show_value_at_max_error(v0, v1, before=0, after=10, index=None):
    if v0.ndim == 1:
        v0, v1 = v0[:, None], v1[:, None]
    else:
        v0, v1 = v0.flatten(0, -2), v1.flatten(0, -2)
    index = (v0 - v1).abs().argmax().item() // v0.shape[1] if index is None else index
    print(torch.cat([v0, v1], dim=1)[max(0, index - before):index + after], f"{index=}")


def test_compute_cov3D():
    print()
    from fast_2d_gs._C import get_python_function
    from utils import get_run_speed, get_rel_error
    # from extension import ops_3d
    from pathlib import Path
    N = 10000
    if 0 and Path(__file__).parent.joinpath('../../results/test.data.pth').exists():
        data = torch.load(Path(__file__).parent.joinpath('../../results/test.data.pth'))
        scaling = data['scales'].cuda()
        rotation = data['rotations'].cuda()
    else:
        scaling = torch.randn(N, 3).fill_(1.)
        rotation = torch.randn(N, 4)
        rotation = ops_3d.quaternion.standardize(ops_3d.quaternion.normalize(rotation))
    py_func = get_python_function('compute_cov3D')
    cu_func = _compute_cov3D.apply
    s1 = scaling.cuda().requires_grad_()
    r1 = rotation.cuda().requires_grad_()
    o_cu = cu_func(s1, r1)

    s2 = scaling.cuda().requires_grad_()
    r2 = rotation.cuda().requires_grad_()
    o_py = py_func(s2, r2)

    get_rel_error(o_cu, o_py, 'forward:')
    g = torch.randn_like(o_cu)
    torch.autograd.backward(o_cu, g)
    torch.autograd.backward(o_py, g)
    get_rel_error(s1.grad, s2.grad, 'grad_scaling error:')
    get_rel_error(r1.grad, r2.grad, 'grad_rotation error:')
    # print(r1[0])
    # print(r1.grad[0])
    # print(r2.grad[0])

    get_run_speed((s1, r1), g, py_func, cu_func, num_test=100)


def test_compute_cov2D():
    import math
    import utils
    from utils.test_utils import get_run_speed, get_rel_error
    from fast_2d_gs._C import get_python_function
    from pathlib import Path

    seed = np.random.randint(0, int(1e9))
    print(f'{seed=}')
    torch.manual_seed(seed)
    utils.set_printoptions(6)
    print()
    # torch.set_default_dtype(torch.float64)

    if 0 and Path(__file__).parent.joinpath('../../results/test.data.pth').exists():
        data = torch.load(Path(__file__).parent.joinpath('../../results/test.data.pth'))
        Tw2v = data['Tw2v'].cuda()
        means = data['points'].detach().contiguous().cuda()
        scaling = data['scales'].cuda()
        rotation = data['rotations'].cuda()
        focal_x, focal_y = data['focal']
        tan_fovx, tan_fovy = data['tan_fov']
    else:
        fovx = fovy = math.radians(60)
        tan_fovx, tan_fovy = math.tan(0.5 * fovx), math.tan(0.5 * fovy)
        size = 400
        focal_x = focal_y = ops_3d.fov_to_focal(fovy, size)
        N = 10000
        scaling = torch.rand(N, 3).cuda() * 0.1 + 0.05  # [0.05, 0.15]
        rotation = torch.randn(N, 4).cuda()
        rotation = ops_3d.quaternion.standardize(ops_3d.quaternion.normalize(rotation))
        Tw2v = ops_3d.look_at(torch.randn(3), torch.zeros(3)).cuda()
        # Tw2v = torch.eye(4).cuda()
        means = torch.randn((N, 3)).cuda()
        means = (means * 0.3).clamp(-1, 1)
    cov3D = compute_cov3D(scaling, rotation)

    cu_func = compute_cov2D
    py_func = get_python_function('compute_cov2D')

    cov3D_py = cov3D.clone().requires_grad_()
    means_py = means.clone().requires_grad_()
    Tw2v_py = Tw2v.clone().requires_grad_()
    cov2D_py = py_func(means_py, cov3D_py, Tw2v_py, focal_x, focal_y, tan_fovx, tan_fovy)
    g = torch.randn_like(cov2D_py)
    torch.autograd.backward(cov2D_py, g)

    cov3D_cu = cov3D.clone().requires_grad_()
    means_cu = means.clone().requires_grad_()
    Tw2v_cu = Tw2v.clone().requires_grad_()
    print(f"{means_cu[0]=}", means_cu.dtype)
    cov2D_cu = cu_func(means_cu, cov3D_cu, Tw2v_cu, focal_x, focal_y, tan_fovx, tan_fovy)
    torch.autograd.backward(cov2D_cu, g)

    print(cov2D_py.shape)
    get_rel_error(cov2D_cu, cov2D_py, 'forward error:')
    index = (cov2D_cu - cov2D_py).abs().argmax().item() // 3
    print(f'{index=}, {cov2D_cu[index]=}, {cov2D_py[index]=}')
    if means_py.grad is not None:
        get_rel_error(means_cu.grad, means_py.grad, f"grad points: ")
    # index = (means_cu.grad - means_py.grad).abs().argmax() // 3
    # print(f'{index=}, {means_cu.grad[index]=}, {means_py.grad[index]=}')
    if cov3D_py.grad is not None:
        get_rel_error(cov3D_cu.grad, cov3D_py.grad, f"grad cov3D: ")
    if Tw2v_py.grad is not None:
        get_rel_error(Tw2v_cu.grad, Tw2v_py.grad, f"grad Tw2v: ")
        print(Tw2v_cu.grad, Tw2v_py.grad, sep='\n')

    get_run_speed((means_py, cov3D_py, Tw2v_py, focal_x, focal_y, tan_fovx, tan_fovy), g, py_func, cu_func)

    # cov2D_inv = torch.inverse(torch.stack([cov2D_py[:, 0:2], cov2D_py[:, 1:3]], dim=-2))
    # cov2D_det = cov2D_cu[:, 0] * cov2D_cu[:, 2] - cov2D_cu[:, 1] * cov2D_cu[:, 1]
    # cov2D_inv_ = torch.stack([cov2D_cu[:, 2], -cov2D_cu[:, 1], -cov2D_cu[:, 1], cov2D_cu[:, 0]], dim=-1) / cov2D_det[:, None]
    # cov2D_inv_ = cov2D_inv_.view(-1, 2, 2)
    # get_rel_error(cov2D_inv, cov2D_inv_, 'invser cov2D_cu')
    # print(cov2D_inv_[:10])
    # print(cov2D_inv[:10])


@pytest.mark.parametrize('is_opengl', [True, False])
def test_preprocess(is_opengl):
    import utils
    from utils.test_utils import get_run_speed, get_rel_error
    from pathlib import Path
    utils.set_printoptions(precision=6)
    print()
    seed = np.random.randint(0, int(1e9))
    print(f'{seed=}')
    torch.manual_seed(seed)
    # torch.set_default_dtype(torch.float64)
    P = 100000
    sh_degree = 3
    threshold = 1e-0
    device = torch.device('cuda')
    if 0 and Path(__file__).parent.joinpath('../../results/test.data.pth').exists():
        data = torch.load(Path(__file__).parent.joinpath('../../results/test.data.pth'))
        Tw2v = data['Tw2v'].cuda()
        Tv2c = data['Tv2c'].cuda()
        campos = data['campos'].cuda()
        points = data['points'].detach().contiguous().cuda()
        scales = data['scales'].cuda()
        rotate = data['rotations'].cuda()
        opacit = data['opacity'].cuda()
        shs = data['shs'].cuda()
        focal_x, focal_y = data['focal']
        tan_fovx, tan_fovy = data['tan_fov']
        W, H = data['W'], data['H']
        colors = None
    else:
        points = torch.rand(P, 3, device=device) * 2 - 1  # [-1, 1]
        scales = (torch.rand(P, 3, device=device) + 0.5) * 0.1  # [0.05, 0.15]
        rotate = ops_3d.quaternion.normalize(torch.randn(P, 4, device=device))
        opacit = torch.randn(P, 1, device=device)  # [0, 1]
        colors = None  # torch.randn(P, 3, device=device)  # [0, 1]
        shs = torch.randn(P, (1 + sh_degree) ** 2, 3, device=device)

        if is_opengl:
            Tw2v = ops_3d.opengl.look_at(torch.tensor([0, 0, 4.]), torch.zeros(3)).cuda()
        else:
            Tw2v = ops_3d.opencv.look_at(torch.tensor([0, 0, 4.]), torch.zeros(3)).cuda()
        campos = torch.inverse(Tw2v)[:3, 3]
        print(campos)
        focal_x = focal_y = 500
        W, H = 512, 512
        fov_x, fov_y = ops_3d.focal_to_fov(focal_x, W, H)
        tan_fovx, tan_fovy = np.tan(0.5 * fov_x), np.tan(0.5 * fov_y)
        if is_opengl:
            Tv2c = ops_3d.opengl.perspective(fov_y, size=(W, H)).to(Tw2v)
        else:
            Tv2c = ops_3d.opencv.perspective(fov_y, size=(W, H)).to(Tw2v)
    tanFoV = Tw2v.new_tensor([tan_fovx, tan_fovy])
    p_view = ops_3d.apply(points, Tw2v)
    print('points:', *points.aminmax())
    print('view:', *p_view.aminmax())
    p_proj = ops_3d.apply(points, Tv2c @ Tw2v)
    print('proj:', *p_proj.aminmax())

    cov3D = None  # compute_cov3D(scales, rotate)
    cov2D = None  # compute_cov2D(points, cov3D, Tw2v, focal, focal, tan_fovx, tan_fovy)

    cu_func = _preprocess.apply
    py_func = preprocess

    def _grad():
        args = [points, scales, rotate, opacit, shs, colors, (Tw2v, True), (campos, True), cov3D, cov2D]
        inputs = []
        for v in args:
            if v is None:
                inputs.append(v)
            elif isinstance(v, torch.Tensor):
                inputs.append(v.clone().requires_grad_())
            else:
                inputs.append(v[0].clone().requires_grad_(v[1]))
        return inputs

    points_py, scales_py, rotate_py, opacit_py, shs_py, colors_py, Tw2v_py, campos_py, cov3D_py, cov2D_py = _grad()
    outputs_py = py_func(
        points_py, scales_py, rotate_py, opacit_py, shs_py, None,
        Tw2v_py, Tv2c, campos_py, tanFoV, W, H, sh_degree, is_opengl,
        cov3D_py, cov2D_py, None
    )
    print('outputs_py:', utils.show_shape(outputs_py))
    points_cu, scales_cu, rotate_cu, opacit_cu, shs_cu, colors_cu, Tw2v_cu, campos_cu, cov3D_cu, cov2D_cu = _grad()
    outputs_cu = cu_func(
        points_cu, scales_cu, rotate_cu, opacit_cu, shs_cu, None,
        Tw2v_cu, Tv2c, campos_cu, tanFoV, W, H, sh_degree, is_opengl,
        cov3D_cu, cov2D_cu, None, None
    )
    mask_py = outputs_py[3] > 0
    mask_cu = outputs_cu[3] > 0
    mask_all = torch.logical_and(mask_py, mask_cu)
    print('\033[32mnumber of different masks:', torch.logical_xor(mask_py, mask_cu).sum().item(), "\033[0m")
    print('outputs_cu:', utils.show_shape(outputs_cu))
    names = ['means2D', 'depths', 'colors', 'radii', 'tiles_touched', 'conic_opacity']
    outputs_cu, outputs_py = list(outputs_cu), list(outputs_py)
    for i, name in enumerate(names):
        a, b = outputs_cu[i], outputs_py[i]
        a = a.to(b)
        if a.ndim == 2:
            outputs_cu[i] = a * mask_all[:, None]
            outputs_py[i] = b * mask_all[:, None]
        else:
            outputs_cu[i] = a * mask_all
            outputs_py[i] = b * mask_all
        a, b = outputs_cu[i], outputs_py[i]
        index = (a - b).abs().argmax().item()
        if a.ndim == 2:
            index //= a.shape[1]
        print(torch.cat([a[max(0, index - 1):index + 2], b[max(0, index - 1):index + 2]], dim=-1), f"{index=}")
        get_rel_error(a, b, f'error for {name}{list(a.shape)}')

    grad_means2D = torch.randn_like(outputs_cu[0])
    grad_colors = torch.randn_like(outputs_cu[2])
    grad_conic = torch.randn_like(outputs_cu[-1])
    torch.autograd.backward((outputs_py[0], outputs_py[2], outputs_py[-1]), (grad_means2D, grad_colors, grad_conic))
    torch.autograd.backward((outputs_cu[0], outputs_cu[2], outputs_cu[-1]), (grad_means2D, grad_colors, grad_conic))

    if points_py.grad is not None:
        index = (points_cu.grad - points_py.grad).abs().argmax() // rotate.shape[-1]
        print(f"\033[32mmax error index: {index}\033[0m")
        print(points_cu.grad[index])
        print(points_py.grad[index])
        get_rel_error(points_cu.grad, points_py.grad, 'grad for points', threshold=threshold)
    if scales_py.grad is not None:
        get_rel_error(scales_cu.grad, scales_py.grad, 'grad for scales', threshold=threshold)
    if rotate_py.grad is not None:
        get_rel_error(rotate_cu.grad, rotate_py.grad, 'grad for rotate', threshold=threshold)
    if opacit_py.grad is not None:
        get_rel_error(opacit_cu.grad, opacit_py.grad, 'grad for opacity')
    if colors_py is not None and colors_py.grad is not None:
        get_rel_error(colors_cu.grad, colors_py.grad, 'grad for colors', threshold=threshold)
    if shs_py is not None and shs_py.grad is not None:
        get_rel_error(shs_cu.grad, shs_py.grad, 'grad for sh_feature', threshold=threshold)
    if Tw2v_py.grad is not None:
        print(Tw2v_cu.grad, 'Tw2v_cu')
        print(Tw2v_py.grad, 'Tw2v_py')
        get_rel_error(Tw2v_cu.grad, Tw2v_py.grad, 'grad for Tw2v', threshold=threshold)
    if campos_py.grad is not None:
        print(campos_cu.grad, 'campos_cu')
        print(campos_py.grad, 'campos_py')
        get_rel_error(campos_cu.grad, campos_py.grad, 'grad for campos', threshold=threshold)
    if cov2D_py is not None and cov2D_py.grad is not None:
        get_rel_error(cov2D_cu.grad, cov2D_py.grad, 'grad for cov2D', threshold=threshold)
    if cov3D_py is not None and cov3D_py.grad is not None:
        get_rel_error(cov3D_cu.grad, cov3D_py.grad, 'grad for cov3D', threshold=threshold)
    # get_rel_error(_cu.grad, _py.grad, 'grad for :')

    get_run_speed(
        (
            points_py, scales_py, rotate_py, opacit_py, shs_py, None,
            Tw2v_py, Tv2c, campos, tanFoV, W, H, sh_degree, is_opengl, cov3D, cov2D, None, None
        ), (grad_means2D, None, None, None, None, grad_conic), py_func, cu_func
    )


def test_prepare_and_render():
    from pathlib import Path
    import utils
    from utils.test_utils import get_run_speed, get_rel_error, get_abs_error, show_max_different
    from fast_2d_gs.renderer.gaussian_render_origin import render_gs_official_batch

    utils.set_printoptions(precision=6)
    print()
    seed = 5  # np.random.randint(0, int(1e9))
    print(f'{seed=}')
    torch.manual_seed(seed)
    # torch.set_default_dtype(torch.float64)
    P = 10000
    sh_degree = 3
    threshold = 1e-4
    device = torch.device('cuda')
    if 0 and Path(__file__).parent.joinpath('../../results/test.data.pth').exists():
        data = torch.load(Path(__file__).parent.joinpath('../../results/test.data.pth'))
        Tw2v = data['Tw2v'].cuda()
        Tv2c = data['Tv2c'].cuda()
        campos = data['campos'].cuda()
        points = data['points'].detach().contiguous().cuda()
        scales = data['scales'].cuda()
        rotate = data['rotations'].cuda()
        opacit = data['opacity'].cuda()
        shs = data['shs'].cuda()
        index = torch.randint(0, points.shape[0], (P,), device=device).unique()
        points, scales, rotate, opacit, shs = points[index], scales[index], rotate[index], opacit[index], shs[index]
        tan_fovx, tan_fovy = data['tan_fov']
        fov_x, fov_y = np.arctan(tan_fovx), np.arctan(tan_fovy)
        W, H = data['W'], data['H']
        colors = None
    else:
        points = torch.rand(P, 3, device=device) * 2 - 1  # [-1, 1]
        scales = (torch.rand(P, 3, device=device) + 0.5) * 0.1  # [0.05, 0.15]
        rotate = ops_3d.quaternion.normalize(torch.randn(P, 4, device=device))
        opacit = torch.rand(P, 1, device=device)  # [0, 1]
        colors = None  # torch.randn(P, 3, device=device)  # [0, 1]
        shs = torch.randn(P, (1 + sh_degree) ** 2, 3, device=device)

        Tw2v = ops_3d.opencv.look_at(torch.tensor([0, 0, 4.]), torch.zeros(3)).cuda()
        campos = torch.inverse(Tw2v)[:3, 3]
        focal = 500
        W, H = 511, 127
        fov_x, fov_y = ops_3d.focal_to_fov(focal, W, H)
        tan_fovx, tan_fovy = np.tan(0.5 * fov_x).item(), np.tan(0.5 * fov_y).item()
        Tv2c = ops_3d.opencv.perspective(fov_y, size=(W, H)).to(Tw2v)
    tanFoV = Tw2v.new_tensor([tan_fovx, tan_fovy])
    cov3D = None  # compute_cov3D(scales, rotate)
    cov2D = None  # compute_cov2D(points, cov3D, Tw2v, focal, focal, tan_fovx, tan_fovy)

    mean2D, depths, colors, radii, tiles_touched, conic_opacity = _preprocess.apply(
        points, scales, rotate, opacit, shs, None,
        Tw2v, Tv2c, campos, tanFoV, W, H, sh_degree, False,
        cov3D, cov2D, None, None
    )

    cu_func = GS_prepare
    cu_fun2 = get_C_function('GS_prepare_v2')
    py_func = get_python_function('GS_prepare')
    print('inputs:', utils.show_shape(mean2D, depths, tiles_touched, radii))
    tile_ranges_py, point_list_py = py_func(W, H, mean2D, depths, radii, tiles_touched, False)
    tile_ranges_cu, point_list_cu = cu_func(W, H, mean2D, depths, radii, tiles_touched, False)
    tile_ranges_v2, point_list_v2 = cu_fun2(W, H, mean2D, depths / depths.max(), radii, tiles_touched, False)
    print(
        'outputs:', utils.show_shape(point_list_py, tile_ranges_py), 'vs',
        utils.show_shape(point_list_cu, tile_ranges_cu)
    )
    print(utils.show_shape(point_list_v2, tile_ranges_v2))
    # point_list_cu = point_list_cu.int()
    # tile_ranges_cu = tile_ranges_cu.int()
    # print(torch.cat([tile_ranges_cu, tile_ranges_py], dim=-1).tolist())
    get_rel_error(tile_ranges_cu, tile_ranges_py, 'tile_ranges', threshold)
    get_rel_error(tile_ranges_v2, tile_ranges_py, 'tile_ranges v2')
    # print(torch.stack([point_list_cu, point_list_py], dim=0))
    # index = 0  # (point_list_cu - point_list_py).abs().argmax().item()
    # print(index)
    # print(torch.stack([point_list_cu, point_list_py], dim=0)[:, max(0, index - 10):index + 10])
    # print(depths[point_list_cu[index]].item())
    # print(depths[point_list_py[index]].item())
    get_rel_error(point_list_cu, point_list_py, 'point_list', threshold)
    get_rel_error(point_list_v2, point_list_py, 'point_list v2', threshold)
    get_run_speed((W, H, mean2D, depths, radii, tiles_touched, False), None, cu_func=cu_func, cu_v2=cu_fun2)
    get_run_speed(
        (W, H, mean2D, depths, radii, tiles_touched, False), None,
        py_func=py_func, num_test=10
    )  # py_v0=prepare_v0,
    ####################### render #########################
    cu_func = GS_render
    py_func = get_python_function('GS_render')
    cu_fast = _GS_render_fast.apply
    mean2D_py = mean2D.clone().requires_grad_()
    colors_py = colors.clone().requires_grad_()
    conic_opacity_py = conic_opacity.clone().requires_grad_()
    rgb_py, alpha_py, _, n_contrib_py = py_func(
        W, H, mean2D_py, colors_py, conic_opacity_py, tile_ranges_py, point_list_py, None, None, None, None
    )

    mean2D_cu = mean2D.clone().requires_grad_()
    colors_cu = colors.clone().requires_grad_()
    conic_opacity_cu = conic_opacity.clone().requires_grad_()
    rgb_cu, alpha_cu, _, n_contrib_cu = cu_func(
        W, H, mean2D_cu, colors_cu, conic_opacity_cu, tile_ranges_cu, point_list_cu, None, None, None, None
    )
    print('render outputs:', utils.show_shape(rgb_py, alpha_py), 'vs', utils.show_shape(rgb_cu, alpha_cu))

    mean2D_ft = mean2D.clone().requires_grad_()
    colors_ft = colors.clone().requires_grad_()
    conic_opacity_ft = conic_opacity.clone().requires_grad_()
    rgb_ft, alpha_ft, _, n_contrib_ft = cu_fast(
        W, H, mean2D_ft, colors_ft, conic_opacity_ft, tile_ranges_cu, point_list_cu, None, None, None, None
    )

    get_rel_error(rgb_cu, rgb_py, 'rgb')
    get_rel_error(rgb_ft, rgb_cu, 'rgb fast')
    get_rel_error(alpha_cu, alpha_py, 'alpha')
    get_rel_error(alpha_ft, alpha_cu, 'alpha fast')
    get_rel_error(n_contrib_cu, n_contrib_py, 'n_contrib')
    _show_value_at_max_error(n_contrib_cu.view(-1), n_contrib_py.view(-1), index=None)

    with torch.no_grad():
        out_offical = render_gs_official_batch(
            (W, H), Tw2v, Tv2c, torch.tensor([fov_x, fov_y]).cuda(), points, opacit,
            campos, scales, rotate, None, shs, None, None, torch.zeros(3).cuda(),
            sh_degree
        )
        print(utils.show_shape(out_offical))
        rgb_gt = out_offical['images']  # .permute(1, 2, 0)
        alpha_gt = out_offical['alpha']
        radii_gt = out_offical['radii'].squeeze(0)
    get_rel_error(rgb_cu, rgb_gt, 'rgb offical')
    # _show_value_at_max_error(rgb_cu, rgb_gt)
    if alpha_gt is not None:
        get_rel_error(alpha_cu, alpha_gt.squeeze(0), 'alpha offical')
    get_abs_error(radii, radii_gt, 'radii offical', warn_t=1, error_t=10)

    g_rgb, g_alpha = torch.randn_like(rgb_py), torch.randn_like(alpha_py)
    torch.autograd.backward((rgb_py, alpha_py), (g_rgb, g_alpha))
    torch.autograd.backward((rgb_cu, alpha_cu), (g_rgb, g_alpha))
    torch.autograd.backward((rgb_ft, alpha_ft), (g_rgb, g_alpha))
    get_rel_error(mean2D_cu.grad, mean2D_py.grad, 'grad means2D')
    get_rel_error(mean2D_ft.grad, mean2D_cu.grad, 'grad means2D fast')
    get_rel_error(colors_cu.grad, colors_py.grad, 'grad colors')
    get_rel_error(colors_ft.grad, colors_py.grad, 'grad colors fast')
    get_rel_error(conic_opacity_cu.grad, conic_opacity_py.grad, 'grad conic_opacity')
    get_rel_error(conic_opacity_ft.grad, conic_opacity_py.grad, 'grad conic_opacity fast')
    # _show_value_at_max_error(conic_opacity_ft.grad, conic_opacity_py.grad)

    cu_func2 = _render_extra.apply
    g_rgb, g_alpha = torch.randn_like(rgb_py), torch.zeros_like(alpha_py)
    mean2D_py.grad.zero_()
    rgb_py, alpha_py, _, n_contrib_py = py_func(
        W, H, mean2D_py, colors_py, conic_opacity_py, tile_ranges_py, point_list_py, None, None, None, None
    )
    torch.autograd.backward((rgb_py, alpha_py), (g_rgb, g_alpha))

    mean2D_cu2 = mean2D.clone().requires_grad_()
    colors_cu2 = colors.clone().requires_grad_()
    conic_opacity_cu2 = conic_opacity.clone().requires_grad_()

    rgb_cu2 = cu_func2(
        W, H, colors_cu2, mean2D_cu2, conic_opacity_cu2, tile_ranges_cu, point_list_cu, n_contrib_cu, alpha_cu.detach()
    )

    g_rgb = g_rgb.permute(1, 2, 0)
    torch.autograd.backward((rgb_cu2,), (g_rgb,))
    get_rel_error(rgb_cu2.permute(2, 0, 1), rgb_py, 'rgb_extra')
    get_rel_error(mean2D_cu2.grad, mean2D_py.grad, 'grad means2D_extra')
    get_run_speed(
        (W, H, colors_cu2, mean2D_cu2, conic_opacity_cu2, tile_ranges_cu, point_list_cu, n_contrib_cu,
         alpha_cu.detach()), g_rgb, render_extra=cu_func2, num_test=10
    )


def test():
    import utils
    from utils.test_utils import get_run_speed, get_rel_error
    utils.set_printoptions(precision=6)
    print()
    # torch.manual_seed(1234)
    # torch.set_default_dtype(torch.float64)
    P = 100000
    sh_degree = 3
    threshold = 1e-4
    is_opengl = False
    device = torch.device('cuda')
    points = torch.rand(P, 3, device=device) * 2 - 1  # [-1, 1]
    scales = (torch.rand(P, 3, device=device) + 0.5) * 0.1  # [0.05, 0.15]
    rotate = ops_3d.quaternion.normalize(torch.randn(P, 4, device=device))
    opacit = torch.rand(P, 1, device=device)  # [0, 1]
    colors = None  # torch.randn(P, 3, device=device)  # [0, 1]
    shs = torch.randn(P, (1 + sh_degree) ** 2, 3, device=device)
    if is_opengl:
        Tw2v = ops_3d.opengl.look_at(torch.tensor([0, 0, 4.]), torch.zeros(3)).cuda()
    else:
        Tw2v = ops_3d.opencv.look_at(torch.tensor([0, 0, 4.]), torch.zeros(3)).cuda()
    campos = torch.inverse(Tw2v)[:3, 3]
    focal = 500
    W, H = 512, 512
    fov_x, fov_y = ops_3d.focal_to_fov(focal, W, H)
    tan_fovx, tan_fovy = np.tan(0.5 * fov_x), np.tan(0.5 * fov_y)
    tanFoV = Tw2v.new_tensor([tan_fovx, tan_fovy])
    Tv2c = ops_3d.opencv.perspective(fov_y, size=(W, H)).to(Tw2v)

    means2D = points.new_zeros(P, 2)

    means3D_py = points.clone().requires_grad_(True)
    means2D_py = means2D.clone().requires_grad_(True)
    scaling_py = scales.clone().requires_grad_(True)
    rotations_py = rotate.clone().requires_grad_(True)
    opacities_py = opacit.clone().requires_grad_(True)
    shs_py = shs.clone().requires_grad_(True)
    Tw2v_py = Tw2v.clone().requires_grad_(True)
    campos_py = campos.clone().requires_grad_(True)
    image_py, out_opactiy_py, = RasterizeGaussians(
        means3D_py, means2D_py, shs_py, None, opacities_py, scaling_py, rotations_py,
        None, None, None, (W, H), sh_degree,
        Tw2v_py, Tv2c, campos_py, tanFoV, is_opengl, None, None, None, None
    )[:2]

    means3D_cu = points.clone().requires_grad_(True)
    means2D_cu = means2D.clone().requires_grad_(True)
    scaling_cu = scales.clone().requires_grad_(True)
    rotations_cu = rotate.clone().requires_grad_(True)
    opacities_cu = opacit.clone().requires_grad_(True)
    shs_cu = shs.clone().requires_grad_(True)
    Tw2v_cu = Tw2v.clone().requires_grad_(True)
    campos_cu = campos.clone().requires_grad_(True)
    image_cu, out_opactiy_cu, _, _, _ = _RasterizeGaussians.apply(
        means3D_cu, means2D_cu, shs_cu, None, opacities_cu, scaling_cu, rotations_cu,
        None, None, None, (W, H), sh_degree,
        Tw2v_cu, Tv2c, campos_cu, tanFoV, is_opengl,
        None, None, None, None,
    )
    get_rel_error(image_cu, image_py, 'images')
    get_rel_error(out_opactiy_cu, out_opactiy_py, 'opacity')
    print('outputs:', utils.show_shape(image_cu, out_opactiy_cu), 'vs', utils.show_shape(image_py, out_opactiy_py))

    g_image, g_opacity = torch.randn_like(image_py), torch.randn_like(out_opactiy_py)
    torch.autograd.backward((image_py, out_opactiy_py), (g_image, g_opacity))
    torch.autograd.backward((image_cu, out_opactiy_cu), (g_image, g_opacity))
    if means3D_py.grad is not None:
        # index = (means3D_cu.grad - means3D_py.grad).abs().argmax().item() // 3
        # print(means3D_cu.grad.abs().max())
        # print(index)
        # print(torch.cat([means3D_cu.grad, means3D_py.grad], dim=-1)[max(0, index - 10):index + 10])
        get_rel_error(means3D_cu.grad, means3D_py.grad, 'grad for means3D')
    if means2D_py.grad is not None:
        get_rel_error(means2D_cu.grad, means2D_py.grad, 'grad for means2D')
        print('average grad:', means3D_py.grad.abs().mean(), means2D_py.grad.abs().mean())
        # index = (means2D_cu.grad - means2D_py.grad).abs().argmax().item() // 2
        # print(means2D_cu.grad.abs().max())
        # print(index)
        # print(torch.cat([means2D_cu.grad, means2D_py.grad], dim=-1)[max(0, index - 10):index + 10])
    if scaling_py.grad is not None:
        get_rel_error(scaling_cu.grad, scaling_py.grad, 'grad for scaling')
    if rotations_py.grad is not None:
        get_rel_error(rotations_cu.grad, rotations_py.grad, 'grad for rotations')
    if opacities_py.grad is not None:
        # index = (opacities_cu.grad - opacities_py.grad).abs().argmax().item() // 1
        # print(index)
        # print(torch.cat([opacities_cu.grad, opacities_py.grad], dim=-1)[max(0, index - 10):index + 10])
        get_rel_error(opacities_cu.grad, opacities_py.grad, 'grad for opacities')
    if shs_py.grad is not None:
        get_rel_error(shs_cu.grad, shs_py.grad, 'grad for shs')
    if Tw2v_py.grad is not None:
        get_rel_error(Tw2v_cu.grad, Tw2v_py.grad, 'grad for Tw2v')
    if campos_py.grad is not None:
        get_rel_error(campos_cu.grad, campos_py.grad, 'grad for campos')
    get_run_speed(
        (means3D_cu, means2D_cu, shs_cu, None, opacities_cu, scaling_cu, rotations_cu, None, None, (W, H), sh_degree,
         Tw2v_cu, Tv2c, campos_cu, [tan_fovx, tan_fovy], is_opengl), (g_image, g_opacity, None, None, None),
        merge_v=_RasterizeGaussians.apply, not_merge=RasterizeGaussians
    )


def debug_backward():
    import utils
    dump = torch.load('snapshot_bw.dump', map_location='cuda')
    print(utils.show_shape(dump))
    (
        sh_degree, debug,
        Tw2v, Tv2c, cam_pos, tan_fovx, tan_fovy,
        means3D, scales, rotations, sh, extras, colors, cov3Ds, cov2Ds,  # inputs
        tile_ranges, point_list, n_contrib,
        means2D, conic_opacity, radii, opaticy,  # outputs
        grad_out_color, grad_out_opacity, grad_out_extra,  # grad_outputs
        grad_means2D, grad_conic,  # grad_internal
        grad_means3D, grad_scales, grad_rotations, grad_opacities, grad_shs, grad_extra,  # grad inputs
        grad_Tw2v, grad_campos, grad_colors, grad_cov,  # grad camera; grad pre-computed
    ) = dump
    assert 0 <= point_list.amin() and point_list.amax() < means3D.shape[0]
    assert 0 <= tile_ranges.amin() and tile_ranges.amax() <= point_list.shape[0]
    assert torch.all(tile_ranges[:, 0].le(tile_ranges[:, 1]))
    assert 0 <= n_contrib.amin()
    # dump = list(dump)  # debug
    # print(dump[4])
    # dump[4] = True
    get_C_function("rasterize_gaussians_backward")(*dump)
    print('No Error')


if __name__ == '__main__':
    test_prepare_and_render()
