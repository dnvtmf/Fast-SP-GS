import math
from math import floor
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torch import Tensor

from fast_2d_gs._C import get_C_function, try_use_C_extension
import utils
from utils import ops_3d
from fast_2d_gs.renderer.gaussian_render import GS_prepare

_BLOCK_X = 16
_BLOCK_Y = 16
FilterInvSquare = 2.0

__all__ = ['GS_2D_fast_render']


class _fast_preprocess(torch.autograd.Function):
    _forward_func = get_C_function('gs_2d_fast_preprocess_forward')
    _backward_func = get_C_function('gs_2d_fast_preprocess_backward')

    @staticmethod
    def forward(ctx, *inputs):
        (
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, Tv2s, campos, trans_precomp, means2D, culling
        ) = inputs
        means2D, colors, trans_mat, inverse_m, normal_opacity, depths, radii, tile_range, point_list = _fast_preprocess._forward_func(
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, Tv2s, campos,
            trans_precomp, means2D, culling,
            True  # debug
        )
        ctx.save_for_backward(
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest, Tw2v, Tv2c, Tv2s, campos,
            means2D, colors, trans_mat, radii, normal_opacity
        )
        ctx._info = (W, H, sh_degree, is_opengl)
        return colors, means2D, trans_mat, inverse_m, normal_opacity, depths, radii, tile_range, point_list

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        (means3D, scales, rotations, opacities, shs_or_colors, shs_rest, Tw2v, Tv2c, Tv2s, campos,
         means2D, colors, trans_mat, radii, normal_opacity) = ctx.saved_tensors
        W, H, sh_degree, is_opengl = ctx._info
        grad_colors, grad_means2D, grad_mat, grad_inverse_m, grad_normal_opacity, grad_dpeth = grad_outputs[:6]
        grad_inputs = [
            None, None, None, None,
            torch.zeros_like(means3D), torch.zeros_like(scales), torch.zeros_like(rotations),
            torch.zeros_like(opacities),
            torch.zeros_like(shs_or_colors) if shs_or_colors is not None and ctx.needs_input_grad[8] else None,
            torch.zeros_like(shs_rest) if shs_rest is not None and ctx.needs_input_grad[9] else None,
            None, None, None, None, None, None, None
        ]
        _fast_preprocess._backward_func(
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2s, campos,
            trans_mat, radii, colors, normal_opacity,
            grad_means2D, grad_mat, grad_inverse_m, grad_dpeth, grad_colors, grad_normal_opacity,
            grad_inputs[4], grad_inputs[5], grad_inputs[6], grad_inputs[7], grad_inputs[8], grad_inputs[9],
            grad_inputs[10], grad_inputs[11]
        )
        return tuple(grad_inputs)


def GS_2D_fast_preprocess(
    W: int, H: int, sh_degree: int, is_opengl: bool,
    means3D: Tensor, scales: Tensor, rotations: Tensor, opacities: Tensor, shs_or_colors: Tensor, shs_features_rest,
    Tw2v: Tensor, Tv2c: Tensor, Tv2s: Tensor, campos: Tensor,
    trans_precomp: Tensor = None, means2D: Tensor = None, culling: Tensor = None,
):
    return _fast_preprocess.apply(W, H, sh_degree, is_opengl,
                                  means3D, scales, rotations, opacities, shs_or_colors, shs_features_rest,
                                  Tw2v, Tv2c, Tv2s, campos, trans_precomp, means2D, culling)


def _GS_2D_fast_preprocess_py(
    W: int, H: int, sh_degree: int, is_opengl: bool,
    means3D: Tensor, scales: Tensor, rotations: Tensor, opacities: Tensor,
    shs_or_colors: Tensor, shs_features_rest: Optional[Tensor],
    Tw2v: Tensor, Tv2c: Tensor, Tv2s: Tensor, campos: Tensor,
    trans_precomp: Tensor = None, means2D: Tensor = None, culling: Tensor = None
):
    cutoff = 3.0
    FilterSize = 0.707106  # sqrt(2) / 2
    N, _ = means3D.shape
    size = means3D.new_tensor([W, H])
    p_hom = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    p_view = torch.einsum('ij,pj->pi', Tw2v, p_hom)  # Tw2v @ points

    # p_clip = torch.einsum('ij,pj->pi', Tv2c, p_view)  # Tv2c @ points
    # p_w = 1 / (p_clip[:, 3:4] + 1e-7)
    # p_proj = p_clip[:, :3] * p_w

    debug_i = 2369245

    def show_grad(grad):
        print(grad[0])

    # sign = means3D.new_tensor([1., -1. if is_opengl else 1.])
    # offset = 0  # if is_opengl else -1.0
    # if means2D is not None:
    #     means2D.register_hook(scale_grad)
    #     means2D.data = ((p_proj[:, :2] * sign + 1.0) * size + offset) * 0.5
    # else:
    #     means2D = ((p_proj[:, :2] * sign + 1.0) * size + offset) * 0.5
    # depth = p_proj[:, 2]
    # mask = (-1 <= depth) & (depth <= 1)
    mask = p_view[:, 2] <= 0.2

    trans_mat, normal = fast_compuate_transmat(means3D, scales, rotations, Tw2v, Tv2s)

    def scale_grad(grad):
        return 0.5 * size * grad  # * trans_mat[:, 2, 2, None] * trans_mat_grad[:, :2, 2]

    # if means2D is not None and means2D.requires_grad:
    #     means2D.register_hook(scale_grad)
    cos = -ops_3d.dot(p_view[:, :3], normal)
    mask = mask | cos[:, 0].eq(0)
    normal = torch.sign(cos) * normal

    # compute aabb
    t = trans_mat.new_tensor([cutoff ** 2, cutoff ** 2, -1.])
    d = ops_3d.dot(t, trans_mat[:, 2] ** 2)
    mask = mask | d[:, 0].eq(0)
    f = (1 / d) * t
    p = torch.cat([
        ops_3d.dot(f, trans_mat[:, 0] * trans_mat[:, 2]),
        ops_3d.dot(f, trans_mat[:, 1] * trans_mat[:, 2])
    ], dim=-1)
    # print('p:', p[debug_i], p.shape)
    h = p * p - torch.cat([
        ops_3d.dot(f, trans_mat[:, 0] * trans_mat[:, 0]),
        ops_3d.dot(f, trans_mat[:, 1] * trans_mat[:, 1])
    ], dim=-1)

    # print((p * p)[debug_i])
    # print((trans_mat[:, 0] * trans_mat[:, 0])[debug_i])
    # print((trans_mat[:, 1] * trans_mat[:, 1])[debug_i])
    # print('h:', h[debug_i], h.shape)
    h = h.clamp_min(1e-4).sqrt()
    # print('h:', h[debug_i])
    radius = torch.ceil(h.amax(dim=-1).clamp_min(cutoff * FilterSize))
    # print('radius:', radius[debug_i], radius.shape)

    grid_x = (W + _BLOCK_X - 1) // _BLOCK_X
    grid_y = (H + _BLOCK_Y - 1) // _BLOCK_Y
    rect = torch.stack(
        [
            ((p[:, 0] - radius).int() // _BLOCK_X).clamp(0, grid_x),
            ((p[:, 1] - radius).int() // _BLOCK_Y).clamp(0, grid_y),
            ((p[:, 0] + radius + _BLOCK_X - 1).int() // _BLOCK_X).clamp(0, grid_x),
            ((p[:, 1] + radius + _BLOCK_Y - 1).int() // _BLOCK_Y).clamp(0, grid_y),
        ], dim=-1
    )
    # print('rect:', rect[debug_i], rect.shape)
    tiles_touched = ((rect[:, 2] - rect[:, 0]) * (rect[:, 3] - rect[:, 1]))
    # print('mask:', mask.shape, tiles_touched.shape, (tiles_touched > 0).sum())
    mask = mask | (tiles_touched <= 0)

    if sh_degree >= 0:
        if shs_features_rest is not None:
            shs_or_colors = torch.cat([shs_or_colors, shs_features_rest], dim=1)
        colors = ops_3d.SH_to_RGB(shs_or_colors, means3D, campos, sh_degree, True)
    else:
        colors = shs_or_colors

    depths = p_view[..., 2]
    radii = radius.int()
    normal_opacity = torch.cat([normal, opacities], dim=-1)

    mask = ~mask
    depths = depths * mask
    radii = radii * mask
    # means2D = means2D * mask[:, None]
    normal_opacity = normal_opacity * mask[:, None]
    tiles_touched = tiles_touched * mask
    colors = colors * mask[:, None]
    trans_mat = trans_mat * mask[:, None, None]

    if means2D is not None:
        means2D.data = p * mask[:, None]
    else:
        means2D = p * mask[:, None]

    M = trans_mat
    inverse_m = torch.zeros_like(M)
    inverse_m[..., 0, 0] = (M[..., 1, 1] * M[..., 2, 2] - M[..., 2, 1] * M[..., 1, 2])
    inverse_m[..., 0, 1] = (M[..., 2, 1] * M[..., 0, 2] - M[..., 0, 1] * M[..., 2, 2])
    inverse_m[..., 0, 2] = (M[..., 0, 1] * M[..., 1, 2] - M[..., 1, 1] * M[..., 0, 2])
    inverse_m[..., 1, 0] = (M[..., 2, 0] * M[..., 1, 2] - M[..., 1, 0] * M[..., 2, 2])
    inverse_m[..., 1, 1] = (M[..., 0, 0] * M[..., 2, 2] - M[..., 2, 0] * M[..., 0, 2])
    inverse_m[..., 1, 2] = (M[..., 1, 0] * M[..., 0, 2] - M[..., 0, 0] * M[..., 1, 2])
    inverse_m[..., 2, 0] = (M[..., 1, 0] * M[..., 2, 1] - M[..., 2, 0] * M[..., 1, 1])
    inverse_m[..., 2, 1] = (M[..., 2, 0] * M[..., 0, 1] - M[..., 0, 0] * M[..., 2, 1])
    inverse_m[..., 2, 2] = (M[..., 0, 0] * M[..., 1, 1] - M[..., 1, 0] * M[..., 0, 1])
    points_list, tile_range = _prepare_py(radii, depths, means2D, inverse_m, opacities, W, H)
    return means2D, colors, trans_mat, inverse_m, normal_opacity, depths, radii, tile_range, points_list


def _gs_tile_intersect(cx, cy, Wxx, Wyy, Wxy, Wx_, Wy_, W__, W, H, plt_vis=False):
    assert _BLOCK_X == _BLOCK_Y
    grid_x, grid_y = (W + _BLOCK_X - 1) // _BLOCK_X, (H + _BLOCK_Y - 1) // _BLOCK_Y
    results = []
    march_x = abs(Wyy) > abs(Wxx)
    SIZE = _BLOCK_Y
    if march_x:
        cx, cy = cy, cx
        Wxx, Wyy, Wx_, Wy_ = Wyy, Wxx, Wy_, Wx_
        SIZE = _BLOCK_X
        grid_x, grid_y = grid_y, grid_x
    # if Wxx < 0:
    #     Wxx, Wyy, Wxy, Wx_, Wy_, W__ = -Wxx, -Wyy, -Wxy, -Wx_, -Wy_, -W__
    judge = lambda _x, _y: Wxx * _x * _x + Wyy * _y * _y + Wxy * _x * _y + Wx_ * _x + Wy_ * _y + W__ <= 0
    ## up
    y0 = floor(cy / SIZE) * SIZE
    x0 = -(Wxy * y0 + Wx_) / (2 * Wxx)
    print(f'{x0=}, {y0=}; {cx=}, {cy=}', judge(x0, y0), judge(cx, cy))
    print(Wxx * x0 * x0 + Wyy * y0 * y0 + Wxy * x0 * y0 + Wx_ * x0 + Wy_ * y0 + W__)
    dx, dy = -Wxy / (2 * Wxx) * SIZE, SIZE
    l = floor(x0 / SIZE) * SIZE
    r = l + SIZE
    while judge(l, y0):
        l -= SIZE
    while judge(r, y0):
        r += SIZE
    if plt_vis:
        plt.scatter(y0 if march_x else x0, x0 if march_x else y0, c='cyan', marker='+')
        plt.scatter((y0, y0) if march_x else (l, r), (l, r) if march_x else (y0, y0), c='b', marker='x')
        print('W:', Wxx, Wyy, Wxy, Wx_, Wy_, W__)
        print(x0 / SIZE, y0 / SIZE, l / SIZE, r / SIZE)
    while True:
        x0, y0 = x0 + dx, y0 + dy
        l2 = min(l, floor(x0 / SIZE) * SIZE)
        while judge(l2, y0):
            l2 -= SIZE
        while l2 + SIZE < x0 and not judge(l2 + SIZE, y0):
            l2 += SIZE
        r2 = max(r, math.ceil(x0 / SIZE) * SIZE)
        while judge(r2, y0):
            r2 += SIZE
        while x0 < r2 - SIZE and not judge(r2 - SIZE, y0):
            r2 -= SIZE
        y = int(y0 // SIZE) - 1
        for x in range(int(min(l, l2)) // SIZE, int(max(r, r2)) // SIZE):
            if 0 <= x < grid_x and 0 <= y < grid_y:
                results.append((y, x) if march_x else (x, y))
        l, r = l2, r2
        if plt_vis:
            plt.scatter(y0 if march_x else x0, x0 if march_x else y0, c='cyan', marker='+')
            plt.scatter((y0, y0) if march_x else (l, r), (l, r) if march_x else (y0, y0), c='b', marker='x')
            print(x0 / SIZE, y0 / SIZE, l / SIZE, r / SIZE)
        if not judge(x0, y0) or y0 >= grid_y * SIZE:
            break
    ## down
    y0 = floor(cy / SIZE) * SIZE
    x0 = -(Wxy * y0 + Wx_) / (2 * Wxx)
    l = floor(x0 / SIZE) * SIZE
    r = l + SIZE
    while judge(l, y0):
        l -= SIZE
    while judge(r, y0):
        r += SIZE
    # plt.scatter(y0 if march_x else x0, x0 if march_x else y0, c='cyan', marker='+')
    # plt.scatter((y0, y0) if march_x else (l, r), (l, r) if march_x else (y0, y0), c='b', marker='x')
    while True:
        x0, y0 = x0 - dx, y0 - dy
        l2 = min(l, floor(x0 / SIZE) * SIZE)
        while judge(l2, y0):
            l2 -= SIZE
        while l2 + SIZE < x0 and not judge(l2 + SIZE, y0):
            l2 += SIZE
        r2 = max(r, math.ceil(x0 / SIZE) * SIZE)
        while judge(r2, y0):
            r2 += SIZE
        while x0 < r2 - SIZE and not judge(r2 - SIZE, y0):
            r2 -= SIZE
        y = int(y0 // SIZE)
        for x in range(int(min(l, l2)) // SIZE, int(max(r, r2)) // SIZE):
            if 0 <= x < grid_x and 0 <= y < grid_y:
                results.append((y, x) if march_x else (x, y))
        l, r = l2, r2
        if plt_vis:
            plt.scatter(y0 if march_x else x0, x0 if march_x else y0, c='cyan', marker='+')
            plt.scatter((y0, y0) if march_x else (l, r), (l, r) if march_x else (y0, y0), c='b', marker='x')
            print(x0 / SIZE, y0 / SIZE, l / SIZE, r / SIZE)
        if not judge(x0, y0) or y0 <= 0:
            break
    return results


def _prepare_py(radii: Tensor, depths: Tensor, means2D: Tensor, M: Tensor, opacity: Tensor, W, H):
    grid = ((W + _BLOCK_X - 1) // _BLOCK_X, (H + _BLOCK_Y - 1) // _BLOCK_Y)
    d = -2 * torch.log(255. * opacity)
    Wxx = (M[..., 0, 0] * M[..., 0, 0] + M[..., 1, 0] * M[..., 1, 0] + d * M[..., 2, 0] * M[..., 2, 0])
    Wyy = (M[..., 0, 1] * M[..., 0, 1] + M[..., 1, 1] * M[..., 1, 1] + d * M[..., 2, 1] * M[..., 2, 1])
    Wxy = 2 * (M[..., 0, 0] * M[..., 0, 1] + M[..., 1, 0] * M[..., 1, 1] + d * M[..., 2, 0] * M[..., 2, 1])
    Wx = 2 * (M[..., 0, 0] * M[..., 0, 2] + M[..., 1, 0] * M[..., 1, 2] + d * M[..., 2, 0] * M[..., 2, 2])
    Wy = 2 * (M[..., 0, 1] * M[..., 0, 2] + M[..., 1, 1] * M[..., 1, 2] + d * M[..., 2, 1] * M[..., 2, 2])
    W_ = M[..., 0, 2] * M[..., 0, 2] + M[..., 1, 2] * M[..., 1, 2] + M[..., 2, 2] * M[..., 2, 2] * d
    tile_value = []
    gs_indices = []
    mi, mx = depths.aminmax()
    print(mi, mx)
    depths = (depths - mi) / (mx - mi).clamp_min(1e-8) * 0.5 + 0.25  # [0.25, 0.75]
    for i in range(len(radii)):
        if radii[i] <= 0:
            continue
        points = _gs_tile_intersect(means2D[i, 0].item(), means2D[i, 1].item(), Wxx[i].item(), Wyy[i].item(),
                                    Wxy[i].item(), Wx[i].item(), Wy[i].item(), W_[i].item(), W, H)
        tile_idx = [y * grid[0] + depths[i].item() + x for x, y in points]
        tile_value.extend(tile_idx)
        gs_indices.extend([i] * len(tile_idx))
    gs_indices = torch.tensor(gs_indices, device=means2D.device, dtype=torch.int32)
    tile_value = torch.tensor(tile_value, device=means2D.device)
    value, index = tile_value.sort()
    gs_indices = gs_indices[index]
    value = torch.floor(value).int()
    last = 0
    last_i = 0
    tile_range = []
    for i in range(len(value)):
        now = value[i].item()
        while last < now:
            tile_range.append((last_i, i))
            last_i = i
            last += 1
    while last < grid[0] * grid[1]:
        tile_range.append((last_i, len(value)))
        last_i = len(value)
        last += 1
    tile_range = torch.tensor(tile_range, device=means2D.device, dtype=torch.long)
    return gs_indices, tile_range


class _fast_rasterize(torch.autograd.Function):
    _forward_func = get_C_function('GS_2D_fast_render_forward')
    _backward_func = get_C_function('GS_2D_fast_render_backward')

    @staticmethod
    def forward(ctx, *inputs):
        (W, H, means2D, colors, normal_opacity, trans_mat, inverse_m, tile_range, point_list, near_n, far_n, only_image,
         trans_mat_t2, accum_max_count, accum_weights_p, accum_weights_count,) = inputs
        pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib = _fast_rasterize._forward_func(
            W, H, near_n, far_n, only_image,
            means2D, colors, normal_opacity, trans_mat, inverse_m, point_list, tile_range, trans_mat_t2,
            accum_max_count, accum_weights_p, accum_weights_count, None
        )
        ctx._info = (near_n, far_n)
        ctx.save_for_backward(means2D, normal_opacity, trans_mat, inverse_m, trans_mat_t2, colors,
                              pixel_colors, pixel_extras, pixel_opacity, pixel_flow, tile_range, point_list, n_contrib)
        return pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib

    @staticmethod
    def backward(ctx, *grad_outputs):
        (means2D, normal_opacity, trans_mat, inverse_m, trans_mat_t2, colors,
         pixel_colors, pixel_extras, pixel_opacity, pixel_flow, tile_range, point_list,
         n_contrib) = ctx.saved_tensors
        grad_images, grad_opacity, grad_others, grad_flow, _ = grad_outputs
        grad_colors, grad_means2D, grad_trans, grad_inv_m, grad_normal_opacity = _fast_rasterize._backward_func(
            means2D, normal_opacity, colors, trans_mat, inverse_m, trans_mat_t2,
            pixel_colors, pixel_opacity, pixel_extras, pixel_flow,
            grad_images, grad_opacity, grad_others, grad_flow,
            tile_range, point_list, n_contrib,
            *ctx._info,
            None, None, None, None, None, None
        )
        grad_inputs = [None] * 21
        grad_inputs[2] = grad_means2D
        grad_inputs[3] = grad_colors
        grad_inputs[4] = grad_normal_opacity
        grad_inputs[5] = grad_trans
        grad_inputs[6] = grad_inv_m
        return tuple(grad_inputs)


class _fast_rasterize_v2(torch.autograd.Function):
    _forward_func = get_C_function('GS_2D_fast_render_forward')
    _backward_func = get_C_function('GS_2D_fast_render_backward')

    @staticmethod
    def forward(ctx, *inputs):
        (W, H, means2D, colors, normal_opacity, trans_mat, inverse_m, tile_range, point_list, near_n, far_n, only_image,
         trans_mat_t2, accum_max_count, accum_weights_p, accum_weights_count,) = inputs
        per_tile_bucket_offset = (tile_range[..., 1] - tile_range[..., 0] + 31) // 32
        per_tile_bucket_offset = torch.cumsum(per_tile_bucket_offset, dim=-1).contiguous()
        (pixel_colors, pixel_opacity, pixel_extras, pixel_flows, n_contrib,
         max_contrib, bucket_to_tile, sampled_T, sampled_acc, sampled_aux) = _fast_rasterize._forward_func(
            W, H, near_n, far_n, only_image,
            means2D, colors, normal_opacity, trans_mat, inverse_m, point_list, tile_range, trans_mat_t2,
            accum_max_count, accum_weights_p, accum_weights_count, per_tile_bucket_offset
        )
        ctx._info = (near_n, far_n)
        ctx.save_for_backward(means2D, normal_opacity, trans_mat, inverse_m, trans_mat_t2, colors,
                              pixel_colors, pixel_extras, pixel_opacity, pixel_flows,
                              tile_range, point_list, n_contrib, max_contrib,
                              per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_acc, sampled_aux)
        return pixel_colors, pixel_opacity, pixel_extras, pixel_flows, n_contrib

    @staticmethod
    def backward(ctx, *grad_outputs):
        (means2D, normal_opacity, trans_mat, inverse_m, trans_mat_t2, colors,
         pixel_colors, pixel_extras, pixel_opacity, pixel_flows,
         tile_range, point_list, n_contrib, max_contrib,
         per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_acc, sampled_aux) = ctx.saved_tensors
        grad_images, grad_opacity, grad_others, grad_flows, _ = grad_outputs
        grad_colors, grad_means2D, grad_trans, grad_inv_m, grad_normal_opacity = _fast_rasterize._backward_func(
            means2D, normal_opacity, colors, trans_mat, inverse_m, trans_mat_t2,
            pixel_colors, pixel_opacity, pixel_extras, pixel_flows,
            grad_images, grad_opacity, grad_others, grad_flows,
            tile_range, point_list, n_contrib,
            *ctx._info,
            per_tile_bucket_offset, bucket_to_tile, max_contrib, sampled_T, sampled_acc, sampled_aux
        )
        grad_inputs = [None] * 21
        grad_inputs[2] = grad_means2D
        grad_inputs[3] = grad_colors
        grad_inputs[4] = grad_normal_opacity
        grad_inputs[5] = grad_trans
        grad_inputs[6] = grad_inv_m
        return tuple(grad_inputs)


@try_use_C_extension(_fast_rasterize.apply, "GS_2D_fast_render_forward", "GS_2D_fast_render_backward")
def GS_2D_fast_rasterize(
    W: int, H: int, mean2D: Tensor, colors: Tensor, normal_opacity: Tensor, trans_mat: Tensor, inverse_m,
    tile_range: Tensor, point_list: Tensor,
    near_n=0.2, far_n=100., only_image=False,
    accum_max_count=None, accum_weights_p=None, accum_weights_count=None,
):
    grid_x = (W + _BLOCK_X - 1) // _BLOCK_X
    grid_y = (H + _BLOCK_Y - 1) // _BLOCK_Y
    images = mean2D.new_zeros(colors.shape[-1], H, W)
    n_contrib = normal_opacity.new_zeros(2, H, W, dtype=torch.int)
    opacities = normal_opacity.new_zeros(H, W)
    out_extra = mean2D.new_zeros(8, H, W)

    pix_id = 57 * W + 141
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
            gs_normal_opactiy = normal_opacity[index]  # [P, 4]
            gs_color = colors[index]
            gs_Tu, gs_Tv, gs_Tw = trans_mat[index].unbind(dim=1)

            k = xy[:, None, 0:1] * gs_Tw - gs_Tu
            l = xy[:, None, 1:2] * gs_Tw - gs_Tv
            p = torch.cross(k, l, dim=-1)

            mask = p[:, :, 2] != 0
            s = p[:, :, :2] / p[:, :, 2:]
            rho3d = s.square().sum(dim=-1)
            d_xy = xy[:, None, :] - gs_xy[None, :, :]
            rho2d = FilterInvSquare * d_xy.square().sum(dim=-1)
            rho = torch.minimum(rho3d, rho2d)
            depth = torch.where(rho3d <= rho2d, s[:, :, 0] * gs_Tw[:, 0] + s[:, :, 1] * gs_Tw[:, 1] + gs_Tw[:, 2],
                                gs_Tw[:, 2])
            mask = mask & (depth >= near_n)

            power = -0.5 * rho
            mask = mask & (power <= 0)
            alpha = (gs_normal_opactiy[:, 3] * power.exp()) * mask  # [T_W * T_H, P]
            alpha = alpha + (alpha.clamp_max(0.99) - alpha).detach()
            mask = mask & (alpha >= 1. / 255)
            alpha = torch.where(alpha < 1. / 255, torch.zeros_like(alpha), alpha)
            sigma = (1 - alpha).cumprod(dim=1)
            idx_ = torch.arange(W_ * H_, device=index.device, dtype=torch.int)
            with torch.no_grad():
                mask = torch.logical_and(mask, sigma >= 0.0001)
                num_used = (mask.flip(1).cumsum(dim=1) > 0).sum(dim=1)

            n_contrib[0, sy:ey, sx:ex] = num_used.reshape(H_, W_)
            opacities[sy:ey, sx:ex] = (1 - sigma[idx_, num_used - 1]).reshape(H_, W_)  # * mask.any(dim=1)
            m = far_n / (far_n - near_n) * (1 - near_n / depth)
            mid_idx = torch.searchsorted(-sigma, sigma.new_tensor([-0.5]).expand(sigma.shape[0], 1).contiguous())[:, 0]
            mask_ = mid_idx < sigma.shape[1]
            n_contrib[1, sy:ey, sx:ex] = torch.where(mask_, mid_idx + 1, torch.full_like(mid_idx, -1)).reshape(H_, W_)
            mid_idx = mid_idx.clamp(0, sigma.shape[1] - 1)
            out_extra[4, sy:ey, sx:ex] = (depth[idx_, mid_idx] * mask_).reshape(H_, W_)
            sigma = torch.cat([torch.ones_like(sigma[:, :1]), sigma[:, :-1]], dim=-1)
            A = 1 - sigma
            sigma = sigma * alpha * mask
            M1 = m * sigma
            M2 = M1 * m
            M1 = torch.cumsum(M1, dim=-1)
            M2 = torch.cumsum(M2, dim=-1)
            M1 = torch.constant_pad_nd(M1, (1, 0, 0, 0))
            M2 = torch.constant_pad_nd(M2, (1, 0, 0, 0))
            out_extra[5, sy:ey, sx:ex] = torch.einsum(
                'pi,pi->p', sigma, (m * m * A + M2[:, :-1] - 2 * m * M1[:, :-1])).reshape(H_, W_)
            images[:, sy:ey, sx:ex] = torch.einsum('pi,ij->jp', sigma, gs_color).reshape(-1, H_, W_)
            out_extra[1:4, sy:ey, sx:ex] = torch.einsum(
                'pi,ij->jp', sigma, gs_normal_opactiy[:, :3]).reshape(-1, H_, W_)
            out_extra[0, sy:ey, sx:ex] = torch.einsum('pi,pi->p', sigma, depth).reshape(H_, W_)
            out_extra[6, sy:ey, sx:ex] = M1[:, -1].reshape(H_, W_)
            out_extra[7, sy:ey, sx:ex] = M2[:, -1].reshape(H_, W_)
            # if torch.any(index == 3968):
            #     for i in range(W_ * H_):
            #         print(xy[i, 1] * W + xy[i, 0], num_used[i])

            # if x == debug_tx and y == debug_ty:
            #     print('\033[33m')
            #     print(f"{num_used[debug_i]=}")
            #     for i in range(len(index)):
            #         if mask[debug_i, i]:
            #             print(
            #                 f"gs={index[i].item():}, xy={gs_xy[i, 0]:.6f}, {gs_xy[i, 1]:.6f}, "
            #                 f"rho={rho[debug_i, i].item():.6f}, depth={depth[debug_i, i].item():.6f}, "
            #                 f"alpha={alpha[debug_i, i].item():.6f}, sigma={sigma[debug_i, i].item():.6f}"
            #             )
            #             # if index[i].item() == 2369245:
            #             #     print(f"rho3={rho3d[debug_i, i].item():.6f}, rho2={rho2d[debug_i, i].item():.6f}, ")
            #             #     print(f"s={s[debug_i, i]}, p={p[debug_i, i]}")
            #             #     print(f"k={k[debug_i, i]}, l={l[debug_i, i]}")
            #     print('\033[0m')

            #     def _show(mask_):
            #         def show_grad(grad):
            #             print(grad.shape, debug_i, mask_.shape)
            #             print('alpha grad:', grad[debug_i][mask_[debug_i]])
            #
            #         return show_grad
            #
            #     alpha.register_hook(_show(mask))
    return images, opacities, None if only_image else out_extra, n_contrib


def depths_to_points(Tw2v, Tv2c, size, depthmap):
    c2w = Tw2v.inverse()
    W, H = size
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ (Tv2c @ Tw2v).T
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=c2w.device).float(), torch.arange(H, device=c2w.device).float(),
        indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(Tw2v, Tw2c, size, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(Tw2v, Tw2c, size, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


class _Fast_Render(torch.autograd.Function):
    _preprocess_forward = get_C_function('gs_2d_fast_preprocess_forward')
    _preprocess_backward = get_C_function('gs_2d_fast_preprocess_backward')
    _render_forward = get_C_function('GS_2D_fast_render_forward')
    _render_backward = get_C_function('GS_2D_fast_render_backward')

    @staticmethod
    def forward(
        ctx, means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
        Tw2v, Tv2c, Tv2s, campos,
        size, sh_degree, is_opengl, near_n, far_n, only_image,
        trans_precomp, means2D, trans_mat_t2,
        culling, accum_max_count, accum_weights_p, accum_weights_count
    ):
        W, H = size
        means2D, colors, trans_mat, inverse_m, normal_opacity, depths, radii, tile_range, point_list = _Fast_Render._preprocess_forward(
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, Tv2s, campos,
            trans_precomp, means2D, culling,
            True  # debug
        )
        pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib = _Fast_Render._render_forward(
            W, H, near_n, far_n, only_image,
            means2D, colors, normal_opacity, trans_mat, inverse_m, point_list, tile_range, trans_mat_t2,
            accum_max_count, accum_weights_p, accum_weights_count, None
        )
        ctx._info = (W, H, sh_degree, is_opengl, near_n, far_n)
        ctx.save_for_backward(
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, Tv2s, campos,
            means2D, trans_mat, inverse_m, trans_mat_t2, radii, colors, normal_opacity, tile_range, point_list,
            pixel_colors, pixel_extras, pixel_opacity, pixel_flow, n_contrib
        )
        return pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib, radii

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        (
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, Tv2s, campos,
            means2D, trans_mat, inverse_m, trans_mat_t2, radii, colors, normal_opacity, tile_range, point_list,
            pixel_colors, pixel_extras, pixel_opacity, pixel_flow, n_contrib
        ) = ctx.saved_tensors

        W, H, sh_degree, is_opengl, near_n, far_n = ctx._info
        grad_images, grad_opacity, grad_others, grad_flow = grad_outputs[:4]
        grad_colors, grad_means2D, grad_trans, grad_inv_m, grad_normal_opacity = _Fast_Render._render_backward(
            means2D, normal_opacity, colors, trans_mat, inverse_m, trans_mat_t2,
            pixel_colors, pixel_opacity, pixel_extras, pixel_flow,
            grad_images, grad_opacity, grad_others, grad_flow,
            tile_range, point_list, n_contrib,
            near_n, far_n,
            None, None, None, None, None, None
        )
        grad_inputs = [
            torch.zeros_like(means3D), torch.zeros_like(scales), torch.zeros_like(rotations),
            torch.zeros_like(opacities),
            torch.zeros_like(shs_or_colors) if ctx.needs_input_grad[4] else None,
            torch.zeros_like(shs_rest) if shs_rest is not None and ctx.needs_input_grad[5] else None,
            None, None, None, None,
            None, None, None, None, None, None,
            None, grad_means2D, None,
            None, None, None, None,
        ]
        _Fast_Render._preprocess_backward(
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2s, campos,
            trans_mat, radii, colors, normal_opacity,
            grad_means2D, grad_trans, grad_inv_m, None, grad_colors, grad_normal_opacity,
            grad_inputs[0], grad_inputs[1], grad_inputs[2], grad_inputs[3], grad_inputs[4], grad_inputs[5],
            grad_inputs[6], grad_inputs[9]
        )
        return tuple(grad_inputs)


def GS_2D_fast_render(
    points: Tensor,
    scales: Tensor,
    rotations: Tensor,
    opacity: Tensor,
    size: Union[Tuple[int, int], Tensor],

    Tw2v: Tensor,
    Tv2c: Tensor,
    Tv2s: Tensor,
    campos: Tensor = None,
    focals: Tensor = None,
    FoV: Tensor = None,
    is_opengl=False,
    near=0.2,
    far=100,

    sh_degree=0,
    colors: Tensor = None,
    sh_features: Tensor = None,
    sh_features_rest: Tensor = None,

    depth_ratio=1.,
    culling: Tensor = None,
    accum_max_count: Tensor = None,
    accum_weights_p: Tensor = None,
    accum_weights_count: Tensor = None,
    trans_mat_t2: Tensor = None,
    **kwargs,
):
    means2D = points.new_zeros(points.shape[0], 2, requires_grad=True)
    W, H = size
    if campos is None:
        campos = Tw2v.inverse()[:3, 3].contiguous()
    # if focals is None:
    #     focals = ops_3d.fov_to_focal(FoV, FoV.new_tensor(size))
    # try:
    #     means2D.retain_grad()
    # except:
    #     pass
    # means2D_, colors, trans_mat, normal_opacity, depths, radii, tiles_touched = GS_2D_preprocess(
    #     W, H, sh_degree if colors is None else -1, is_opengl,
    #     points, scales, rotations, opacity, sh_features if colors is None else colors, sh_features_rest,
    #     Tw2v, Tv2c, campos, None, means2D
    # )
    # colors, means2D, trans_mat, normal_opacity, depths, radii, tiles_touched = _GS_2D_preprocess.apply(
    #     W, H, sh_degree, is_opengl,
    #     points, scales, rotations, opacity, sh_features if colors is None else colors, sh_features_rest,
    #     Tw2v, Tv2c, campos, None, means2D)
    #
    # tile_range, point_list = GS_prepare(W, H, means2D, depths, radii, tiles_touched, False)
    # images, opacities, out_extras, n_contrib, = GS_2D_rasterize(
    #     W, H, means2D, colors, normal_opacity, trans_mat, tile_range, point_list, 0.2, 100., False,
    # )
    images, opacities, out_extras, flows, n_contrib, radii = _Fast_Render.apply(
        points, scales, rotations, opacity, sh_features if colors is None else colors, sh_features_rest,
        Tw2v.contiguous(), Tv2c.contiguous(), Tv2s.contiguous(), campos.contiguous(),
        size, sh_degree if colors is None else -1, is_opengl, near, far, False,
        None, means2D, trans_mat_t2,
        culling, accum_max_count, accum_weights_p, accum_weights_count,
    )
    depths, normals, median_depths, distortion, _ = out_extras.split([1, 3, 1, 1, 2], dim=0)
    normals = (normals.permute(1, 2, 0) @ Tw2v[:3, :3]).permute(2, 0, 1)
    median_depths = torch.nan_to_num(median_depths, 0, 0)
    depths = torch.nan_to_num(depths, 0, 0)
    surf_depth = depths * (1 - depth_ratio) + depth_ratio * median_depths
    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(Tw2v, Tv2c, (W, H), surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is normalized.
    surf_normal = surf_normal * opacities.detach()
    outputs = {
        'images': images,
        # 'depths': median_depths,
        'normals': normals,
        'opacity': opacities,
        'distortion': distortion,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'radii': radii,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        'flows': flows,
    }
    return outputs


class _fast_compute_trans_mat(torch.autograd.Function):
    _forward_func = get_C_function('gs_2d_fast_compute_transmat_forward')
    _backward_func = get_C_function('gs_2d_fast_compute_trans_mat_backward')

    @staticmethod
    def forward(ctx, *inputs):
        means3D, scales, rotations, Tw2v, Tv2s = inputs
        matrix, normals = _fast_compute_trans_mat._forward_func(means3D, scales, rotations, Tw2v, Tv2s)

        ctx.save_for_backward(means3D, scales, rotations, Tw2v, Tv2s)
        return matrix, normals

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_matrix, grad_normals = grad_outputs
        means3D, scales, rotations, Tw2v, Tv2s = ctx.saved_tensors
        grad_means3D, grad_scales, grad_rotations = _fast_compute_trans_mat._backward_func(
            means3D, scales, rotations, Tw2v, Tv2s, grad_matrix, grad_normals
        )
        return grad_means3D, grad_scales, grad_rotations, None, None


def fast_compuate_transmat(means3D: Tensor, scales: Tensor, rotations: Tensor, Tw2v: Tensor, Tv2s: Tensor):
    from fast_2d_gs.renderer.gs_2d_render import quat_toR
    # L = ops_3d.quaternion.toR(rotations)
    L = quat_toR(rotations)
    normal = ops_3d.apply(L[:, :, 2], Tw2v[:3, :3])
    L = torch.stack([L[:, :, 0] * scales[:, 0:1], L[:, :, 1] * scales[:, 1:2], means3D], dim=-1)
    tmp = Tw2v[:3, :3] @ L
    tmp[..., 2, 2] += Tw2v[2, 3]
    # ndc2pix = Tw2v.new_tensor([[0.5 * W, 0, 0, 0.5 * (W)], [0, 0.5 * H, 0, 0.5 * (H)], [0, 0, 0, 1.]])
    # Tv2s = ndc2pix @ Tv2c
    return Tv2s @ tmp, normal


def ray_splat_intersection(M: Tensor, pixels: Tensor):
    hx = torch.zeros((*pixels.shape[:-1], 3), device=M.device)
    hx[..., 0] = -1
    hx[..., -1] = pixels[..., 0]
    hy = torch.zeros((*pixels.shape[:-1], 3), device=M.device)
    hy[..., 1] = -1
    hy[..., -1] = pixels[..., 1]
    hu = hx @ M
    hv = hy @ M
    u = (hu[..., 1] * hv[..., -1] - hu[..., -1] * hv[..., 1]) / (hu[..., 0] * hv[..., 1] - hu[..., 1] * hv[..., 0])
    v = (hu[..., -1] * hv[..., 0] - hu[..., 0] * hv[..., -1]) / (hu[..., 0] * hv[..., 1] - hu[..., 1] * hv[..., 0])
    d = u * M[..., -1, 0] + v * M[..., -1, 1] + M[..., -1, 2]
    return torch.stack([u, v], dim=-1), d


def ray_splat_intersection_fast(M: Tensor, pixels: Tensor):
    uv = torch.zeros_like(pixels, dtype=M.dtype, device=M.device)
    a = (M[..., 2, 1] * M[..., 0, 2] - M[..., 0, 1] * M[..., 2, 2]) * pixels[..., 1] + \
        (M[..., 1, 1] * M[..., 2, 2] - M[..., 2, 1] * M[..., 1, 2]) * pixels[..., 0] + \
        (M[..., 0, 1] * M[..., 1, 2] - M[..., 1, 1] * M[..., 0, 2])
    b = (M[..., 0, 0] * M[..., 2, 2] - M[..., 2, 0] * M[..., 0, 2]) * pixels[..., 1] + \
        (M[..., 2, 0] * M[..., 1, 2] - M[..., 1, 0] * M[..., 2, 2]) * pixels[..., 0] + \
        (M[..., 1, 0] * M[..., 0, 2] - M[..., 0, 0] * M[..., 1, 2])
    c = (M[..., 2, 0] * M[..., 0, 1] - M[..., 0, 0] * M[..., 2, 1]) * pixels[..., 1] + \
        (M[..., 1, 0] * M[..., 2, 1] - M[..., 2, 0] * M[..., 1, 1]) * pixels[..., 0] + \
        (M[..., 0, 0] * M[..., 1, 1] - M[..., 1, 0] * M[..., 0, 1])
    uv[..., 0] = a / c
    uv[..., 1] = b / c
    # from extension.utils.test_utils import get_rel_error
    # hx = torch.zeros((*pixels.shape[:-1], 3), device=M.device)
    # hx[..., 0] = -1
    # hx[..., -1] = pixels[..., 0]
    # hy = torch.zeros((*pixels.shape[:-1], 3), device=M.device)
    # hy[..., 1] = -1
    # hy[..., -1] = pixels[..., 1]
    # hu = hx @ M
    # hv = hy @ M
    # get_rel_error(hu[..., 1] * hv[..., -1] - hu[..., -1] * hv[..., 1], a, 'A')
    # get_rel_error(hu[..., -1] * hv[..., 0] - hu[..., 0] * hv[..., -1], b, 'B')
    # get_rel_error(hu[..., 0] * hv[..., 1] - hu[..., 1] * hv[..., 0], c, 'C')
    # print(*c.abs().aminmax())
    # u = (hu[..., 1] * hv[..., -1] - hu[..., -1] * hv[..., 1])  # / (hu[..., 0] * hv[..., 1] - hu[..., 1] * hv[..., 0])
    # v = (hu[..., -1] * hv[..., 0] - hu[..., 0] * hv[..., -1])  # / (hu[..., 0] * hv[..., 1] - hu[..., 1] * hv[..., 0])
    # d = (hu[..., 0] * hv[..., 1] - hu[..., 1] * hv[..., 0])
    # print(utils.show_shape(u, v, c, d))
    # get_rel_error(u * c, a * d, 'u')
    # get_rel_error(v * c, b * d, 'v')
    d = uv[..., 0] * M[..., -1, 0] + uv[..., 1] * M[..., -1, 1] + M[..., -1, 2]
    return uv, d


def is_block_intersect_ellipse(M: Tensor, pixels: Tensor, opacity: Tensor, W, H):
    Ax = (M[..., 1, 1] * M[..., 2, 2] - M[..., 2, 1] * M[..., 1, 2])
    Ay = (M[..., 2, 1] * M[..., 0, 2] - M[..., 0, 1] * M[..., 2, 2])
    Az = (M[..., 0, 1] * M[..., 1, 2] - M[..., 1, 1] * M[..., 0, 2])
    Bx = (M[..., 2, 0] * M[..., 1, 2] - M[..., 1, 0] * M[..., 2, 2])
    By = (M[..., 0, 0] * M[..., 2, 2] - M[..., 2, 0] * M[..., 0, 2])
    Bz = (M[..., 1, 0] * M[..., 0, 2] - M[..., 0, 0] * M[..., 1, 2])
    Cx = (M[..., 1, 0] * M[..., 2, 1] - M[..., 2, 0] * M[..., 1, 1])
    Cy = (M[..., 2, 0] * M[..., 0, 1] - M[..., 0, 0] * M[..., 2, 1])
    Cz = (M[..., 0, 0] * M[..., 1, 1] - M[..., 1, 0] * M[..., 0, 1])
    print('M`:', Ax.item(), Ay.item(), Az.item(), Bx.item(), By.item(), Bz.item(), Cx.item(), Cy.item(), Cz.item())
    x, y = pixels.float().unbind(dim=-1)
    # a = Ax * x + Ay * y + Az
    # b = Bx * x + By * y + Bz
    # c = Cx * x + Cy * y + Cz
    # uv = [a/c, b/c]
    d = -2 * torch.log(255. * opacity)
    print('d:', d.item(), opacity.item())
    # Ax[:], Ay[:], Az[:], Bx[:], By[:], Bz[:], Cx[:], Cy[:], Cz[:] = \
    #     -155.206070, 74.927078, -9226.741211, -156.358932, 125.012260, 2197.281738, -58.675568, 49.692616, -3199.211182
    # d[:] = -10.769946

    Wxx = (Ax * Ax + Bx * Bx + d * Cx * Cx)
    Wyy = (Ay * Ay + By * By + d * Cy * Cy)
    Wxy = 2 * (Ax * Ay + Bx * By + d * Cx * Cy)
    Wx = 2 * (Ax * Az + Bx * Bz + d * Cx * Cz)
    Wy = 2 * (Ay * Az + By * Bz + d * Cy * Cz)
    W_ = Az * Az + Bz * Bz + Cz * Cz * d
    cx = (M[..., 0, 2] / M[..., 2, 2]).item()
    cy = (M[..., 1, 2] / M[..., 2, 2]).item()
    print('center:', cx, cy)
    print(f'{cx=}, {cy=}', -(Wxy * cy + Wx), (2 * Wxx))
    in_ellipse = Wxx * x * x + Wyy * y * y + Wxy * x * y + Wx * x + Wy * y + W_ <= 0
    points = _gs_tile_intersect(cx, cy, Wxx.item(), Wyy.item(), Wxy.item(), Wx.item(), Wy.item(), W_.item(), W, H, True)
    mask = np.zeros((H, W), dtype=int)
    for x, y in points:
        mask[y * 16:y * 16 + 16, x * 16:x * 16 + 16] = 1
    return in_ellipse, Wyy, Wxy, mask


def _gen_random_data(P=1024, seed: int = None):
    seed = np.random.randint(0, int(1e9)) if seed is None else seed
    print(f'{seed=}')
    torch.manual_seed(seed)

    device = torch.device('cuda')
    sh_degree = 0
    means3D = torch.randn(P, 3).cuda()
    scales = torch.rand(P, 2).cuda()
    rotations = ops_3d.quaternion.normalize(torch.randn(P, 4).cuda())
    opacit = torch.rand(P, 1, device=device)  # [0, 1]
    colors = None  # torch.randn(P, 3, device=device)  # [0, 1]
    shs = torch.randn(P, (1 + sh_degree) ** 2, 3, device=device)

    Tw2v = ops_3d.opencv.look_at(torch.tensor([0, 0, 4.]), torch.zeros(3)).cuda()
    campos = torch.inverse(Tw2v)[:3, 3]
    focal = 500
    W, H = 128, 128
    fov_x, fov_y = ops_3d.focal_to_fov(focal, W, H)
    FoV = torch.tensor([fov_x, fov_y])
    # tan_fovx, tan_fovy = np.tan(0.5 * fov_x).item(), np.tan(0.5 * fov_y).item()
    Tv2c = ops_3d.opencv.perspective(fov_y, size=(W, H)).to(Tw2v)
    # Tw2c = Tv2c @ Tw2v
    Tv2s = ops_3d.opencv.camera_intrinsics(size=(W, H), fov=torch.tensor([fov_x, fov_y])).to(device)
    focals = torch.tensor([focal, focal], device=device)
    ndc2pix = Tw2v.new_tensor([[0.5 * W, 0, 0, 0.5 * (W - 1)], [0, 0.5 * H, 0, 0.5 * (H - 1)], [0, 0, 0, 1.]])
    Tv2s = (ndc2pix @ Tv2c)[:3, :3].contiguous()
    return W, H, sh_degree, means3D, scales, rotations, opacit, shs, Tw2v, Tv2c, campos, Tv2s, FoV


def test_fast_compuate_transmat():
    from utils.test_utils import get_run_speed, clone_tensors, get_rel_error
    from fast_2d_gs.renderer.gs_2d_render import GS_2D_compute_trans_mat, _GS_2D_compute_trans_mat
    W, H, sh_degree, means3D, scales, rotations, opacit, shs, Tw2v, Tv2c, campos, Tv2s, FoV = _gen_random_data(10240)

    py1_func = GS_2D_compute_trans_mat
    cu1_func = _GS_2D_compute_trans_mat.apply
    py2_func = fast_compuate_transmat
    cu2_func = _fast_compute_trans_mat.apply

    means3D_py1, scales_py1, rotations_py1 = clone_tensors(means3D, scales, rotations, device='cuda')
    mat_py1, normal_py1 = py1_func(means3D_py1, scales_py1, rotations_py1, Tw2v, Tv2c, W, H)

    means3D_cu1, scales_cu1, rotations_cu1 = clone_tensors(means3D, scales, rotations, device='cuda')
    mat_cu1, normal_cu1 = cu1_func(means3D_cu1, scales_cu1, rotations_cu1, Tw2v, Tv2c, W, H)

    means3D_py2, scales_py2, rotations_py2 = clone_tensors(means3D, scales, rotations, device='cuda')
    mat_py2, normal_py2 = py2_func(means3D_py2, scales_py2, rotations_py2, Tw2v, Tv2s)

    means3D_cu2, scales_cu2, rotations_cu2 = clone_tensors(means3D, scales, rotations, device='cuda')
    mat_cu2, normal_cu2 = cu2_func(means3D_cu2, scales_cu2, rotations_cu2, Tw2v, Tv2s)

    get_rel_error(mat_py2, mat_py1, 'mat py2')
    get_rel_error(mat_cu1, mat_py1, 'mat cu1')
    get_rel_error(mat_cu2, mat_py1, 'mat cu2')
    get_rel_error(normal_py2, normal_py1, 'normal py2')
    get_rel_error(normal_cu1, normal_py1, 'normal cu1')
    get_rel_error(normal_cu2, normal_py1, 'normal cu2')

    g_mat = torch.randn_like(mat_py1)
    g_normal = torch.randn_like(normal_py1)
    torch.autograd.backward([mat_cu1, normal_cu1], [g_mat, g_normal])
    torch.autograd.backward([mat_py1, normal_py1], [g_mat, g_normal])
    torch.autograd.backward([mat_py2, normal_py2], [g_mat, g_normal])
    torch.autograd.backward([mat_cu2, normal_cu2], [g_mat, g_normal])
    get_rel_error(means3D_cu1.grad, means3D_py1.grad, "cu1 grad_means3d")
    get_rel_error(scales_cu1.grad, scales_py1.grad, "cu1 grad_scale")
    get_rel_error(rotations_cu1.grad, rotations_py1.grad, "cu1  grad_rotation")
    get_rel_error(means3D_py2.grad, means3D_py1.grad, "py2 grad_means3d")
    get_rel_error(scales_py2.grad, scales_py1.grad, "py2 grad_scale")
    get_rel_error(rotations_py2.grad, rotations_py1.grad, "py2 grad_rotation")
    get_rel_error(means3D_cu2.grad, means3D_py1.grad, "cu2 grad_means3d")
    get_rel_error(scales_cu2.grad, scales_py1.grad, "cu2 grad_scale")
    get_rel_error(rotations_cu2.grad, rotations_py1.grad, "cu2 grad_rotation")

    get_run_speed((means3D_cu1, scales_cu1, rotations_cu1, Tw2v, Tv2c, W, H), (g_mat, g_normal),
                  cu1_func=cu1_func, py1_func=py1_func)
    get_run_speed((means3D_cu1, scales_cu1, rotations_cu1, Tw2v, Tv2s), (g_mat, g_normal),
                  py2_func=py2_func, cu2_func=cu2_func)


def test_precise_intersection():
    """精确相交算法确定每个Tile对应的高斯"""
    from utils.test_utils import get_rel_error
    from fast_2d_gs.renderer.gs_2d_render import GS_2D_render, GS_2D_preprocess
    seed = 5  # int(time.time())
    print(f"{seed=}")
    torch.manual_seed(seed)
    ops_3d.set_coord_system('opencv')
    utils.set_printoptions(6)
    device = torch.device('cuda')
    # torch.set_default_dtype(torch.float64)
    sh_degree = -1
    P = 1
    means3D = torch.randn(P, 3).cuda()
    scales = torch.rand(P, 2).cuda() * 0.25 + 0.25
    # scales = torch.full((P, 2), 1e-3, device=device)  # very small
    rotations = ops_3d.quaternion.normalize(torch.zeros(P, 4) + torch.tensor([0, 0, 0, 1])).cuda()
    rotations = ops_3d.quaternion.normalize(torch.randn(P, 4)).cuda()
    # opacity = torch.randn(P, 1, device=device)  # [0, 1]
    opacity = torch.ones(P, 1, device=device)  # [0, 1]
    shs = torch.ones(P, 3, device=device)  # [0, 1]
    # shs = torch.randn(P, (1 + sh_degree) ** 2, 3, device=device)

    Tw2v = ops_3d.opencv.look_at(torch.tensor([0, 0, 4.]), torch.zeros(3)).cuda()
    campos = torch.inverse(Tw2v)[:3, 3]
    W, H = 512, 512
    focal = W
    fov_x, fov_y = ops_3d.focal_to_fov(focal, W, H)
    Tv2c = ops_3d.opencv.perspective(fov_y, size=(W, H)).to(Tw2v)
    # Tv2s = ops_3d.opencv.camera_intrinsics(focal=focal, size=(W, H)).to(device)
    ndc2pix = Tw2v.new_tensor([[0.5 * W, 0, 0, 0.5 * (W - 1)], [0, 0.5 * H, 0, 0.5 * (H - 1)], [0, 0, 0, 1.]])
    Tv2s = (ndc2pix @ Tv2c)[:3, :3].contiguous().to(device)
    print(Tv2s)

    W, H = 128, 128
    # means3D = torch.tensor([[-0.3917272686958313, -0.10304337739944458, 2.5885653495788574]], device=device)
    # scales = torch.tensor([[0.699916660785675, 0.5541119575500488]], device=device)
    # rotations = torch.tensor([[-0.39769402146339417, 0.2379016876220703, 0.4948843717575073, -0.7350726127624512]],
    #                          device=device)
    # opacity = torch.tensor([[0.7443339228630066]], device=device)
    # shs = torch.tensor([[0.593890905380249, -0.874317467212677, 0.3221900165081024]], device=device)
    # Tw2v = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 4.0], [0.0, 0.0, 0.0, 1.0]],
    #                     device=device)
    # Tv2c = torch.tensor(
    #     [[7.8125, 0.0, 0.0, 0.0], [0.0, 7.8125, 0.0, 0.0], [0.0, 0.0, 1.000100016593933, -0.10001000016927719],
    #      [0.0, 0.0, 1.0, 0.0]], device=device)
    # campos = torch.tensor([0.0, 0.0, 4.0], device=device)
    # Tv2s = torch.tensor([[500.0, 0.0, 63.5], [0.0, 500.0, 63.5], [0.0, 0.0, 1.0]], device=device)
    means3D = torch.tensor([[-1.032691240310669, 0.18953491747379303, 2.672884941101074]], device=device)
    scales = torch.tensor([[0.48984575271606445, 0.31145793199539185]], device=device)
    rotations = torch.tensor([[-0.6498619318008423, 0.3036384880542755, 0.13403694331645966, 0.6837523579597473]],
                             device=device)
    opacity = torch.tensor([[0.8064265251159668]], device=device)
    shs = torch.tensor([[0.1557363122701645, -0.29530543088912964, 1.3676812648773193]], device=device)
    Tw2v = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 4.0], [0.0, 0.0, 0.0, 1.0]],
                        device=device)
    Tv2c = torch.tensor(
        [[7.8125, 0.0, 0.0, 0.0], [0.0, 7.8125, 0.0, 0.0], [0.0, 0.0, 1.000100016593933, -0.10001000016927719],
         [0.0, 0.0, 1.0, 0.0]], device=device)
    campos = torch.tensor([0.0, 0.0, 4.0], device=device)
    Tv2s = torch.tensor([[500.0, 0.0, 63.5], [0.0, 500.0, 63.5], [0.0, 0.0, 1.0]], device=device)

    torch.no_grad().__enter__()
    print('scales', scales)
    colors, means2D, trans_mat_, normal_opacity, depths, radii, tiles_touched = GS_2D_preprocess(
        W, H, sh_degree, False,
        means3D, scales, rotations, opacity, shs, None,
        Tw2v, Tv2c, campos, trans_precomp=None, means2D=None,
    )
    trans_mat, normal = fast_compuate_transmat(means3D, scales, rotations, Tw2v, Tv2s)
    print('trans_mat', trans_mat)
    get_rel_error(trans_mat_, trans_mat, 'trans mat')

    print('means2D', means2D)
    print('radii', radii)
    tile_range, point_list = GS_prepare(W, H, means2D, depths, radii, tiles_touched, False)
    # print(tile_range, point_list)
    output_my = GS_2D_render(means3D, scales, rotations, opacity, sh_features=shs,
                             Tw2v=Tw2v, Tv2c=Tv2c, campos=campos, size=(W, H), sh_degree=sh_degree)
    print(output_my['radii'], output_my['viewspace_points'])
    print(utils.show_shape(output_my))
    pixels = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).cuda()
    uv_fast, _ = ray_splat_intersection_fast(trans_mat, pixels)
    uv, _ = ray_splat_intersection(trans_mat, pixels)
    get_rel_error(uv_fast, uv.squeeze(), 'ray_splat_intersection_fast')

    img_ = torch.minimum(uv.square().sum(dim=-1), (pixels - means2D.view(-1)).square().sum(dim=-1) * 2) * -0.5
    img_ = torch.clamp_max(opacity * img_.exp(), 0.99)
    img_[img_ < 1 / 255.] = 0
    img_ = img_.reshape(H, W)
    # cmp_cx, cmp_cy = 50, 110
    # print(img_[cmp_cy - 4:cmp_cy + 4, cmp_cx - 4:cmp_cx + 4])
    # print(output_my['opacity'][cmp_cy - 4:cmp_cy + 4, cmp_cx - 4:cmp_cx + 4])
    # plt.subplot(131)
    # plt.imshow(img_.cpu())
    # plt.subplot(132)
    # plt.imshow(output_my['opacity'].cpu())
    # plt.subplot(133)
    # plt.imshow((output_my['opacity'] - img_).abs().cpu())
    # plt.show()
    get_rel_error(img_, output_my['opacity'], 'opacity')
    # return
    plt.figure(figsize=(4 * 4, 4 * 4))
    # plt.subplot(1, 5, (1, 4))
    # plt.subplot(121)
    mask = torch.any(output_my['images'] > 0, dim=0)  # .detach().cpu().numpy()
    img = output_my['images'].detach().permute(1, 2, 0).clamp(0, 1)  # utils.as_np_image(output_my['images'])
    img = utils.mask_boundary_add(img, mask, color=(1., 0, 0), kernel_size=3)
    # plt.subplot(121)
    for i in range(0, W, 16):
        plt.axvline(i, 0, H, c='w', alpha=0.3)
    for i in range(0, H, 16):
        plt.axhline(i, 0, W, c='w', alpha=0.3)
    # img = torch.cat([img, torch.full_like(img[:, :, :1], 0.5)], dim=-1)
    # for i, (start, end) in enumerate(tile_range):
    #     if start != end:
    #         x, y = i // (W // 16) * 16, i % (W // 16) * 16
    #         img[y:y + 16, x:x + 16, -1] = 1.0
    img = utils.as_np_image(img)
    plt.imshow(img)
    cx, cy = means2D[0].detach().cpu().numpy()
    # plt.scatter(cx, cy, c='r')
    print('center:', trans_mat @ trans_mat.new_tensor([0, 0, 1.]), 'vs', means2D[0])
    # compute aabb
    cutoff = 3.0
    FilterSize = 0.707106  # sqrt(2) / 2
    t = trans_mat.new_tensor([cutoff ** 2, cutoff ** 2, -1.])
    d = ops_3d.dot(t, trans_mat[:, 2] ** 2)
    f = (1 / d) * t
    p = torch.cat([
        ops_3d.dot(f, trans_mat[:, 0] * trans_mat[:, 2]),
        ops_3d.dot(f, trans_mat[:, 1] * trans_mat[:, 2])
    ], dim=-1)
    # print('p:', p[debug_i], p.shape)
    h = p * p - torch.cat([
        ops_3d.dot(f, trans_mat[:, 0] * trans_mat[:, 0]),
        ops_3d.dot(f, trans_mat[:, 1] * trans_mat[:, 1])
    ], dim=-1)
    print('h:', h)
    t = trans_mat.new_tensor([cutoff, cutoff, -1.])
    v = trans_mat @ trans_mat @ t
    print(v[..., :2] / v[..., 2:])

    h = h.clamp_min(1e-4).sqrt()
    radius = torch.ceil(h.amax(dim=-1).clamp_min(cutoff * FilterSize)).item()
    print('radius:', radius, 'center:', [cx, cy])
    ax = plt.gca()

    extent = 0
    # pixels = torch.stack(torch.meshgrid(torch.linspace(-extent, extent, 5000),
    #                                     torch.linspace(-extent, extent, 5000), indexing='xy'), dim=-1).cuda()
    # uv_fast, d = ray_splat_intersection_fast(trans_mat, pixels)
    # img_ = (uv_fast.square().sum(dim=-1)) * -0.5
    # img_ = torch.clamp_max(opacity * img_.exp(), 0.99)
    # img_[img_ < 1 / 255.] = 0
    # # print(img_.shape, img_.aminmax(), pixels.shape, d)
    # plt.imshow(img_.cpu().numpy(), extent=(-extent, extent, extent, -extent))
    #
    # # plt.show()
    # # return
    # ax.add_patch(patches.Rectangle((cx - radius, cy - radius), radius * 2, radius * 2,
    #                                linewidth=3, edgecolor='y', facecolor='none'))
    # h = h[0]
    # ax.add_patch(patches.Rectangle((cx - h[0].item(), cy - h[1].item()), h[0].item() * 2, h[1].item() * 2,
    #                                linewidth=3, edgecolor='g', facecolor='none'))
    # print(p, radius)
    # plt.show()
    # return

    p_axis = trans_mat.new_tensor([[[0, 0, 1]], [[cutoff, 0, 1]], [[0, cutoff, 1]]]) @ trans_mat.transpose(-1, -2)
    p_axis = (p_axis[:, 0, :2] / p_axis[:, 0, 2:]).detach().cpu().numpy()
    print('p_axis:', p_axis)
    plt.plot(p_axis[(0, 1), 0], p_axis[(0, 1), 1], ls='--', c='r')
    plt.plot(p_axis[(0, 2), 0], p_axis[(0, 2), 1], ls='--', c='y')

    # pixels = torch.stack(torch.meshgrid(torch.arange(0, W + 1, 16), torch.arange(0, H + 1, 16), indexing='xy'), dim=-1)
    pixels = pixels.cpu()
    mask, Wyy, Wxy, valid = is_block_intersect_ellipse(trans_mat, pixels.cuda(), opacity, W, H)
    # mask = mask.cpu()
    # valid = np.stack([valid, np.zeros_like(valid), np.zeros_like(valid)], axis=-1) * 80
    # plt.imshow(valid, alpha=1.0)
    # plt.scatter(pixels[mask][:, 0], pixels[mask][:, 1], c='cyan', marker='x', alpha=0.01)

    means2D, colors, trans_mat, inverse_m, normal_opacity, depths, radii, tile_range, points_list = GS_2D_fast_preprocess(
        W, H, -1, False, means3D, scales, rotations, opacity, shs, None,
        Tw2v, Tv2c, Tv2s, campos,
        None, None, None)
    print(utils.show_shape(points_list, tile_range))
    valid = np.zeros((H, W, 3), dtype=int)
    for y in range((H + _BLOCK_Y - 1) // _BLOCK_Y):
        BX = (W + _BLOCK_X - 1) // _BLOCK_X
        for x in range(BX):
            if tile_range[y * BX + x, 0] != tile_range[y * BX + x, 1]:
                # print(x, y)
                valid[y * _BLOCK_Y:y * _BLOCK_Y + _BLOCK_Y, x * _BLOCK_X:x * _BLOCK_X + _BLOCK_X, :1] += 80
    print(trans_mat)
    print('num valid', np.sum(valid[:, :, 0] > 0) // (_BLOCK_X * _BLOCK_Y))
    # print(tile_range[21 * 32:21 * 32 + 32, 1] - tile_range[21 * 32:21 * 32 + 32, 0], tile_range.max().item())
    # print(points_list)
    plt.imshow(valid, alpha=0.7)

    # img_ = ((uv.square().sum(dim=-1) * -0.5).exp() * opacity).cpu().numpy()
    # y_max = np.argmax(img_, axis=0)
    # x_max = np.arange(0, W)
    # m = img_[y_max, x_max] > 1 / 255
    # plt.plot(x_max[m], y_max[m], c='g')
    # x_max = np.argmax(img_, axis=1)
    # y_max = np.arange(0, H)
    # m = img_[y_max, x_max] > 1 / 255
    # plt.plot(x_max[m], y_max[m], c='g')
    # # plt.xticks(range(0, W + 1, 16))
    # # plt.yticks(range(0, H + 1, 16))
    # for i in range(4):
    #     plt.subplot(4, 5, 5 * (i + 1))
    #     plt.plot(np.arange(H), img_[208 + i * 16])
    d_range = extent  # 200
    plt.xlim(-d_range, W + d_range)
    plt.ylim(-d_range, H + d_range)
    plt.show()


def test_rasterize_fast():
    from utils.test_utils import get_rel_error, clone_tensors, get_run_speed
    from fast_2d_gs.renderer.gs_2d_render import _GS_2D_rasterize, _GS_2D_preprocess_py
    utils.set_printoptions(6)
    print()
    device = torch.device('cuda')
    (W, H, sh_degree, means3D, scales, rotations, opacit, shs, Tw2v, Tv2c, campos, Tv2s, FoV) = _gen_random_data(102400)
    near_n, far_n = 0.2, 100.
    only_image = False
    means2D, colors, trans_mat, inverse_m, normal_opacity, depths, radii, tile_range, point_list = _Fast_Render._preprocess_forward(
        W, H, sh_degree, False,
        means3D, scales, rotations, opacit, shs, None,
        Tw2v, Tv2c, Tv2s, campos,
        None, None, None,
        True  # debug
    )
    means2D, colors, trans_mat, normal_opacity, depths, radii, tiles_touched = _GS_2D_preprocess_py(
        W, H, sh_degree, False,
        means3D, scales, rotations, opacit, shs, None,
        Tw2v, Tv2c, campos, trans_precomp=None, means2D=None,
    )
    trans_mat_t2 = fast_compuate_transmat(
        means3D + torch.randn_like(means3D) * 0.1, scales + torch.randn_like(scales) * 0.001, rotations, Tw2v, Tv2s)[0]
    trans_mat_t2 = trans_mat_t2 * (radii[:, None, None] > 0)
    # get_rel_error(trans_mat, trans_mat_t2, 'trans_mat')
    tile_range, point_list = GS_prepare(W, H, means2D, depths, radii, tiles_touched, False)

    cu_func = _GS_2D_rasterize.apply
    # cu_fast = _fast_rasterize.apply
    cu_fast = _fast_rasterize_v2.apply

    means2D_cu, colors_cu, normal_opacity_cu, trans_mat_cu = clone_tensors(
        means2D, colors, normal_opacity, trans_mat, device=device)
    images_cu, opacities_cu, out_extra_cu, out_flow_cu, n_contrib_cu = cu_func(
        W, H, means2D_cu, colors_cu, normal_opacity_cu, trans_mat_cu, tile_range, point_list, near_n, far_n,
        only_image, trans_mat_t2, None, None, None)

    means2D_ft, colors_ft, normal_opacity_ft, trans_mat_ft, inverse_m_ft = clone_tensors(
        means2D, colors, normal_opacity, trans_mat, inverse_m, device=device)
    images_ft, opacities_ft, out_extra_ft, out_flow_ft, n_contrib_ft = cu_fast(
        W, H, means2D_ft, colors_ft, normal_opacity_ft, trans_mat_ft, inverse_m_ft, tile_range, point_list, near_n,
        far_n, only_image, trans_mat_t2, None, None, None)

    def _get_grad_M(M, gIM, gM):
        for (a, b), (c, d), (e, f) in [
            [(0, 0), (1, 1), (2, 2)], [(0, 1), (2, 1), (0, 2)], [(0, 2), (0, 1), (1, 2)],
            [(1, 0), (2, 0), (1, 2)], [(1, 1), (0, 0), (2, 2)], [(1, 2), (1, 0), (0, 2)],
            [(2, 0), (1, 0), (2, 1)], [(2, 1), (2, 0), (0, 1)], [(2, 2), (0, 0), (1, 1)]
        ]:
            gM[..., c, d] += gIM[..., a, b] * M[..., e, f]
            gM[..., e, f] += gIM[..., a, b] * M[..., c, d]
        for (a, b), (c, d), (e, f) in [
            [(0, 0), (2, 1), (1, 2)], [(0, 1), (0, 1), (2, 2)], [(0, 2), (1, 1), (0, 2)],
            [(1, 0), (1, 0), (2, 2)], [(1, 1), (2, 0), (0, 2)], [(1, 2), (0, 0), (1, 2)],
            [(2, 0), (2, 0), (1, 1)], [(2, 1), (0, 0), (2, 1)], [(2, 2), (1, 0), (0, 1)]
        ]:
            gM[..., c, d] -= gIM[..., a, b] * M[..., e, f]
            gM[..., e, f] -= gIM[..., a, b] * M[..., c, d]

    get_rel_error(images_ft, images_cu, 'images')
    get_rel_error(opacities_ft, opacities_cu, 'opacities')
    get_rel_error(n_contrib_ft[0], n_contrib_cu[0], 'n_contrib')
    if out_extra_ft is not None:
        get_rel_error(n_contrib_ft[1], n_contrib_cu[1], 'median_contributor')
        # show_max_different(n_contrib_ft[1],n_contrib_cu[1])
        get_rel_error(out_extra_ft[0], out_extra_cu[0], 'depth')
        get_rel_error(out_extra_ft[1:4], out_extra_cu[1:4], 'normal')
        get_rel_error(out_extra_ft[4], out_extra_cu[4], 'mid depth')
        # show_max_different(out_extra_ft[4], out_extra_cu[4])
        get_rel_error(out_extra_ft[5], out_extra_cu[5], 'distortion')
        # show_max_different(out_extra_ft[5], out_extra_cu[5])
        get_rel_error(out_extra_ft[6], out_extra_cu[6], 'M1')
        get_rel_error(out_extra_ft[7], out_extra_cu[7], 'M2')
        # show_max_different(out_extra_ft, out_extra_cu, dim=0)
    if out_flow_cu is not None:
        get_rel_error(out_flow_ft, out_flow_cu, 'flow')
        # show_max_different(out_flow_ft, out_flow_cu, dim=-1)
    g_images = torch.randn_like(images_ft)
    g_opactiy = torch.randn_like(opacities_ft)
    g_others = torch.randn_like(out_extra_ft)
    g_flow = torch.randn_like(out_flow_cu)
    # g_others[:6] = 0
    g_others[5:] = 0  # TODO: backward for distortion, M1, M2
    grads = [g_images, g_opactiy, g_others, g_flow]
    torch.autograd.backward([images_ft, opacities_ft, out_extra_ft, out_flow_ft], grads
                            )
    torch.autograd.backward([images_cu, opacities_cu, out_extra_cu, out_flow_cu], grads)
    get_rel_error(means2D_ft.grad, means2D_cu.grad, "grad means2D")
    # show_max_different(means2D_ft.grad, means2D_cu.grad, dim=-1)
    get_rel_error(colors_ft.grad, colors_cu.grad, "grad colors")
    # show_max_different(colors_ft.grad, colors_cu.grad, dim=-1)
    get_rel_error(normal_opacity_ft.grad[..., :3], normal_opacity_cu.grad[..., :3], "grad normal")
    get_rel_error(normal_opacity_ft.grad[..., 3:], normal_opacity_cu.grad[..., 3:], "grad opacity")
    _get_grad_M(trans_mat_ft, inverse_m_ft.grad, trans_mat_ft.grad)
    get_rel_error(trans_mat_ft.grad, trans_mat_cu.grad, "grad trans mat")

    get_run_speed(
        (means2D_ft, colors_ft, normal_opacity_ft, trans_mat_ft, inverse_m_ft,), grads,
        cu_func=lambda a, b, c, d, e: cu_func(W, H, a, b, c, d, tile_range, point_list, near_n, far_n,
                                              only_image, trans_mat_t2, None, None, None)[:4],
        fast=lambda a, b, c, d, e: cu_fast(W, H, a, b, c, d, e, tile_range, point_list, near_n,
                                           far_n, only_image, trans_mat_t2, None, None, None)[:4],
    )

    # fast v1    time: forward 0.097 ms, backward 0.750 ms, total: 0.847 ms
    # cuda       time: forward 0.092 ms, backward 0.678 ms, total: 0.770 ms
    # fast v2    time: forward 0.243 ms, backward 0.287 ms, total: 0.530 ms
    # cuda       time: forward 0.100 ms, backward 0.718 ms, total: 0.818 ms


def test_preprocess():
    from utils.test_utils import get_rel_error, clone_tensors, show_max_different
    from fast_2d_gs.renderer.gs_2d_render import GS_2D_preprocess
    utils.set_printoptions(6)
    print()
    device = torch.device('cuda')
    (W, H, sh_degree, means3D, scales, rotations, opacity, shs, Tw2v, Tv2c, campos, Tv2s, FoV) = \
        _gen_random_data(102400)
    cu_func = GS_2D_preprocess
    cu_fast = _fast_preprocess.apply

    means3D_cu, scales_cu, rotations_cu, opacity_cu, shs_cu = clone_tensors(
        means3D, scales, rotations, opacity, shs, device=device)
    colors_cu, means2D_cu, trans_mat_cu, normal_opacity_cu, depths_cu, radii_cu, tiles_touched = cu_func(
        W, H, sh_degree, False,
        means3D_cu, scales_cu, rotations_cu, opacity_cu, shs_cu, None,
        Tw2v, Tv2c, campos, trans_precomp=None, means2D=None,
    )

    means3D_ft, scales_ft, rotations_ft, opacity_ft, shs_ft = clone_tensors(
        means3D, scales, rotations, opacity, shs, device=device)
    (colors_ft, means2D_ft, trans_mat_ft, inverse_m_ft, normal_opacity_ft, depths_ft, radii_ft, tile_range,
     point_list) = cu_fast(
        W, H, sh_degree, False,
        means3D_ft, scales_ft, rotations_ft, opacity_ft, shs_ft, None,
        Tw2v, Tv2c, Tv2s, campos, None, None, None
    )
    get_rel_error(colors_ft, colors_cu, 'colors')
    get_rel_error(means2D_ft, means2D_cu, 'means')
    get_rel_error(trans_mat_ft, trans_mat_cu, 'trans_mat')
    get_rel_error(normal_opacity_ft, normal_opacity_cu, 'normal')
    get_rel_error(depths_ft, depths_cu, 'depths')
    get_rel_error(radii_ft.float(), radii_cu.float(), 'radii')

    grad_colors = torch.randn_like(colors_cu)
    grad_means2D = torch.randn_like(means2D_cu)
    grad_mat = torch.randn_like(trans_mat_cu)
    grad_no = torch.randn_like(normal_opacity_cu)
    torch.autograd.backward([colors_cu, means2D_cu, trans_mat_cu, normal_opacity_cu],
                            [grad_colors, grad_means2D.clone(), grad_mat, grad_no])
    torch.autograd.backward([colors_ft, means2D_ft, trans_mat_ft, normal_opacity_ft],
                            [grad_colors, grad_means2D.clone(), grad_mat, grad_no])
    get_rel_error(means3D_ft.grad, means3D_cu.grad, 'means grad')
    # show_max_different(means3D_ft.grad, means3D_cu.grad, dim=-1)
    get_rel_error(scales_ft.grad, scales_cu.grad, 'scales grad')
    get_rel_error(rotations_ft.grad, rotations_cu.grad, 'rotations grad')
    get_rel_error(opacity_ft.grad, opacity_cu.grad, 'opacity grad')
    get_rel_error(shs_ft.grad, shs_cu.grad, 'shs grad')


def test():
    from utils.test_utils import get_rel_error, clone_tensors, show_max_different, get_run_speed
    from fast_2d_gs.renderer.gs_2d_render_origin import render_2d_gs_offical
    from fast_2d_gs.renderer.gs_2d_render import GS_2D_render
    utils.set_printoptions(6)
    print()
    device = torch.device('cuda')
    (W, H, sh_degree, means3D, scales, rotations, opacit, shs, Tw2v, Tv2c, campos, Tv2s, FoV) = \
        _gen_random_data(102400, 5)
    near_n, far_n = 0.2, 100.
    only_image = False
    idx = 83583
    print(f'W, H = {W}, {H}')
    print(f'means3D = torch.tensor([{means3D[idx].tolist()}], device=device)')
    print(f'scales = torch.tensor([{scales[idx].tolist()}], device=device)')
    print(f'rotations = torch.tensor([{rotations[idx].tolist()}], device=device)')
    print(f'opacity = torch.tensor([{opacit[idx].tolist()}], device=device)')
    print(f'shs = torch.tensor({shs[idx].tolist()}, device=device)')
    print(f'Tw2v = torch.tensor({Tw2v.tolist()}, device=device)')
    print(f'Tv2c = torch.tensor({Tv2c.tolist()}, device=device)')
    print(f'campos = torch.tensor({campos.tolist()}, device=device)')
    print(f'Tv2s = torch.tensor({Tv2s.tolist()}, device=device)')

    means3D_my, scales_my, rotations_my, opacit_my, shs_my = clone_tensors(
        means3D, scales, rotations, opacit, shs, device=device)
    output_my = GS_2D_render(means3D_my, scales_my, rotations_my, opacit_my, sh_features=shs_my,
                             Tw2v=Tw2v, Tv2c=Tv2c, campos=campos, size=(W, H), sh_degree=sh_degree)

    means3D_of, scales_of, rotations_of, opacit_of, shs_of = clone_tensors(
        means3D, scales, rotations, opacit, shs, device=device)
    output_of = render_2d_gs_offical(means3D_of, opacit_of, scales_of, rotations_of, shs_of, Tw2v=Tw2v, Tv2c=Tv2c,
                                     campos=campos, size=(W, H), sh_degree=sh_degree, FoV=FoV)

    means3D_fs, scales_fs, rotations_fs, opacit_fs, shs_fs = clone_tensors(
        means3D, scales, rotations, opacit, shs, device=device)
    output_fs = GS_2D_fast_render(means3D_fs, scales_fs, rotations_fs, opacit_fs, sh_features=shs_fs,
                                  Tw2v=Tw2v, Tv2c=Tv2c, Tv2s=Tv2s, campos=campos, size=(W, H), sh_degree=sh_degree)

    print(utils.show_shape(output_my))
    print(utils.show_shape(output_of))
    print(utils.show_shape(output_fs))
    for k, v in output_fs.items():
        if k == 'viewspace_points':
            continue
        # get_rel_error(v.float(), output_of[k].float(), k)
        if v is not None and k in output_my and output_my[k] is not None:
            get_rel_error(v.float(), output_my[k].float(), f"{k}{list(v.shape)}")
        else:
            print('\033[41m', k, '\033[0m', utils.show_shape(v, output_my.get(k, None)))
    show_max_different(output_fs['images'], output_my['images'], dim=0)
    show_max_different(output_fs['radii'], output_my['radii'])
    # names = ['images', 'normals', 'surf_depth', 'surf_normal']
    names = ['images']
    grads = [torch.randn_like(output_my[name]) for name in names]
    torch.autograd.backward([output_my[name] for name in names], grads)
    torch.autograd.backward([output_fs[name] for name in names], grads)
    get_rel_error(means3D_fs.grad, means3D_my.grad, "grad means3D")
    show_max_different(means3D_fs.grad, means3D_my.grad, dim=1)
    get_rel_error(scales_fs.grad, scales_my.grad, "grad scales")
    show_max_different(scales_fs.grad, scales_my.grad, dim=1)
    get_rel_error(rotations_fs.grad, rotations_my.grad, "grad rotations")
    if sh_degree >= 0:
        get_rel_error(shs_fs.grad, shs_my.grad, "grad shs")
    get_rel_error(opacit_fs.grad, opacit_my.grad, "grad opacity")
    get_rel_error(output_fs['viewspace_points'].grad, output_my['viewspace_points'].grad[:, :2], "means2D")
    show_max_different(output_fs['viewspace_points'].grad, output_my['viewspace_points'].grad[:, :2], dim=1)

    def my_func(*args, **kwargs):
        output_my = GS_2D_render(*args, **kwargs)
        return [output_my[name] for name in names]

    get_run_speed(
        (means3D_my, scales_my, rotations_my, opacit_my,
         (W, H), Tw2v, Tv2c, campos, None, FoV, False, 0.2, 100, sh_degree,
         None, shs_my, None,),
        grads, my_func=my_func
    )

    def fast_func(*args, **kwargs):
        output_my = GS_2D_fast_render(*args, **kwargs)
        return [output_my[name] for name in names]

    get_run_speed(
        (means3D_fs, scales_fs, rotations_fs, opacit_fs,
         (W, H), Tw2v, Tv2c, Tv2s, campos, None, FoV, False, 0.2, 100, sh_degree,
         None, shs_fs, None,),
        grads, fast_func=fast_func
    )
