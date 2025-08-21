import math
from typing import Optional, Tuple, Union, Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from fast_2d_gs._C import get_C_function, try_use_C_extension, get_python_function
import utils
from utils import ops_3d
from fast_2d_gs.renderer.gaussian_render import GS_prepare
from utils.test_utils import get_run_speed

_BLOCK_X = 16
_BLOCK_Y = 16
FilterInvSquare = 2.0

__all__ = ['GS_2D_render', 'RasterizeBuffer', 'GS_2D_topk_weights']


class RasterizeBuffer(NamedTuple):
    W: int
    """output image width"""
    H: int
    """output image width"""
    P: int
    """number of gaussian"""
    R: int
    """number of rendered"""
    near: float
    far: float
    tile_ranges: Tensor
    point_list: Tensor
    n_contribed: Tensor
    means2D: Tensor
    normal_opacity: Tensor
    trans_mat: Tensor


class _GS_2D_compute_trans_mat(torch.autograd.Function):
    _forward_func = get_C_function('gs_2d_compute_transmat_forward')
    _backward_func = get_C_function('gs_2d_compute_trans_mat_backward')

    @staticmethod
    def forward(ctx, *inputs):
        means3D, scales, rotations, Tw2v, Tv2c, W, H = inputs
        Tw2c = (Tv2c @ Tw2v).contiguous()
        matrix, normals = _GS_2D_compute_trans_mat._forward_func(W, H, means3D, scales, rotations, Tw2v, Tw2c)
        ctx._size = (W, H)
        ctx.save_for_backward(means3D, scales, rotations, Tw2v, Tw2c)
        return matrix, normals

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_matrix, grad_normals = grad_outputs
        W, H = ctx._size
        means3D, scales, rotations, Tw2v, Tw2c = ctx.saved_tensors
        grad_means3D, grad_scales, grad_rotations = _GS_2D_compute_trans_mat._backward_func(
            W, H, means3D, scales, rotations, Tw2v, Tw2c, grad_matrix, grad_normals
        )
        return grad_means3D, grad_scales, grad_rotations, None, None, None, None


def GS_2D_compute_trans_mat(
        means3D: Tensor, scales: Tensor, rotations: Tensor, Tw2v: Tensor, Tv2c: Tensor, W: int, H: int
):
    L = quat_toR(rotations)
    normal = ops_3d.apply(L[:, :, 2], Tw2v[:3, :3])
    L = torch.stack([L[:, :, 0] * scales[:, 0:1], L[:, :, 1] * scales[:, 1:2], means3D], dim=-1)
    L = torch.cat([L, L.new_tensor([0, 0, 1]).expand(L.shape[0], 1, 3)], dim=1)
    ndc2pix = Tw2v.new_tensor([[0.5 * W, 0, 0, 0.5 * (W - 1)], [0, 0.5 * H, 0, 0.5 * (H - 1)], [0, 0, 0, 1.]])
    trans_mat = ndc2pix @ ((Tv2c @ Tw2v) @ L)
    return trans_mat, normal


class _GS_2D_preprocess(torch.autograd.Function):
    _forward_func = get_C_function('gs_2d_preprocess_forward')
    _backward_func = get_C_function('gs_2d_preprocess_backward')

    @staticmethod
    def forward(ctx, *inputs):
        (
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, campos, trans_precomp, means2D, culling
        ) = inputs
        means2D, colors, trans_mat, normal_opacity, depths, radii, tiles_touched = _GS_2D_preprocess._forward_func(
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, campos,
            trans_precomp, means2D, culling
        )
        ctx.save_for_backward(
            means3D, means2D, scales, rotations, opacities, shs_or_colors, shs_rest, Tw2v, Tv2c, campos, trans_mat,
            radii,
            colors, normal_opacity
        )
        ctx._info = (W, H, sh_degree, is_opengl)
        return colors, means2D, trans_mat, normal_opacity, depths, radii, tiles_touched

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        (means3D, means2D, scales, rotations, opacities, shs_or_colors, shs_rest, Tw2v, Tv2c, campos, trans_mat, radii,
         colors, normal_opacity) = ctx.saved_tensors
        W, H, sh_degree, is_opengl = ctx._info
        grad_colors, grad_means2D, grad_mat, grad_normal_opacity, _, _, _ = grad_outputs
        grad_inputs = [
            None, None, None, None,
            torch.zeros_like(means3D), torch.zeros_like(scales), torch.zeros_like(rotations),
            torch.zeros_like(opacities),
            torch.zeros_like(shs_or_colors) if shs_or_colors is not None and ctx.needs_input_grad[8] else None,
            torch.zeros_like(shs_rest) if shs_rest is not None and ctx.needs_input_grad[9] else None,
            None, None, None, None, None, None, None
        ]
        _GS_2D_preprocess._backward_func(
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, campos,
            trans_mat, radii, colors, normal_opacity,
            grad_means2D, grad_mat, None, grad_colors, grad_normal_opacity,
            grad_inputs[4], grad_inputs[5], grad_inputs[6], grad_inputs[7], grad_inputs[8], grad_inputs[9],
            grad_inputs[10], grad_inputs[11]
        )
        return tuple(grad_inputs)


def quat_toR(q: Tensor):
    """将四元数标准化并得到旋转矩阵"""
    x, y, z, w = q.unbind(-1)
    # yapf: disable
    R = torch.stack([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * w * y + 2 * x * z,
        2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x,
        2 * x * z - 2 * w * y, 2 * w * x + 2 * y * z, 1 - 2 * x * x - 2 * y * y,
    ], dim=-1).reshape(*x.shape, 3, 3)
    # yapf: enable
    return R


def GS_2D_preprocess(
        W, H, sh_degree, is_opengl,
        means3D, scales, rotations, opacities, shs_or_colors, shs_features_rest,
        Tw2v, Tv2c, campos,
        trans_precomp=None, means2D: Tensor = None, culling: Tensor = None,
):
    return _GS_2D_preprocess.apply(W, H, sh_degree, is_opengl,
                                   means3D, scales, rotations, opacities, shs_or_colors, shs_features_rest,
                                   Tw2v, Tv2c, campos, trans_precomp, means2D, culling)


def _GS_2D_preprocess_py(
        W: int, H: int, sh_degree: int, is_opengl: bool,
        means3D: Tensor, scales: Tensor, rotations: Tensor, opacities: Tensor,
        shs_or_colors: Tensor, shs_features_rest: Optional[Tensor],
        Tw2v: Tensor, Tv2c: Tensor, campos: Tensor,
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

    trans_mat, normal = GS_2D_compute_trans_mat(means3D, scales, rotations, Tw2v, Tv2c, W, H)

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
    return means2D, colors, trans_mat, normal_opacity, depths, radii, tiles_touched


class _GS_2D_rasterize(torch.autograd.Function):
    _forward_func = get_C_function('GS_2D_render_forward')
    _backward_func = get_C_function('GS_2D_render_backward')

    @staticmethod
    def forward(ctx, *inputs):
        (W, H, means2D, colors, normal_opacity, trans_mat, tile_range, point_list, near_n, far_n, only_image,
         trans_mat_t2, accum_max_count, accum_weights_p, accum_weights_count,) = inputs
        pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib = _GS_2D_rasterize._forward_func(
            W, H, means2D, colors, normal_opacity, trans_mat, point_list, tile_range, near_n, far_n, only_image,
            trans_mat_t2, accum_max_count, accum_weights_p, accum_weights_count,
        )
        ctx._info = (near_n, far_n)
        ctx.save_for_backward(means2D, normal_opacity, trans_mat, colors, trans_mat_t2, tile_range, point_list,
                              n_contrib, pixel_extras, pixel_opacity, pixel_flow)
        return pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib

    @staticmethod
    def backward(ctx, *grad_outputs):
        (means2D, normal_opacity, trans_mat, colors, trans_mat_t2, tile_range, point_list,
         n_contrib, pixel_extras, pixel_opacity, pixel_flow) = ctx.saved_tensors
        grad_images, grad_opacity, grad_others, grad_flow, _ = grad_outputs
        grad_colors, grad_means2D, grad_trans, grad_normal_opacity = _GS_2D_rasterize._backward_func(
            means2D, normal_opacity, colors, trans_mat, trans_mat_t2,
            pixel_opacity, pixel_extras, pixel_flow,
            grad_images, grad_opacity, grad_others, grad_flow,
            tile_range, point_list, n_contrib,
            *ctx._info
        )
        grad_inputs = [None] * 21
        grad_inputs[2] = grad_means2D
        grad_inputs[3] = grad_colors
        grad_inputs[4] = grad_normal_opacity
        grad_inputs[5] = grad_trans
        return tuple(grad_inputs)


@try_use_C_extension(_GS_2D_rasterize.apply, "GS_2D_render_forward", "GS_2D_render_backward")
def GS_2D_rasterize(
        W: int, H: int, mean2D: Tensor, colors: Tensor, normal_opacity: Tensor, trans_mat: Tensor,
        tile_range: Tensor, point_list: Tensor,
        near_n=0.2, far_n=100., only_image=False,
        mat_t2=None, accum_max_count=None, accum_weights_p=None, accum_weights_count=None,

):
    grid_x = (W + _BLOCK_X - 1) // _BLOCK_X
    grid_y = (H + _BLOCK_Y - 1) // _BLOCK_Y
    images = mean2D.new_zeros(colors.shape[-1], H, W)
    n_contrib = normal_opacity.new_zeros(2, H, W, dtype=torch.int)
    opacities = normal_opacity.new_zeros(H, W)
    out_extra = mean2D.new_zeros(9, H, W)
    out_flow = mean2D.new_zeros(H, W, 2) if mat_t2 is not None else None

    pix_id = 761
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
            depth = s[:, :, 0] * gs_Tw[:, 0] + s[:, :, 1] * gs_Tw[:, 1] + gs_Tw[:, 2]
            # depth = torch.where(rho3d <= rho2d, depth, gs_Tw[:, 2])
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
            out_extra[8, sy:ey, sx:ex] = sigma.sum(dim=-1).reshape(H_, W_)
            if mat_t2 is not None:
                u_t2 = mat_t2[index, 0, 0] * s[..., 0] + mat_t2[index, 0, 1] * s[..., 1] + mat_t2[index, 0, 2]
                v_t2 = mat_t2[index, 1, 0] * s[..., 0] + mat_t2[index, 1, 1] * s[..., 1] + mat_t2[index, 1, 2]
                w_t2 = mat_t2[index, 2, 0] * s[..., 0] + mat_t2[index, 2, 1] * s[..., 1] + mat_t2[index, 2, 2]
                # w_t2_ = torch.where(abs(w_t2) > 1e-5, w_t2, torch.full_like(w_t2, 1e-5))
                # w_t2 = w_t2 + (w_t2_ - w_t2).detach()
                flow = torch.stack([u_t2 / w_t2, v_t2 / w_t2], dim=-1) - xy[:, None, :]
                out_flow[sy:ey, sx:ex] = torch.einsum('pi,pij->pj', sigma, flow).reshape(H_, W_, 2)
                out_flow[sy:ey, sx:ex] /= sigma.sum(dim=-1).reshape(H_, W_, 1)  # .detach()

            # if torch.any(index == 3968):q
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
    return images, opacities, None if only_image else out_extra, out_flow, n_contrib


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


class _GS_2D_Render(torch.autograd.Function):
    _preprocess_forward = get_C_function('gs_2d_preprocess_forward')
    _preprocess_backward = get_C_function('gs_2d_preprocess_backward')
    _render_forward = get_C_function('GS_2D_render_forward')
    _render_backward = get_C_function('GS_2D_render_backward')

    @staticmethod
    def forward(ctx, means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
                Tw2v, Tv2c, campos,
                size, sh_degree, is_opengl, near_n, far_n, only_image,
                trans_precomp, means2D,
                transe_mat_t2, culling, accum_max_count, accum_weights_p, accum_weights_count
                ):
        W, H = size
        means2D, colors, trans_mat, normal_opacity, depths, radii, tiles_touched = _GS_2D_Render._preprocess_forward(
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, campos,
            trans_precomp, means2D,
            culling
        )
        tile_range, point_list = GS_prepare(W, H, means2D, depths, radii, tiles_touched, False)
        pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib = _GS_2D_Render._render_forward(
            W, H, means2D, colors, normal_opacity, trans_mat, point_list, tile_range, near_n, far_n, only_image,
            transe_mat_t2, accum_max_count, accum_weights_p, accum_weights_count
        )
        ctx._info = (W, H, sh_degree, is_opengl, near_n, far_n)
        ctx.save_for_backward(
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, campos,
            means2D, trans_mat, transe_mat_t2, radii, colors, normal_opacity, tile_range, point_list,
            pixel_extras, pixel_opacity, pixel_flow, n_contrib
        )
        buffer = RasterizeBuffer(int(W), int(H), means3D.shape[0], point_list.shape[0], near_n, far_n,
                                 tile_range, point_list, n_contrib, means2D, normal_opacity, trans_mat)
        return pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib, radii, buffer

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        (
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, campos,
            means2D, trans_mat, transe_mat_t2, radii, colors, normal_opacity, tile_range, point_list,
            pixel_extras, pixel_opacity, pixel_flow, n_contrib
        ) = ctx.saved_tensors
        W, H, sh_degree, is_opengl, near_n, far_n = ctx._info
        grad_images, grad_opacity, grad_others, grad_flows = grad_outputs[:4]
        grad_colors, grad_means2D, grad_trans, grad_normal_opacity = _GS_2D_Render._render_backward(
            means2D, normal_opacity, colors, trans_mat, transe_mat_t2,
            pixel_opacity, pixel_extras, pixel_flow,
            grad_images, grad_opacity, grad_others, grad_flows,
            tile_range, point_list, n_contrib,
            near_n, far_n
        )
        grad_inputs = [
            torch.zeros_like(means3D), torch.zeros_like(scales), torch.zeros_like(rotations),
            torch.zeros_like(opacities),
            torch.zeros_like(shs_or_colors) if ctx.needs_input_grad[4] else None,
            torch.zeros_like(shs_rest) if shs_rest is not None and ctx.needs_input_grad[5] else None,
            None, None, None,
            None, None, None, None, None, None,
            None, grad_means2D,
            None, None, None, None, None,
        ]
        _GS_2D_Render._preprocess_backward(
            W, H, sh_degree, is_opengl,
            means3D, scales, rotations, opacities, shs_or_colors, shs_rest,
            Tw2v, Tv2c, campos,
            trans_mat, radii, colors, normal_opacity,
            grad_means2D, grad_trans, None, grad_colors, grad_normal_opacity,
            grad_inputs[0], grad_inputs[1], grad_inputs[2], grad_inputs[3], grad_inputs[4], grad_inputs[5],
            grad_inputs[6], grad_inputs[8]
        )
        return tuple(grad_inputs)


def GS_2D_render(
        points: Tensor,
        scales: Tensor,
        rotations: Tensor,
        opacity: Tensor,
        size: Union[Tuple[int, int], Tensor],
        Tw2v: Tensor, Tv2c: Tensor,
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
        trans_mat_t2: Tensor = None,
        accum_max_count: Tensor = None,
        accum_weights_p: Tensor = None,
        accum_weights_count: Tensor = None,
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
    images, opacities, out_extras, flow, n_contrib, radii, buffer = _GS_2D_Render.apply(
        points, scales, rotations, opacity, sh_features if colors is None else colors, sh_features_rest,
        Tw2v.contiguous(), Tv2c.contiguous(), campos.contiguous(),
        size, sh_degree if colors is None else -1, is_opengl, near, far, False,
        None, means2D,
        trans_mat_t2, culling, accum_max_count, accum_weights_p, accum_weights_count,
    )
    depths, normals, median_depths, distortion, _ = out_extras.split([1, 3, 1, 1, 3], dim=0)
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
        'depths': median_depths,
        'normals': normals,
        'opacity': opacities,
        'distortion': distortion,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'radii': radii,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        'buffer': buffer,
    }
    if flow is not None:
        outputs['flow'] = flow
    return outputs


def GS_2D_topk_weights(topk: int, buffer: RasterizeBuffer) -> Tuple[Tensor, Tensor]:
    """return topk indices and weights """
    return get_C_function('gs_2d_topk_weights')(
        topk, buffer.W, buffer.H, buffer.near, buffer.far, buffer.means2D,
        buffer.normal_opacity, buffer.trans_mat, buffer.tile_ranges, buffer.point_list
    )


def _gen_random_data(P=1024, seed: int = None, W=511, H=127):
    seed = np.random.randint(0, int(1e9)) if seed is None else seed
    print(f'{seed=}')
    torch.manual_seed(seed)

    device = torch.device('cuda')
    sh_degree = 0
    means3D = torch.randn(P, 3).cuda()
    scales = torch.rand(P, 2).cuda()
    rotations = ops_3d.quaternion.normalize(torch.randn(P, 4).cuda())
    opacit = torch.randn(P, 1, device=device)  # [0, 1]
    colors = None  # torch.randn(P, 3, device=device)  # [0, 1]
    shs = torch.randn(P, (1 + sh_degree) ** 2, 3, device=device)

    Tw2v = ops_3d.opencv.look_at(torch.tensor([0, 0, 4.]), torch.zeros(3)).cuda()
    campos = torch.inverse(Tw2v)[:3, 3]
    focal = 500
    # W, H = 511, 127
    fov_x, fov_y = ops_3d.focal_to_fov(focal, W, H)
    tan_fovx, tan_fovy = np.tan(0.5 * fov_x).item(), np.tan(0.5 * fov_y).item()
    Tv2c = ops_3d.opencv.perspective(fov_y, size=(W, H)).to(Tw2v)
    Tw2c = Tv2c @ Tw2v
    Tv2s = ops_3d.opencv.camera_intrinsics(size=(W, H), fov=torch.tensor([fov_x, fov_y])).to(device)
    focals = torch.tensor([focal, focal], device=device)
    return W, H, sh_degree, means3D, scales, rotations, opacit, shs, Tw2v, Tv2c, campos, focals


def test_compute_trans_mat():
    from utils.test_utils import get_rel_error, clone_tensors
    print()
    utils.set_printoptions(6)
    (W, H, sh_degree, means3D, scales, rotations, opacit, shs, Tw2v, Tv2c, campos, focals) = _gen_random_data(1024)

    py_func = GS_2D_compute_trans_mat
    cu_func = _GS_2D_compute_trans_mat.apply

    means3D_py, scales_py, rotations_py = clone_tensors(means3D, scales, rotations, device='cuda')
    mat_py, normal_py = py_func(means3D_py, scales_py, rotations_py, Tw2v, Tv2c, W, H)

    means3D_cu, scales_cu, rotations_cu = clone_tensors(means3D, scales, rotations, device='cuda')
    mat_cu, normal_cu = cu_func(means3D_cu, scales_cu, rotations_cu, Tw2v, Tv2c, W, H)

    get_rel_error(mat_cu, mat_py, "matrix")
    get_rel_error(normal_cu, normal_py, "normal")

    g_mat = torch.randn_like(mat_py)
    g_normal = torch.randn_like(normal_py)
    torch.autograd.backward([mat_cu, normal_cu], [g_mat, g_normal])
    torch.autograd.backward([mat_py, normal_py], [g_mat, g_normal])
    get_rel_error(means3D_cu.grad, means3D_py.grad, "grad_means3d")
    get_rel_error(scales_cu.grad, scales_py.grad, "grad_scale")
    get_rel_error(rotations_cu.grad, rotations_py.grad, "grad_rotation")

    get_run_speed((means3D_cu, scales_cu, rotations_cu, Tw2v, Tv2c, W, H), (g_mat, g_normal),
                  cu_func=cu_func, py_func=py_func)


def test_preprocess():
    from utils.test_utils import get_rel_error, clone_tensors
    utils.set_printoptions(6)
    print()
    device = torch.device('cuda')
    P = 10240
    (W, H, sh_degree, means3D, scales, rotations, opacit, shs, Tw2v, Tv2c, campos, focals) = _gen_random_data(P)

    py_func = _GS_2D_preprocess_py
    cu_func = GS_2D_preprocess
    means2D = means3D.new_zeros(P, 2)

    means_py, scales_py, rotations_py, opacities_py, shs_py, means2D_py = clone_tensors(
        means3D, scales, rotations, opacit, shs, means2D, device=device)
    means2D_py_, colors_py, trans_mat_py, normal_opacity_py, depths_py, radii_py, tiles_touched_py = py_func(
        W, H, sh_degree, False,
        means_py, scales_py, rotations_py, opacities_py, shs_py, None,
        Tw2v, Tv2c, campos, trans_precomp=None, means2D=means2D_py,
    )

    means_cu, scales_cu, rotations_cu, opacities_cu, shs_cu, means2D_cu = clone_tensors(
        means3D, scales, rotations, opacit, shs, means2D, device=device)
    means2D_cu_, colors_cu, trans_mat_cu, normal_opacity_cu, depths_cu, radii_cu, tiles_touched_cu = cu_func(
        W, H, sh_degree, False,
        means_cu, scales_cu, rotations_cu, opacities_cu, shs_cu, None,
        Tw2v, Tv2c, campos, trans_precomp=None, means2D=means2D_cu,
    )

    # index = (means2D_cu - means2D_py).abs().argmax() // means2D_py.shape[-1]
    # print(f'{index = }')
    # print(means2D_cu[index], means2D_py[index])
    get_rel_error(means2D_cu, means2D_py, "means2D")
    get_rel_error(means2D_cu_, means2D_py_, "means2D_")
    get_rel_error(colors_cu, colors_py, "colors")
    get_rel_error(trans_mat_cu, trans_mat_py, "trans_mat")
    get_rel_error(depths_cu, depths_py, "depths")
    get_rel_error(radii_cu, radii_py, "radii")
    get_rel_error(tiles_touched_cu, tiles_touched_py, "tiles_touched")

    grad_means2D = torch.randn_like(means2D_py)
    grad_colors = torch.randn_like(colors_py)
    grad_trans_mat = torch.randn_like(trans_mat_py)
    grad_normal = torch.randn_like(normal_opacity_cu)
    torch.autograd.backward([means2D_cu, colors_cu, trans_mat_cu, normal_opacity_cu],
                            [grad_means2D.clone(), grad_colors, grad_trans_mat, grad_normal])
    torch.autograd.backward([means2D_py, colors_py, trans_mat_py, normal_opacity_py],
                            [grad_means2D.clone(), grad_colors, grad_trans_mat, grad_normal])

    get_rel_error(means_cu.grad, means_py.grad, "grad means")
    get_rel_error(scales_cu.grad, scales_py.grad, "grad scale")
    get_rel_error(rotations_cu.grad, rotations_py.grad, "grad rotation")
    get_rel_error(opacities_cu.grad, opacities_py.grad, "grad opacities")
    get_rel_error(shs_cu.grad, shs_py.grad, "grad shs")
    get_rel_error(means2D_cu.grad, means2D_py.grad, "grad means2D")
    index = (means2D_cu.grad - means2D_py.grad).abs().argmax() // means2D_py.grad.shape[-1]
    print(f'{index = }')
    print(means2D_cu.grad[index], means2D_py.grad[index])

    get_run_speed(
        inputs=(W, H, sh_degree, False,
                means_cu, scales_cu, rotations_cu, opacities_cu, shs_cu, None,
                Tw2v, Tv2c, campos, None, means2D_cu),
        grads=(grad_means2D.clone(), grad_colors, grad_trans_mat, grad_normal, None, None, None),
        py_func=py_func, cu_func=cu_func
    )


def test_rasterize():
    from utils.test_utils import get_rel_error, clone_tensors, show_max_different
    utils.set_printoptions(6)
    print()
    device = torch.device('cuda')
    (W, H, sh_degree, means3D, scales, rotations, opacit, shs, Tw2v, Tv2c, campos, focals) = \
        _gen_random_data(1024)
    near_n, far_n = 0.2, 100.
    only_image = False
    means2D, colors, trans_mat, normal_opacity, depths, radii, tiles_touched = _GS_2D_preprocess_py(
        W, H, sh_degree, False,
        means3D, scales, rotations, opacit, shs, None,
        Tw2v, Tv2c, campos, trans_precomp=None, means2D=None,
    )
    tile_range, point_list = GS_prepare(W, H, means2D, depths, radii, tiles_touched, False)
    trans_mat_t2 = trans_mat + torch.randn_like(trans_mat) * 1e-3

    py_func = get_python_function('GS_2D_rasterize')
    cu_func = _GS_2D_rasterize.apply

    means2D_py, colors_py, normal_opacity_py, trans_mat_py = clone_tensors(
        means2D, colors, normal_opacity, trans_mat, device=device)
    images_py, opacities_py, out_extra_py, out_flow_py, n_contrib_py = py_func(
        W, H, means2D_py, colors_py, normal_opacity_py, trans_mat_py, tile_range, point_list, near_n, far_n,
        only_image, trans_mat_t2, None, None, None)

    means2D_cu, colors_cu, normal_opacity_cu, trans_mat_cu = clone_tensors(
        means2D, colors, normal_opacity, trans_mat, device=device)
    images_cu, opacities_cu, out_extra_cu, out_flow_cu, n_contrib_cu = cu_func(
        W, H, means2D_cu, colors_cu, normal_opacity_cu, trans_mat_cu, tile_range, point_list, near_n, far_n,
        only_image, trans_mat_t2, None, None, None)

    print(utils.show_shape(images_py, opacities_py, out_extra_py, n_contrib_py))
    print(utils.show_shape(images_cu, opacities_cu, out_extra_cu, n_contrib_cu))
    get_rel_error(images_cu, images_py, 'images')
    get_rel_error(opacities_cu, opacities_py, 'opacities')
    get_rel_error(n_contrib_cu[0], n_contrib_py[0], 'n_contrib')
    if out_extra_py is not None:
        get_rel_error(n_contrib_cu[1], n_contrib_py[1], 'median_contributor')
        # show_max_different(n_contrib_cu[1], n_contrib_py[1])
        get_rel_error(out_extra_cu[0], out_extra_py[0], 'depth')
        get_rel_error(out_extra_cu[1:4], out_extra_py[1:4], 'normal')
        get_rel_error(out_extra_cu[4], out_extra_py[4], 'mid depth')
        # show_max_different(out_extra_cu[4], out_extra_py[4])
        get_rel_error(out_extra_cu[5], out_extra_py[5], 'distortion')
        # show_max_different(out_extra_cu[5], out_extra_py[5])
        get_rel_error(out_extra_cu[6], out_extra_py[6], 'M1')
        get_rel_error(out_extra_cu[7], out_extra_py[7], 'M2')
        get_rel_error(out_extra_cu[8], out_extra_py[8], 'weight sum')
        # show_max_different(out_extra_cu, out_extra_py, dim=0)
    if out_flow_py is not None:
        get_rel_error(out_flow_cu, out_flow_py, 'flow')
        show_max_different(out_flow_cu, out_flow_py, dim=-1)

    g_images = torch.randn_like(images_py) * 0
    g_flow = torch.randn_like(out_flow_py)
    g_opacity = torch.randn_like(opacities_py) * 0
    torch.autograd.backward([images_py, out_flow_py, opacities_py], [g_images, g_flow, g_opacity])
    torch.autograd.backward([images_cu, out_flow_cu, opacities_cu], [g_images, g_flow, g_opacity])
    get_rel_error(means2D_cu.grad, means2D_py.grad, "grad means2D")
    get_rel_error(colors_cu.grad, colors_py.grad, "grad colors")
    get_rel_error(normal_opacity_cu.grad, normal_opacity_py.grad, "grad normal")
    get_rel_error(trans_mat_cu.grad, trans_mat_py.grad, "grad trans mat")
    show_max_different(trans_mat_cu.grad, trans_mat_py.grad, dim=-1)


def test():
    from utils.test_utils import get_rel_error, clone_tensors, show_max_different
    from fast_2d_gs.renderer.gs_2d_render_origin import render_2d_gs_offical
    utils.set_printoptions(6)
    print()
    device = torch.device('cuda')
    (W, H, sh_degree, means3D, scales, rotations, opacit, shs, Tw2v, Tv2c, campos, focals) = _gen_random_data(10240)
    near_n, far_n = 0.2, 100.
    only_image = False

    means3D_my, scales_my, rotations_my, opacit_my, shs_my = clone_tensors(
        means3D, scales, rotations, opacit, shs, device=device)
    output_my = GS_2D_render(means3D_my, scales_my, rotations_my, opacit_my, sh_features=shs_my,
                             Tw2v=Tw2v, Tv2c=Tv2c, campos=campos, size=(W, H), sh_degree=sh_degree)

    means3D_of, scales_of, rotations_of, opacit_of, shs_of = clone_tensors(
        means3D, scales, rotations, opacit, shs, device=device)
    output_of = render_2d_gs_offical(means3D_of, opacit_of, scales_of, rotations_of, shs_of, Tw2v=Tw2v, Tv2c=Tv2c,
                                     campos=campos, size=(W, H), focals=focals, sh_degree=sh_degree)

    print(utils.show_shape(output_my))
    print(utils.show_shape(output_of))
    for k, v in output_my.items():
        if k == 'viewspace_points':
            continue
        if v is not None and k in output_of and output_of[k] is not None:
            get_rel_error(v.float(), output_of[k].float(), k)
        else:
            print(k, utils.show_shape(v, output_of.get(k, None)))

    names = ['images', 'normals', 'surf_depth', 'surf_normal']
    # names = ['images']
    grads = [torch.randn_like(output_my[name]) for name in names]
    torch.autograd.backward([output_my[name] for name in names], grads)
    torch.autograd.backward([output_of[name] for name in names], grads)
    get_rel_error(means3D_my.grad, means3D_of.grad, "grad means3D")
    show_max_different(means3D_my.grad, means3D_of.grad)
    get_rel_error(scales_my.grad, scales_of.grad, "grad scales")
    show_max_different(scales_my.grad, scales_of.grad, dim=1)
    get_rel_error(rotations_my.grad, rotations_of.grad, "grad rotations")
    get_rel_error(shs_my.grad, shs_of.grad, "grad shs")
    get_rel_error(opacit_my.grad, opacit_of.grad, "grad opacity")
    get_rel_error(output_my['viewspace_points'].grad, output_my['viewspace_points'].grad[:, :2], "means2D")


def test_show_2d_gs():
    import matplotlib.pyplot as plt
    from lietorch import SO3
    from utils.test_utils import get_abs_error
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import time
    utils.set_printoptions(6)
    seed = int(time.time() * 1000)
    # seed = 1730509694
    print(f'{seed = }')
    torch.manual_seed(seed)

    plt.figure(figsize=(20, 10))
    ax = plt.subplot(121, projection='3d')
    N = 100
    u = torch.linspace(-1, 1, N)
    v = torch.linspace(-1, 1, N)
    u, v = torch.meshgrid(u, v)
    # u, v = u.reshape(-1), v.reshape(-1)
    print(u.shape, v.shape)
    c = torch.exp(-(u ** 2 + v ** 2) * 0.5).numpy()
    # ax.scatter(u, v, c, c=c)
    su = torch.rand(1) * 0.4 + 0.1
    sv = torch.rand(1) * 0.4 + 0.1
    q = ops_3d.quaternion.normalize(torch.randn(4))
    R = ops_3d.quaternion.toR(q)
    tu, tv, tw = R.unbind(-1)
    mu = torch.rand(3) - 0.5
    for x, color in zip([tu, tv, tw], 'rgb'):
        ax.plot((mu[0], mu[0] + x[0]), (mu[1], mu[1] + x[1]), (mu[2], mu[2] + x[2]), c=color)
    # ax.set_aspect('equal')
    ax_len = 1.0
    ax.set_xlim(-ax_len, ax_len)
    ax.set_ylim(-ax_len, ax_len)
    ax.set_zlim(-ax_len, ax_len)

    su2 = torch.rand(1) * 0.4 + 0.1
    sv2 = torch.rand(1) * 0.4 + 0.1
    q2 = ops_3d.quaternion.normalize(torch.randn(4))
    R2 = ops_3d.quaternion.toR(q2)
    tu2, tv2, tw2 = R2.unbind(-1)
    mu2 = torch.rand(3) - 0.5
    for x, color in zip([tu2, tv2, tw2], 'rgb'):
        ax.plot((mu2[0], mu2[0] + x[0]), (mu2[1], mu2[1] + x[1]), (mu2[2], mu2[2] + x[2]), c=color)

    p = mu + su * tu * u[..., None] + sv * tv * v[..., None]
    p2 = mu2 + su2 * tu2 * u[..., None] + sv2 * tv2 * v[..., None]
    print(utils.show_shape(su, tu, u, sv, tv, v))
    colors_map = plt.get_cmap('jet_r')
    cNorm = colors.Normalize(vmin=np.min(c), vmax=np.max(c))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colors_map)
    facecolors = scalarMap.to_rgba(c)
    ax.plot_surface(p[..., 0], p[..., 1], p[..., 2], facecolors=facecolors, alpha=0.5, cmap=colors_map)
    ax.plot_surface(p2[..., 0], p2[..., 1], p2[..., 2], facecolors=facecolors, alpha=0.5, cmap=colors_map)
    ax.view_init(elev=90, azim=-90)
    FoV = np.pi * 0.5
    ax.set_proj_type('persp', 1 / np.tan(FoV * 0.5))
    # print('Axes3D proj matrix', ax.get_proj())
    ax.plot((0, 3), (0, 0), (0, 0), c='r')
    ax.plot((0, 0), (0, 3), (0, 0), c='g')
    ax.plot((0, 0), (0, 0), (0, 3), c='b')
    plt.subplot(222)
    ops_3d.set_coord_system('opencv')
    eye = torch.tensor([0, 0, 1.2])
    at = torch.tensor([0, 0, 0])
    Tw2v = ops_3d.look_at(eye, at)
    size = 256
    Tv2c = ops_3d.perspective(FoV)
    # print(Tv2c)
    Tv2s = ops_3d.camera_intrinsics(size=(size, size), fov=FoV)
    pixels = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size)), dim=-1)
    print(utils.show_shape(mu, su, sv, q, Tw2v, Tv2c))
    M_, normal_ = GS_2D_compute_trans_mat(mu[None], torch.as_tensor([[su, sv, 0.]]),
                                          q[None], Tw2v, Tv2c, size, size)
    print('M_gt:', M_)

    H = torch.zeros((4, 4))
    H[:3, 0] = su * tu
    H[:3, 1] = sv * tv
    H[:3, 3] = mu
    H[3, 3] = 1
    Tv2s_ = torch.zeros((4, 4))
    Tv2s_[:2, :2] = Tv2s[:2, :2]
    Tv2s_[:3, 3] = Tv2s[:3, 2]

    ndc2pix = Tv2s.new_tensor(
        [[0.5 * size, 0, 0, 0.5 * (size - 1)], [0, 0.5 * size, 0, 0.5 * (size - 1)], [0, 0, 1, 0.], [0, 0, 0, 1.]])
    print(Tv2s, Tv2s_, utils.show_shape(Tv2s_, Tw2v, H))
    M = ndc2pix @ Tv2c @ Tw2v @ H
    print('M:', M)
    hx = torch.zeros((*pixels.shape[:-1], 4))
    hx[..., 0] = -1
    hx[..., -1] = pixels[..., 0]
    hy = torch.zeros((*pixels.shape[:-1], 4))
    hy[..., 1] = -1
    hy[..., -1] = pixels[..., 1]
    hu = hx @ M
    hv = hy @ M
    # hu = pixels[..., 0:1] * M[3, :] - M[0, :]
    # hv = pixels[..., 1:2] * M[3, :] - M[1, :]
    # print(utils.show_shape(hu, hv))
    # print(hu - hu_)
    u_ = (hu[..., 1] * hv[..., 3] - hu[..., 3] * hv[..., 1]) / (hu[..., 0] * hv[..., 1] - hu[..., 1] * hv[..., 0])
    v_ = (hu[..., 3] * hv[..., 0] - hu[..., 0] * hv[..., 3]) / (hu[..., 0] * hv[..., 1] - hu[..., 1] * hv[..., 0])
    d = u_ * M[3, 0] + v_ * M[3, 1] + M[3, 2]
    print('u_, v_, d', utils.show_shape(u_, v_, d))
    # p1_ = ops_3d.pixel2points(d, Tv2s=Tv2s, Tw2v=Tw2v, pixel=pixels)
    p1_ = mu + su * tu * u_[..., None] + sv * tv * v_[..., None]
    g = np.exp(-0.5 * (u_ ** 2 + v_ ** 2))
    c1 = scalarMap.to_rgba(g)
    mask = g > 0.005
    x_min, x_max = [x.item() for x in mask.any(dim=0).nonzero()[:, 0].aminmax()]
    y_min, y_max = [x.item() for x in mask.any(dim=1).nonzero()[:, 0].aminmax()]
    print('p_1, c1, H', utils.show_shape(p1_, c1, H), x_min, x_max, y_min, y_max)
    p1_ = p1_[y_min:y_max, x_min:x_max]
    ax.plot_surface(p1_[..., 0], p1_[..., 1], p1_[..., 2], facecolors=c1[y_min:y_max, x_min:x_max], alpha=0.1,
                    cmap=colors_map)
    # p1_ = ops_3d.point2pixel(torch.from_numpy(p), Tw2v=Tw2v, Tv2s=Tv2s)[0]
    # p2_ = ops_3d.point2pixel(torch.from_numpy(p2), Tw2v=Tw2v, Tv2s=Tv2s)[0]
    # print(p1_.min(), p1_.max(), p2_.min(), p2_.max())
    plt.subplot(222)
    plt.imshow(c1)
    plt.subplot(224)
    H = np.zeros((4, 4))
    H[:3, 0] = su2 * tu2
    H[:3, 1] = sv2 * tv2
    H[:3, 3] = mu2
    H[3, 3] = 1
    M = ndc2pix @ Tv2c @ Tw2v @ H
    hu = pixels[..., 0:1] * M[3, :] - M[0, :]
    hv = pixels[..., 1:2] * M[3, :] - M[1, :]
    u_ = (hu[..., 1] * hv[..., 3] - hu[..., 3] * hv[..., 1]) / (hu[..., 0] * hv[..., 1] - hu[..., 1] * hv[..., 0])
    v_ = (hu[..., 3] * hv[..., 0] - hu[..., 0] * hv[..., 3]) / (hu[..., 0] * hv[..., 1] - hu[..., 1] * hv[..., 0])
    print(utils.show_shape(u_, v_))
    plt.imshow(scalarMap.to_rgba(np.exp(-0.5 * (u_ ** 2 + v_ ** 2))))
    # plt.scatter(p1_[..., 0].reshape(-1), p1_[..., 1].reshape(-1), c=c)
    # plt.scatter(p2_[..., 0].reshape(-1), p2_[..., 1].reshape(-1), c=c)
    plt.tight_layout()
    plt.show()


def test_flow():
    from utils.test_utils import get_rel_error, clone_tensors, show_max_different
    from fast_2d_gs.renderer.gs_2d_render_origin import render_2d_gs_offical
    import cv2
    utils.set_printoptions(6)
    print()
    device = torch.device('cuda')
    seed = 42  # np.random.randint(0, int(1e9))
    print(f'{seed=}')
    torch.manual_seed(seed)

    P = 1
    sh_degree = 0
    means3D = torch.randn(P, 3).cuda()
    scales = torch.rand(P, 2).cuda()
    rotations = ops_3d.quaternion.normalize(torch.randn(P, 4).cuda())
    opacit = torch.full((P, 1), 1., device=device)  # [0, 1]
    colors = torch.ones((P, 3), device=device)
    colors[:, 1:] = 0

    Tw2v = ops_3d.opencv.look_at(torch.tensor([0, 0, 4.]), torch.zeros(3)).cuda()
    campos = torch.inverse(Tw2v)[:3, 3]
    focal = 500
    W, H = 512, 512
    fov_x, fov_y = ops_3d.focal_to_fov(focal, W, H)
    tan_fovx, tan_fovy = np.tan(0.5 * fov_x).item(), np.tan(0.5 * fov_y).item()
    Tv2c = ops_3d.opencv.perspective(fov_y, size=(W, H)).to(Tw2v)
    Tw2c = Tv2c @ Tw2v
    Tv2s = ops_3d.opencv.camera_intrinsics(size=(W, H), fov=torch.tensor([fov_x, fov_y])).to(device)
    focals = torch.tensor([focal, focal], device=device)
    output_t1 = GS_2D_render(means3D, scales, rotations, opacit, colors=colors,
                             Tw2v=Tw2v, Tv2c=Tv2c, campos=campos, size=(W, H), sh_degree=sh_degree)
    img_t1 = utils.as_np_image(output_t1['images'])
    trans_mat_t1 = output_t1['buffer'].trans_mat
    R = ops_3d.rotate(x=torch.pi * 0.1).to(Tw2v.device)
    Tw2v_t2 = R @ Tw2v
    means3D_t2 = means3D + torch.randn_like(means3D) * 0.01
    scales_t2 = scales + scales.new_tensor([-0.1, 0.1])
    colors[:, :] = colors.new_tensor([0, 1., 0])
    output_t2 = GS_2D_render(means3D_t2, scales_t2, rotations, opacit, colors=colors, trans_mat_t2=trans_mat_t1,
                             Tw2v=Tw2v_t2, Tv2c=Tv2c, campos=campos, size=(W, H), sh_degree=sh_degree)
    trans_mat_t2 = output_t2['buffer'].trans_mat
    img_t2 = utils.as_np_image(output_t2['images'])
    print(utils.show_shape(output_t2))
    flow = output_t2['flow'].detach().cpu().numpy()
    img_flow = utils.flow_to_image(flow)
    print(utils.show_shape(img_flow))

    pixels = torch.meshgrid(torch.arange(0, W, device=device), torch.arange(0, H, device=device), indexing='xy')
    pixels = torch.stack(pixels, dim=- 1)
    print(pixels.shape)
    k = pixels[..., 0:1] * trans_mat_t2[0, 2] - trans_mat_t2[0, 0]
    l = pixels[..., 1:2] * trans_mat_t2[0, 2] - trans_mat_t2[0, 1]
    p = torch.cross(k, l, dim=-1)
    uv = p[..., :2] / p[..., 2:]
    power = uv.square().sum(-1)
    alpha = opacit * torch.exp(-0.5 * power)
    get_rel_error(alpha, output_t2['images'][1], 'images')

    pixels_t1 = trans_mat_t1[0, :, 0] * uv[..., 0:1] + trans_mat_t1[0, :, 1] * uv[..., 1:2] + trans_mat_t1[0, :, 2]
    pixels_t1 = pixels_t1[..., :2] / pixels_t1[..., 2:]

    img_t2_ = np.zeros_like(img_t2)
    img_t2_[..., 1] = (alpha * 255).clamp(0, 255).cpu().numpy()
    # return
    images = np.concatenate((img_t2, img_t1, img_flow), axis=1)  # ,(img_t2_ != img_t2) * 255
    # x, y = 444, 60
    # plt.imshow(images)
    # plt.scatter(x, y, c='r', marker='+')
    # print(flow[y, x], pixels_t1[y, x])
    # plt.scatter(x + W + flow[y, x, 0], y + flow[y, x, 1], c='r', marker='+')
    # plt.scatter(pixels_t1[y, x, 0].item() + W, pixels_t1[y, x, 1].item(), c='b', marker='x')
    # plt.show()
    # return

    WIN_NAME = 'Debug'
    images_ = images.copy()
    if images_.shape[0] == 3:
        images_ = np.transpose(images, (1, 2, 0))

    def onmouse_pick_points(event, x, y, flags, param: np.ndarray):
        if event == cv2.EVENT_LBUTTONDOWN:
            param[:] = images_
            if x >= W or y >= H:
                return
            x2, y2 = int(x + W + flow[y, x, 0]), int(y + flow[y, x, 1])
            print(f"{x=}, {y=}, dx={flow[y, x, 0]}, dy={flow[y, x, 1]},"
                  f" color diff={param[y, x, 1] * 1.0 - param[y2, x2, 0]}")
            cv2.drawMarker(param, (x, y), (0, 255, 0))
            cv2.drawMarker(param, (x + W * 2, y), (255, 0, 0))
            if W <= x2 < W * 2 and 0 <= y2 < H:
                cv2.drawMarker(param, (x2, y2), (0, 255, 0))
            # print(f"dx2 = {flow2[y, x, 0]}, dy2 = {flow2[y, x, 1]}", end=' ')
            # x3, y3 = int(x + W + flow2[y, x, 0]), int(y + flow2[y, x, 1])
            # if W <= x3 < W * 2 and 0 <= y3 < H:
            #     cv2.drawMarker(param, (x3, y3), (255, 0, 0))
            # print('')
        return

    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, images)
    while True:
        cv2.imshow(WIN_NAME, images)
        key = cv2.waitKey(30)
        if key == ord('q') or key == 27:
            exit()
    # plt.imshow(np.concatenate((img_t1, img_t2, img_flow), axis=1))
    # plt.show()


if __name__ == '__main__':
    test_flow()
