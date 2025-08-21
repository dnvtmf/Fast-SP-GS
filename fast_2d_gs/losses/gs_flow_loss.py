from typing import Optional

import numpy as np
import torch
from torch import Tensor

from fast_2d_gs._C import get_C_function, try_use_C_extension, get_python_function
import utils
from utils import ops_3d
from fast_2d_gs.renderer.gaussian_render import _BLOCK_Y, _BLOCK_X


def get_gs_flow_matrix(mean2d_t1, mean2d_t2, cov_t1, cov_t2, inv1=True, inv2=True, mask=None, eps=1e-6):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Optional[Tensor], float) -> Tensor
    """
    References: https://github.com/Zerg-Overmind/GaussianFlow
    """
    # FIXME: svd产生的U可能是旋转180度的结果
    shape = mean2d_t1.shape
    if mask is not None:
        mean2d_t1, mean2d_t2, cov_t1, cov_t2 = mean2d_t1[mask], mean2d_t2[mask], cov_t1[mask], cov_t2[mask]
    cov2D_inv_t1 = torch.stack(([cov_t1[..., 0], cov_t1[..., 1], cov_t1[..., 1], cov_t1[..., 2]]), dim=-1)
    cov2D_inv_t1 = cov2D_inv_t1.reshape(*cov_t1.shape[:-1], 2, 2)
    U1, s1, V1 = torch.svd(cov2D_inv_t1)

    cov2D_inv_t2 = torch.stack(([cov_t2[..., 0], cov_t2[..., 1], cov_t2[..., 1], cov_t2[..., 2]]), dim=-1)
    cov2D_inv_t2 = cov2D_inv_t2.reshape(*cov_t2.shape[:-1], 2, 2)
    U2, s2, V2 = torch.svd(cov2D_inv_t2)
    s1, s2 = s1.clamp_min(eps), s2.clamp_min(eps)  # to avoid nan

    if inv1:
        M1 = V1 @ torch.diag_embed(s1.sqrt()) @ U1.transpose(-1, -2)
    else:
        M1 = U1 @ torch.diag_embed(s1.rsqrt()) @ V1.transpose(-1, -2)
    if inv2:
        M2 = U2 @ torch.diag_embed(s2.rsqrt()) @ V2.transpose(-1, -2)
    else:
        M2 = V2 @ torch.diag_embed(s2.sqrt()) @ U2.transpose(-1, -2)
    M = M2 @ M1
    delta_t = mean2d_t2 - mean2d_t1 * 2  # torch.einsum('...i,...ji->...j', mean2d_t1, M)
    if mask is None:
        return torch.cat([M, delta_t[..., None]], dim=-1)
    else:
        results = torch.zeros(*shape, 3, device=M.device)
        results[mask] = torch.cat([M, delta_t[..., None]], dim=-1)
        return results


def decompose_covariance_matrix(cov2D: Tensor, inv=False, return_inv=False, mask: Tensor = None):
    if cov2D.shape[-2:] == (2, 2):
        a = cov2D[..., 0, 0]
        b = cov2D[..., 0, 1]
        c = cov2D[..., 1, 1]
    else:
        a, b, c = cov2D.unbind(-1)[:3]
    d = c - a
    x1 = (-d + (d ** 2 + 4 * b ** 2).sqrt()) * 0.5
    x2 = (-d - (d ** 2 + 4 * b ** 2).sqrt()) * 0.5
    if inv:
        s0_1, s1_1 = (x1 + c).clamp_min(1e-10).rsqrt(), (a - x1).clamp_min(1e-10).rsqrt()
        s0_2, s1_2 = (x2 + c).clamp_min(1e-10).rsqrt(), (a - x2).clamp_min(1e-10).rsqrt()
    else:
        s0_1, s1_1 = (x1 + c).clamp_min(1e-10).sqrt(), (a - x1).clamp_min(1e-10).sqrt()
        s0_2, s1_2 = (x2 + c).clamp_min(1e-10).sqrt(), (a - x2).clamp_min(1e-10).sqrt()
    if mask is not None:
        mask_ = mask == (s0_1 < s1_1)
        x = torch.where(mask_, x1, x2)
        s0 = torch.where(mask_, s0_1, s0_2)
        s1 = torch.where(mask_, s1_1, s1_2)
    else:
        x, s0, s1 = x1, s0_1, s1_1
        mask = s0 < s1
    theta = torch.atan2(b, x)
    cos = theta.cos()
    sin = theta.sin()

    # print(theta, s0, s1)
    if return_inv:
        s0, s1 = 1. / s0, 1. / s1
        Rs = torch.stack([cos * s0, sin * s0, -sin * s1, cos * s1], dim=-1).view(*a.shape, 2, 2)
    else:
        Rs = torch.stack([cos * s0, -sin * s1, sin * s0, cos * s1], dim=-1).view(*a.shape, 2, 2)
    return Rs, mask


def get_gs_flow_matrix_v2(mean2d_t1, mean2d_t2, cov_t1, cov_t2, inv1=True, inv2=True, mask=None):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Optional[Tensor]) -> Tensor
    """解析方法求解"""
    shape = mean2d_t1.shape
    if mask is not None:
        mean2d_t1, mean2d_t2, cov_t1, cov_t2 = mean2d_t1[mask], mean2d_t2[mask], cov_t1[mask], cov_t2[mask]
    M1, mask_ = decompose_covariance_matrix(cov_t1, inv1, True, mask=None)
    M2, mask_ = decompose_covariance_matrix(cov_t2, inv2, False, mask=mask_)
    M = M2 @ M1
    delta_t = mean2d_t2 - torch.einsum('...i,...ji->...j', mean2d_t1, M)
    if mask is None:
        return torch.cat([M, delta_t[..., None]], dim=-1)
    else:
        results = torch.zeros(*shape, 3, device=M.device)
        results[mask] = torch.cat([M, delta_t[..., None]], dim=-1)
        return results


class _gaussian_flow(torch.autograd.Function):
    _forward = get_C_function('gs_flow_forward')
    _backward = get_C_function('gs_flow_backward')

    @staticmethod
    def forward(ctx, *inputs):
        W, H, m_flow, mean2D, conic_opacity, tile_ranges, point_list, n_contrib, out_opacity = inputs
        flow = _gaussian_flow._forward(W, H, m_flow, mean2D, conic_opacity, tile_ranges, point_list, n_contrib)
        ctx.save_for_backward(m_flow, mean2D, conic_opacity, tile_ranges, point_list, n_contrib, out_opacity)
        ctx._info = (W, H)
        return flow

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_flow):
        m_flow, mean2D, conic_opacity, tile_ranges, point_list, n_contrib, out_opacity = ctx.saved_tensors
        W, H = ctx._info
        grad_means2D = None
        grad_conic = None
        grad_m_cov, grad_means2D, grad_conic = _gaussian_flow._backward(
            W, H, m_flow, out_opacity, grad_flow,
            mean2D, conic_opacity, tile_ranges, point_list, n_contrib,
            grad_means2D, grad_conic
        )
        return None, None, grad_m_cov, grad_means2D, grad_conic, None, None, None, None


@try_use_C_extension(_gaussian_flow.apply, 'gs_flow_forward', 'gs_flow_backward')
def GS_flow(
        W: int, H: int, m_flow: Tensor, mean2D: Tensor, conic_opacity: Tensor, tile_range: Tensor,
        point_list: Tensor, n_contrib: Tensor, out_opacity: Tensor
):
    grid_x = (W + _BLOCK_X - 1) // _BLOCK_X
    grid_y = (H + _BLOCK_Y - 1) // _BLOCK_Y
    flow = mean2D.new_zeros(H, W, 2)

    # pix_id = 222103
    # debug_x, debug_y = pix_id % W, pix_id // W
    # debug_tx, debug_ty = debug_x // _BLOCK_X, debug_y // _BLOCK_Y
    # debug_i = (debug_y % _BLOCK_Y) * _BLOCK_X + debug_x % _BLOCK_X
    # print(f"{debug_x=}, {debug_y=}")
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
            d_xy = gs_xy[None, :, :] - xy[:, None, :]
            power = (-0.5 * (gs_cov_inv_opactiy[:, 0] * d_xy[:, :, 0] * d_xy[:, :, 0]
                             + gs_cov_inv_opactiy[:, 2] * d_xy[:, :, 1] * d_xy[:, :, 1])
                     - gs_cov_inv_opactiy[:, 1] * d_xy[:, :, 0] * d_xy[:, :, 1])
            mask = power <= 0
            alpha = (gs_cov_inv_opactiy[:, 3] * power.exp()) * mask  # [T_W * T_H, P]
            alpha = alpha + (alpha.clamp_max(0.99) - alpha).detach()
            mask = mask & (alpha >= 1. / 255)
            alpha = torch.where(alpha < 1. / 255, torch.zeros_like(alpha), alpha)
            sigma = (1 - alpha).cumprod(dim=1)
            with torch.no_grad():
                mask = torch.logical_and(mask, sigma >= 0.0001)

            m_flow_i = m_flow[index]  # [P, 2, 3]
            flow_tile = torch.einsum('npi,pji->npj', d_xy, m_flow_i[:, :2, :2]) + m_flow_i[:, :2, 2]
            flow_tile = flow_tile + xy[:, None, :]  # d_xy
            sigma = torch.cat([torch.ones_like(sigma[:, :1]), sigma[:, :-1]], dim=-1) * alpha * mask
            flow[sy:ey, sx:ex] = torch.einsum('pi,pij->pj', sigma, flow_tile).reshape(H_, W_, -1)
            # if x == debug_tx and y == debug_ty:
            #     print('\033[33m')
            #     for i in range(len(index)):
            #         if mask[debug_i, i]:
            #             print(
            #                 f"gs={index[i].item():}, xy={gs_xy[i, 0]:.6f}, {gs_xy[i, 1]:.6f}, "
            #                 f"power={power[debug_i, i].item():.6f}, "
            #                 f"alpha={alpha[debug_i, i].item():.6f}, "
            #                 f"sigma={sigma[debug_i, i]:.6f}, "
            #                 f"flow_tile={flow_tile[debug_i, i]}"
            #             )
            #             print(m_flow_i[i].view(-1))
            #     print('\033[0m')
            #
            #     def _show(mask_, name=''):
            #         def show_grad(grad):
            #             print(grad.shape, debug_i, mask_.shape)
            #             print(f'{name} grad:', grad[debug_i][mask_[debug_i]])
            #
            #         return show_grad
            #
            #     flow_tile.register_hook(_show(mask, 'flow'))

    return flow
