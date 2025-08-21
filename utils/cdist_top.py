from typing import Callable, Tuple, Optional

import torch
from torch import Tensor

from fast_2d_gs._C import get_C_function, have_C_functions

__all__ = ['cdist_top', 'cdist_top_match_label']


def cdist_top_py(x1, x2, largest=False) -> Tuple[Tensor, Tensor]:
    """
    对x1中的每个向量，在x2中找到与它(欧式)距离最近(远)的向量

    Args:
        x1 (Tensor): shape [..., N, C]
        x2 (Tensor): shape [..., M, C]
        largest (bool):
    Returns:
        return the chamfer distance and the indeices of copprate points. shape [..., N]
    """
    assert x1.ndim == x2.ndim and x1.shape[:-2] == x2.shape[:-2]
    assert x1.shape[-1] == x2.shape[-1]
    shape = x1.shape
    if x1.ndim >= 3:
        x1 = x1.flatten(0, -3)
        x2 = x2.flatten(0, -3)
    else:
        assert x1.ndim == 2
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
    distance = torch.cdist(x1, x2)  # shape:B x N x M
    if largest:
        values, indices = distance.max(dim=-1)  # type: Tuple[Tensor, Tensor]
    else:
        values, indices = distance.min(dim=-1)  # type: Tuple[Tensor, Tensor]
    values = values.view(shape[:-1])
    indices = indices.view(shape[:-1])
    return values, indices


if have_C_functions('cdist_top', 'cdist_top_backward'):
    cdist_top_forward = get_C_function('cdist_top')
    cdist_top_backward = get_C_function('cdist_top_backward')


    class _ChamferDistanceFunction(torch.autograd.Function):
        @staticmethod
        def jvp(ctx, *grad_inputs):
            pass

        @staticmethod
        def forward(ctx, *inputs):
            if len(inputs) == 2:
                x1, x2 = inputs
                largest = False
            else:
                x1, x2, largest = inputs
            x1 = x1.contiguous()
            x2 = x2.contiguous()

            distance, index = cdist_top_forward(x1, x2, largest)
            ctx.save_for_backward(x1, x2, distance, index)
            return distance, index

        @staticmethod
        def backward(ctx, *grad_outputs):
            grad_dist = grad_outputs[0].contiguous()
            points1, points2, distance, index = ctx.saved_tensors
            grad_1, grad_2 = cdist_top_backward(points1, points2, distance, index, grad_dist)
            return grad_1, grad_2, None


    cdist_top = _ChamferDistanceFunction.apply  # type: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
else:
    cdist_top = cdist_top_py


def cdist_top_match_label(x1: Tensor, x2: Tensor, label1: Tensor, label2: Tensor, largest=False):
    max_d = max(x1.abs().max(), x2.abs().max()) * 2.  # >最大距离
    x1 = x1 + label1[..., None] * (max_d + 10)
    x2 = x2 + label2[..., None] * (max_d + 10)
    distance, index = cdist_top(x1, x2, largest)
    mask = distance.gt(max_d)
    distance = torch.masked_fill(distance, mask, -1)
    index = torch.masked_fill(index, mask, -1)
    return distance, index
