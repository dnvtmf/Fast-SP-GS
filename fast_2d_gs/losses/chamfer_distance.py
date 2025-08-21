import torch
from torch import nn, Tensor
from fast_2d_gs._C import get_C_function, try_use_C_extension, get_python_function
from .build import LOSSES


class _chamfer_distance(torch.autograd.Function):
    _forward_func = get_C_function('chamfer_distance_forward')
    _backward_func = get_C_function('chamfer_distance_backward')

    @staticmethod
    def forward(ctx, *inputs):
        x, y, sphere = inputs
        index1, index2, dist1, dist2 = _chamfer_distance._forward_func(x, y, sphere)
        ctx.sphere = sphere
        ctx.save_for_backward(x, y, index1, index2)
        return dist1, dist2, index1, index2

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, y, index1, index2, = ctx.saved_tensors
        grad1, grad2, _, _ = grad_outputs
        grad_x, grad_y = _chamfer_distance._backward_func(
            x, y, ctx.sphere, index1, index2, grad1.contiguous(), grad2.contiguous()
        )
        return grad_x, grad_y, None


@try_use_C_extension(_chamfer_distance.apply, 'chamfer_distance_forward', 'chamfer_distance_backward')
def chamfer_distance(x: Tensor, y: Tensor, sphere: bool):
    if sphere:
        a1 = x[:, None, 0]  # 极地角
        a2 = y[None, :, 0]
        b1 = x[:, None, 1]  # 方位角
        b2 = y[None, :, 1]
        t = a1.sin() * a2.sin() * (b1 - b2).cos() + a1.cos() * a2.cos()
        eps = 1e-6  # torch.finfo(t.dtype).eps
        t = t - (t - t.clamp(-1 + eps, 1 - eps)).detach()  # avoid nan
        # def show_grad(grad):
        #     print('nan t:', t[grad.isnan()].tolist())
        #
        # t.register_hook(show_grad)
        dist = torch.arccos(t)
    else:
        dist = torch.cdist(x, y).square()
    dist1, index1 = dist.min(dim=1)
    dist2, index2 = dist.min(dim=0)
    return dist1, dist2, index1, index2


@LOSSES.register('CD')
class ChamferDistance(nn.Module):
    def __init__(self, sphere=False, reduction='mean', **kwargs):
        super(ChamferDistance, self).__init__()
        self.sphere = sphere
        self.reduction = reduction

    def forward(self, x, y):
        dist1, dist2, index1, index2 = chamfer_distance(x.contiguous(), y.contiguous(), self.sphere)
        if self.reduction == 'mean':
            return (dist1.sum() + dist2.sum()) / (dist1.numel() + dist2.numel())
        elif self.reduction == 'sum':
            return dist1.sum() + dist2.sum()
        else:
            return dist1, dist2, index1, index2
