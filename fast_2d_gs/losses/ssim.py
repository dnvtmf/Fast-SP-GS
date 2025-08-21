from typing import Optional
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .build import LOSSES
from fast_2d_gs._C import get_C_function, try_use_C_extension, get_python_function


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class FusedSSIM(torch.autograd.Function):
    """based on https://github.com/rahul-goel/fused-ssim
    The window size is fixed at 11
    """
    _forward_func = get_C_function('fusedssim')
    _backward_func = get_C_function('fusedssim_backward')
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    @staticmethod
    def forward(ctx, img1, img2, window):
        ssim_map, d1, d2, d3 = FusedSSIM._forward_func(FusedSSIM.C1, FusedSSIM.C2, img1, img2, ctx.needs_input_grad[0])
        assert not ctx.needs_input_grad[1]
        ctx.save_for_backward(img1, img2, d1, d2, d3)
        return ssim_map

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        img1, img2, d1, d2, d3 = ctx.saved_tensors
        grad_img1 = FusedSSIM._backward_func(FusedSSIM.C1, FusedSSIM.C2, img1, img2, grad_outputs[0], d1, d2, d3)
        return grad_img1, None, None


@try_use_C_extension(FusedSSIM.apply, 'fusedssim', 'fusedssim_backward')
def _ssim(img1: Tensor, img2: Tensor, window: Optional[Tensor]) -> Tensor:
    window_size = window.shape[-1]
    channel = img1.shape[1]
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map


@LOSSES.register('SSIM')
class SSIM_Loss(nn.Module):
    window: Tensor

    def __init__(self, window_size=11, reduction='mean', **kwargs):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.reduction = reduction
        self.register_buffer('window', create_window(window_size, self.channel), persistent=False)

    def forward(self, img1: Tensor, img2: Tensor, mask: Tensor = None) -> Tensor:
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)
        if img1.shape[-1] == 3:
            img1 = torch.permute(img1, (0, 3, 1, 2))  # shape: [B, C, H, W]
        if img2.shape[-1] == 3:
            img2 = torch.permute(img2, (0, 3, 1, 2))
        ssim_map = _ssim(img1, img2, self.window.type_as(img1))

        # if mask is not None:
        #     ssim_map = ssim_map.mean(dim=1, keepdim=True) * mask  # [B, H, W]
        #     ssim_map = ssim_map.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)
        #     return 1.0 - ssim_map

        if self.reduction == 'mean':
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(dim=(1, 2, 3))


def test():
    from skimage.metrics import structural_similarity
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure
    from PIL import Image
    import numpy as np
    from utils.test_utils import get_rel_error, get_run_speed
    img1 = np.array(Image.open('/home/wan/data/NeRF/D_NeRF/hook/train/r_000.png').convert('RGB'))
    img2 = np.array(Image.open('/home/wan/data/NeRF/D_NeRF/hook/train/r_001.png').convert('RGB'))
    print(img1.shape, img2.shape)
    # plt.imshow(np.concatenate((img1, img2), axis=1))
    # plt.show()
    v1 = structural_similarity(img1, img2, win_size=11, gaussian_weights=True, full=True, channel_axis=-1)[0]
    img1 = torch.from_numpy(img1).permute(2, 0, 1)[None].float().cuda() / 255.
    img2 = torch.from_numpy(img2).permute(2, 0, 1)[None].float().cuda() / 255.
    print(img1.shape, img2.shape)
    v2 = structural_similarity_index_measure(img1, img2)

    window = create_window(11, 3).cuda()

    py_func = lambda x, y: get_python_function('_ssim')(x, y, window)
    img1_1 = img1.clone().requires_grad_()
    v3 = py_func(img1_1, img2)
    g = torch.randn_like(v3)
    v3.backward(g)
    g1 = img1_1.grad

    cu_func = lambda x, y: FusedSSIM.apply(x, y, None)
    img1_2 = img1.clone().requires_grad_()
    v4 = cu_func(img1_2, img2)
    v4.backward(g)
    g2 = img1_2.grad

    print(v1.item(), v2.item(), v3.mean().item(), v4.mean().item())
    get_rel_error(v3, v4, 'ssim')
    get_rel_error(g1, g2, 'grad')
    get_run_speed((img1_1, img2), g, py_func=py_func, cu_func=cu_func)
