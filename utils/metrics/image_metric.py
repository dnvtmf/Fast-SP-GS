from functools import partial
from typing import Union, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.distributed import all_reduce
import torch.nn.functional as F

from utils.my_logger import logger
from .base import Metric, METRICS

from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.functional.image.ssim import multiscale_structural_similarity_index_measure

__all__ = ['ImageMetric']


@METRICS.register('image')
class ImageMetric(Metric):
    compute_SSIM = structural_similarity_index_measure
    compute_MS_SSIM = multiscale_structural_similarity_index_measure

    def __init__(self, items=None, device=None, **kwargs) -> None:
        self.device = torch.device('cuda') if device is None else device
        self._psnr_sum = torch.zeros(1, device=self.device)
        self._m_psnr_sum = torch.zeros(1, device=self.device)
        self._ssim_sum = torch.zeros(1, device=self.device)
        self._m_ssim_sum = torch.zeros(1, device=self.device)
        self._ms_ssim_sum = torch.zeros(1, device=self.device)
        self._lpips_alex_sum = torch.zeros(1, device=self.device)
        self._m_lpips_alex_sum = torch.zeros(1, device=self.device)
        self._lpips_vgg_sum = torch.zeros(1, device=self.device)
        self._m_lpips_vgg_sum = torch.zeros(1, device=self.device)
        self.num_images = torch.zeros(1, device=self.device, dtype=torch.long)

        if items is not None:
            items = [item.upper() for item in items]
            assert all(item in ['PSNR', 'SSIM', 'MS_SSIM', 'LPIPS', 'LPIPS_ALEX', 'LPIPS_VGG',
                                'MPSNR', 'MSSIM', 'MLPIPS', 'MLPIPS_ALEX', 'MLPIPS_VGG'] for item in items)
        else:
            items = []

        if not items or 'PSNR' in items:
            self.psnr_f = ImageMetric.compute_PSNR
            self._names['PSNR'] = None

        self.m_psnr_f = None
        if 'MPSNR' in items:
            self._names['mPSNR'] = None
            self.m_psnr_f = ImageMetric.compute_mPSNR

        if not items or 'SSIM' in items:
            self.ssim_f = ImageMetric.compute_SSIM
            self._names['SSIM'] = None
        else:
            self.ssim_f = None

        self.m_ssim_f = None
        if 'MSSIM' in items:
            self._names['mSSIM'] = None
            self.m_ssim_f = ImageMetric.compute_mSSIM

        self.ms_ssim_f = None
        if 'MS_SSIM' in items:
            self.ms_ssim_f = ImageMetric.compute_MS_SSIM
            self._names['MS_SSIM'] = None

        if 'LPIPS' in items or 'LPIPS_ALEX' in items:
            from .lpipsPyTorch import lpips, LPIPS
            model = LPIPS(net_type='alex').to(self.device)
            self.lpips_f = partial(lpips, criterion=model)
            self._names['LPIPS'] = None
        else:
            self.lpips_f = None

        self.m_lpips_f = None
        if 'MLPIPS' in items or 'MLPIPS_ALEX' in items:
            import lpips
            model = lpips.LPIPS(net="alex", spatial=True).to(self.device)
            self.m_lpips_f = partial(ImageMetric.compute_mLPIPS, model=model)
            self._names['mLPIPS'] = None

        self.lpips_vgg_f = None
        if 'LPIPS_VGG' in items:
            from .lpipsPyTorch import lpips, LPIPS
            model = LPIPS(net_type='vgg').to(self.device)
            self.lpips_vgg_f = partial(lpips, criterion=model, net_type='vgg')
            self._names['LPIPS_VGG'] = None

        self.m_lpips_vgg_f = None
        if 'MLPIPS_VGG' in items:
            import lpips
            model = lpips.LPIPS(net="vgg", spatial=True).to(self.device)
            self.m_lpips_vgg_f = partial(ImageMetric.compute_mLPIPS, model=model)
            self._names['mLPIPS_VGG'] = None

    def reset(self):
        self.num_images = 0
        self._psnr_sum.zero_()
        self._ssim_sum.zero_()
        self._ms_ssim_sum.zero_()
        self._lpips_alex_sum.zero_()
        self._lpips_vgg_sum.zero_()
        self._m_psnr_sum.zero_()
        self._m_ssim_sum.zero_()
        self._m_lpips_alex_sum.zero_()
        self._m_lpips_vgg_sum.zero_()

    def prepare_input(self, image: Union[np.ndarray, Tensor]):
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        image = image.view(-1, *image.shape[-3:])
        image = image.to(self.device)
        if image.dtype == torch.uint8:
            image = image / 255.  # range in [0., 1]
        if image.ndim == 3:
            image = image[None]
        if image.shape[-1] == 3:  # [B, H, W, 3] --> [B, 3, H, W]
            image = image.moveaxis(-1, 1)
        assert image.ndim == 4 and image.shape[1] == 3
        return image

    @torch.no_grad()
    def update(self, images: Tensor, gt: Tensor, mask: Tensor = None):
        images = self.prepare_input(images)  # shape: [B, 3, H, W]
        gt = self.prepare_input(gt)
        N = len(images)
        self.num_images += N

        def f(fn, *args):
            if fn is None:
                return 0
            else:
                return fn(*args).mean() * N

        self._psnr_sum += f(self.psnr_f, images, gt)
        self._ssim_sum += f(self.ssim_f, images, gt)
        self._ms_ssim_sum += f(self.ms_ssim_f, images, gt)
        self._lpips_alex_sum += f(self.lpips_f, images, gt)
        self._lpips_vgg_sum += f(self.lpips_vgg_f, images, gt)
        if mask is not None:
            mask = mask.squeeze(-1).to(self.device)  # shape: [B, 1, H, W]
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None]

            self._m_psnr_sum += f(self.m_psnr_f, images, gt, mask)
            self._m_ssim_sum += f(self.m_ssim_f, images, gt, mask)
            self._m_lpips_alex_sum += f(self.m_lpips_f, images, gt, mask)
            self._m_lpips_vgg_sum += f(self.m_lpips_vgg_f, images, gt, mask)

    def PSNR(self):
        return (self._psnr_sum / self.num_images).item()

    def mPSNR(self):
        return (self._m_psnr_sum / self.num_images).item()

    def SSIM(self):
        if self.ssim_f is None:
            logger.error('Please install package `torchmetrics` for SSIM metric')

        return (self._ssim_sum / self.num_images).item()

    def mSSIM(self):
        return (self._m_ssim_sum / self.num_images).item()

    def MS_SSIM(self):
        if self.ssim_f is None:
            logger.error('Please install package `torchmetrics` for MS_SSIM metric')
        return (self._ms_ssim_sum / self.num_images).item()

    def LPIPS(self):
        return (self._lpips_alex_sum / self.num_images).item()

    def mLPIPS(self):
        return (self._m_lpips_alex_sum / self.num_images).item()

    def LPIPS_ALEX(self):
        return self.LPIPS()

    def mLPIPS_ALEX(self):
        return self.mLPIPS()

    def LPIPS_VGG(self):
        return (self._lpips_vgg_sum / self.num_images).item()

    def mLPIPS_VGG(self):
        return (self._m_lpips_vgg_sum / self.num_images).item()

    def __repr__(self) -> str:
        s = []
        if 'PSNR' in self.names:
            s.append('PSNR')
        if 'SSIM' in self.names:
            s.append('SSIM')
        if 'MS_SSIM' in self.names:
            s.append('MS_SSIM')
        if 'LPIPS' in self.names or 'LPIPS_ALEX' in self.names:
            s.append(f'LPIPS[ALEX]')
        if 'LPIPS_VGG' in self.names:
            s.append(f'LPIPS[VGG]')
        if 'mPSNR' in self.names:
            s.append('mPSNR')
        if 'mSSIM' in self.names:
            s.append('mSSIM')
        if 'mLPIPS' in self.names or 'mLPIPS_ALEX' in self.names:
            s.append(f'mLPIPS[ALEX]')
        if 'mLPIPS_VGG' in self.names:
            s.append(f'mLPIPS[VGG]')
        return f"{self.__class__.__name__}: [{', '.join(s)}]"

    @staticmethod
    def compute_PSNR(images_a: Union[Tensor, Sequence[Tensor]], images_b: Union[Tensor, Sequence[Tensor]]):
        # simplified since max_pixel_value is 1 here.
        # from torchmetrics.functional.image import peak_signal_noise_ratio
        # self.psnr_f = peak_signal_noise_ratio
        if isinstance(images_a, Tensor):
            images_a = images_a.flatten()
        else:
            images_a = torch.cat([img.flatten() for img in images_a])
        if isinstance(images_b, Tensor):
            images_b = images_b.flatten()
        else:
            images_b = torch.cat([img.flatten() for img in images_b])
        return -10 * torch.log10(torch.mean((images_a - images_b) ** 2))

    @staticmethod
    def compute_mPSNR(images_a: Tensor, images_b: Tensor, mask: Tensor):
        diff = (images_a - images_b) ** 2 * mask
        num_mask = mask.sum().clamp_min(1e-6) * diff.numel() // mask.numel()
        return -10 * torch.log10(diff.sum() / num_mask)

    @staticmethod
    def compute_mSSIM(
            img0: Tensor, img1: Tensor, mask: Tensor,
            max_val: float = 1.0,
            filter_size: int = 11,
            filter_sigma: float = 1.5,
            k1: float = 0.01,
            k2: float = 0.03,
    ):
        hw = filter_size // 2
        shift = (2 * hw - filter_size + 1) / 2
        f_i = ((torch.arange(filter_size, device=img0.device) - hw + shift) / filter_sigma) ** 2
        filt = torch.exp(-0.5 * f_i)
        filt /= torch.sum(filt)
        filt = filt[None, None].repeat_interleave(3, 0)
        one = torch.ones_like(filt[:1])

        # Blur in x and y (faster than the 2D convolution).
        def convolve2d(z, m, f, mf):
            z_ = F.conv2d(z * m, f, padding='valid', groups=3)
            m_ = F.conv2d(m, mf, padding='valid')
            return torch.where(m_ != 0, z_ * mf.numel() / m_, 0), (m_ != 0).to(z.dtype)

        def filt_fn(z, m):
            z, m = convolve2d(z, m, filt[:, :, None, :], one[:, :, None, :])
            z, m = convolve2d(z, m, filt[:, :, :, None], one[:, :, :, None])
            return z, m

        mu0 = filt_fn(img0, mask)[0]
        mu1 = filt_fn(img1, mask)[0]
        mu00 = mu0 * mu0
        mu11 = mu1 * mu1
        mu01 = mu0 * mu1
        sigma00 = filt_fn(img0 ** 2, mask)[0] - mu00
        sigma11 = filt_fn(img1 ** 2, mask)[0] - mu11
        sigma01 = filt_fn(img0 * img1, mask)[0] - mu01

        # Clip the variances and covariances to valid values. Variance must be non-negative:
        sigma00 = sigma00.clamp_min(0)
        sigma11 = sigma11.clamp_min(0)
        sigma01 = torch.sign(sigma01) * torch.minimum(torch.sqrt(sigma00 * sigma11), torch.abs(sigma01))

        c1 = (k1 * max_val) ** 2
        c2 = (k2 * max_val) ** 2
        numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
        denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
        ssim_map = numer / denom
        ssim = ssim_map.mean()
        return ssim

    @staticmethod
    def compute_mLPIPS(img0, img1, mask, model=None):
        import lpips
        if model is None:
            model = lpips.LPIPS(net="alex", spatial=True).to(img0.device)
        x = model(img0 * mask, img1 * mask, normalize=True) * mask
        num_mask = mask.sum().clamp_min(1e-6) * x.numel() // mask.numel()
        return x.sum() / num_mask
