import numpy as np

import torch
from torch import nn, Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.amp import custom_bwd, custom_fwd

from fast_2d_gs._C import get_C_function


class _freq_encoder(Function):
    _forward = get_C_function('freq_encode_forward')
    _backward = get_C_function('freq_encode_backward')

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')  # force float32 for better precision
    def forward(ctx, inputs, degree, include_input, scale):
        if not inputs.is_cuda:
            inputs = inputs.cuda()
        inputs = inputs.contiguous()
        outputs = _freq_encoder._forward(inputs, degree, include_input, scale)
        ctx._info = [degree, include_input, scale]
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad):
        grad = grad.contiguous()
        outputs, = ctx.saved_tensors
        degree, include_input, scale = ctx._info
        grad_inputs = _freq_encoder._backward(grad, outputs, degree, include_input, scale)
        return grad_inputs, None, None, None


freq_encode = _freq_encoder.apply


class FreqEncoder(nn.Module):

    def __init__(self, input_dim=3, degree=4, include_input=True, scale=1.):
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = input_dim * (2 * degree + int(include_input))
        self.include_input = include_input
        self.scale = torch.pi if scale == 'pi' else float(scale)

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, degree={self.degree}, output_dim={self.output_dim}, "
                f"include_input={self.include_input}")

    def forward(self, inputs, **kwargs):
        # inputs: [..., input_dim]
        # return: [..., output_dim]
        return freq_encode(inputs, self.degree, self.include_input, self.scale)


class FreqEncoder_torch(nn.Module):

    def __init__(
            self,
            input_dim,
            degree=4,  # number of frequency
            max_freq_log2=None,
            log_sampling=True,
            include_input=True,
            scale=1.,
            periodic_fns=(torch.sin, torch.cos)
    ):
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        max_freq_log2 = degree - 1 if max_freq_log2 is None else max_freq_log2
        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim
        if scale == 'pi':
            scale = torch.pi

        self.output_dim += self.input_dim * degree * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, max_freq_log2, degree) * scale
        else:
            self.freq_bands = torch.linspace(2 ** 0, 2 ** max_freq_log2, degree) * scale

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):
        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)

        return out

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"
