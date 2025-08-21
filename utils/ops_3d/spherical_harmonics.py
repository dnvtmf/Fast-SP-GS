"""spherical harmonic 球谐系数相关"""
import math
from typing import Optional

import torch
from torch import Tensor
from torch.amp import custom_fwd, custom_bwd
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from fast_2d_gs._C import get_C_function, try_use_C_extension, have_C_functions

__all__ = ['sh_encode', 'rotation_SH', 'SH_to_RGB', 'SH2RGB', 'RGB2SH']


# reference: https://wuli.wiki/online/RYlm.html 和 https://blog.csdn.net/tiao_god/article/details/111240808
def _SH_P(l: int, m: int, x: Tensor):
    """伴连带勒让德函数
    P^m_l(x) = (1-x^2)^{|m|/2} \frac{d^{m}P_l(x)}{dx^{|m|}}

    P_l(x) = 1/(2^l l!) \frac{d^l(x^2-1)^l }{dx^l}
    """
    pmm = 1.0  # P^0_0 = 1
    #  P^m_m = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    if m > 0:
        somx2 = ((1 - x) * (1 + x)).sqrt()
        fact = 1.
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0

    if l == m:
        return pmm if isinstance(pmm, Tensor) else torch.full_like(x, pmm)
    pmmp1 = x * (2. * m + 1.0) * pmm  # p^m_{m+1} = x(2m+1)P^m_m
    if l == m + 1:
        return pmmp1
    #  (l-m)P^m_l = x (2l-1) P^m_{l-1} - (l + m - 1) P^m_{l-2}
    pll = 0
    for ll in range(m + 2, l + 1):
        pll = ((2. * ll - 1.0) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll  # if isinstance(pmm, Tensor) else torch.full_like(x, pll)


def _SH_K(d: int, m: int):
    return math.sqrt(((2.0 * d + 1) * math.factorial(d - m)) / (4.0 * math.pi * math.factorial(d + m)))


def _SH(d: int, m: int, theta: Tensor, phi: Tensor):
    """实球谐函数
    Y_d^m(theta, phi) = K^m_d * P^m_d(cos(theta)) * e^(1j * m *phi)

    Args:
        d: >= 0, degree
        m: |m| <= d
        theta: [0, pi]
        phi: [0, 2*pi]
    """
    assert -d <= m <= d
    if m == 0:
        return _SH_K(d, 0) * _SH_P(d, m, theta.cos())
    elif m > 0:
        return math.sqrt(2) * _SH_K(d, m) * _SH_P(d, m, theta.cos()) * (m * phi).cos()
    else:
        return math.sqrt(2) * _SH_K(d, -m) * _SH_P(d, -m, theta.cos()) * (-m * phi).sin()


def _SH_complex(d: int, m: int, theta: Tensor, phi: Tensor):
    """复球谐函数
    Y_d^m(theta, phi) = K^m_d * P^m_d(cos(theta)) * e^(1j * m *phi)

    Args:
        d: >= 0, degree
        m: |m| <= d
        theta: [0, pi]
        phi: [0, 2*pi]
    """
    assert -d <= m <= d
    if m == 0:
        real = _SH_K(d, 0) * _SH_P(d, m, theta.cos())
        image = torch.zeros_like(real)
    elif m > 0:
        c = math.sqrt(2) * _SH_K(d, m) * _SH_P(d, m, theta.cos())
        real = c * (m * phi).cos()
        image = c * (m * phi).sin()
    else:
        c = math.sqrt(2) * _SH_K(d, -m) * _SH_P(d, -m, theta.cos()) * (-1) ** m
        real = c * (m * phi).cos()
        image = c * (m * phi).sin()
    return torch.stack([real, image], dim=-1)


# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
_SH_C = [
    0.28209479177387814,  # 1/2 * math.sqrt(1/math.pi)

    -0.4886025119029199,  # -1/2 * math.sqrt(3/math.pi)  * y,
    0.4886025119029199,  # 1/2 * math.sqrt(3/math.pi)  * z
    -0.4886025119029199,  # -1/2 * math.sqrt(3/math.pi)  * x

    1.0925484305920792,  # 1/2 * math.sqrt(15/math.pi) * xy
    -1.0925484305920792,  # -1/2 * math.sqrt(15/math.pi) * yz
    0.31539156525252005,  # 1/4 * math.sqrt(5/math.pi) * (3zz-1)
    -1.0925484305920792,  # -1/2 * math.sqrt(15/math.pi) * xz
    0.5462742152960396,  # 1/4 * math.sqrt(15/math.pi) * (xx - yy)

    -0.5900435899266435,  # 1/8 * math.sqrt(70/math.pi) * y(-3xx + yy)
    2.890611442640554,  # 1/2 * math.sqrt(105/math.pi) * (xyz)
    -0.4570457994644658,  # 1/8 * math.sqrt(42/math.pi) * y(1 - 5zz)
    0.3731763325901154,  # 1/4 * math.sqrt(7/math.pi) * z(5zz - 3)
    -0.4570457994644658,  # 1/8 * math.sqrt(42/math.pi) * x(1 - 5zz)
    1.445305721320277,  # 1/4 * math.sqrt(105/math.pi) * z(xx - yy)
    -0.5900435899266435,  # 1/8 * math.sqrt(70/math.pi) * x(-xx + 3yy)

    2.5033429417967046,  # 3/4 * math.sqrt(35/math.pi) * xy * (xx - yy)
    -1.7701307697799304,  # 3/8 * math.sqrt(70/math.pi) * yz * (3xx - yy)
    0.9461746957575601,  # 3/4 * math.sqrt(5/math.pi) * xy * (7zz - 1)
    -0.6690465435572892,  # 3/8 * math.sqrt(10/math.pi) * yz * (7zz - 3)
    0.10578554691520431,  # 3/16 * math.sqrt(1/math.pi) * (35 z4 -30 zz + 3)
    -0.6690465435572892,  # 3/8 * math.sqrt(10/math.pi) * xz * (7zz - 3)
    0.47308734787878004,  # 3/8 * math.sqrt(5/math.pi) * (xx - yy) * (7zz - 1)
    -1.7701307697799304,  # 3/8 * math.sqrt(70/math.pi) * xz * (xx  - 3yy)
    0.6258357354491761,  # 3/16 * math.sqrt(35/math.pi) * (x4 - 6xxyy +y4)
]


class _sh_encoder(Function):
    _forward = get_C_function('sh_encode_forward')
    _backward = get_C_function('sh_encode_backward')

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')  # force float32 for better precision
    def forward(ctx, inputs, degree):
        # inputs: [B, input_dim], float in [-1, 1]
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        B, input_dim = inputs.shape  # batch size, coord dim

        output_dim = (degree + 1) ** 2

        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        _sh_encoder._forward(inputs, outputs, B, input_dim, degree)

        ctx.save_for_backward(inputs)
        ctx.dims = [B, input_dim, degree]

        return outputs

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad):
        # grad: [B, (C + 2) ** 2]
        inputs, = ctx.saved_tensors
        grad = grad.contiguous()
        B, input_dim, degree = ctx.dims
        grad_inputs = torch.zeros_like(inputs)
        _sh_encoder._backward(grad, inputs, B, input_dim, degree, grad_inputs)
        return grad_inputs, None, None


@try_use_C_extension(_sh_encoder.apply, "sh_encode_forward", 'sh_encode_backward')
def sh_encode(dirs: Tensor, degree: int) -> Tensor:
    assert 4 >= degree >= 0
    coeff = (degree + 1) ** 2
    outputs = dirs.new_zeros((*dirs.shape[:-1], coeff))

    x, y, z = dirs.unbind(dim=-1)
    outputs[..., 0] = _SH_C[0] + x * 0  # mask dir_grad=0 when degree = 0
    if degree == 0:
        return outputs
    outputs[..., 1] = _SH_C[1] * y
    outputs[..., 2] = _SH_C[2] * z
    outputs[..., 3] = _SH_C[3] * x
    if degree <= 1:
        return outputs

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    outputs[..., 4] = _SH_C[4] * xy
    outputs[..., 5] = _SH_C[5] * yz
    outputs[..., 6] = _SH_C[6] * (3.0 * zz - 1)
    outputs[..., 7] = _SH_C[7] * xz
    outputs[..., 8] = _SH_C[8] * (xx - yy)

    if degree <= 2:
        return outputs
    outputs[..., 9] = _SH_C[9] * y * (3 * xx - yy)
    outputs[..., 10] = _SH_C[10] * xy * z
    outputs[..., 11] = _SH_C[11] * y * (5 * zz - 1)
    outputs[..., 12] = _SH_C[12] * z * (5 * zz - 3)
    outputs[..., 13] = _SH_C[13] * x * (5 * zz - 1)
    outputs[..., 14] = _SH_C[14] * z * (xx - yy)
    outputs[..., 15] = _SH_C[15] * x * (xx - 3 * yy)
    if degree <= 3:
        return outputs

    outputs[..., 16] = _SH_C[16] * xy * (xx - yy)
    outputs[..., 17] = _SH_C[17] * yz * (3 * xx - yy)
    outputs[..., 18] = _SH_C[18] * xy * (7 * zz - 1)
    outputs[..., 19] = _SH_C[19] * yz * (7 * zz - 3)
    outputs[..., 20] = _SH_C[20] * (zz * (35 * zz - 30) + 3)
    outputs[..., 21] = _SH_C[21] * xz * (7 * zz - 3)
    outputs[..., 22] = _SH_C[22] * (xx - yy) * (7 * zz - 1)
    outputs[..., 23] = _SH_C[23] * xz * (xx - 3 * yy)
    outputs[..., 24] = _SH_C[24] * (xx * xx - 6 * yy * xx + yy * yy)
    return outputs


class _SH_to_RGB(Function):
    _forward = get_C_function('SH_to_RGB_forward')
    _backward = get_C_function('SH_to_RGB_backward')

    @staticmethod
    def forward(ctx, *inputs):
        sh, dirs, campos, degree, clamp = inputs
        ctx._degree = degree
        ctx._clamp = clamp
        dirs = dirs.contiguous()
        sh = sh.contiguous()
        if campos is not None:
            campos = campos.contiguous()
        rgb = _SH_to_RGB._forward(dirs, campos, sh, degree, clamp)
        ctx.save_for_backward(sh, dirs, campos, rgb)
        return rgb

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_rgb):
        sh, dirs, campos, rgb = ctx.saved_tensors
        grad_sh = torch.zeros_like(sh) if ctx.needs_input_grad[0] else None
        grad_dirs = torch.zeros_like(dirs) if ctx.needs_input_grad[1] else None
        grad_campos = torch.zeros_like(campos) if campos is not None and ctx.needs_input_grad[2] else None
        _SH_to_RGB._backward(ctx._degree, ctx._clamp, dirs, campos, sh, rgb, grad_rgb, grad_dirs, grad_campos, grad_sh)
        return grad_sh, grad_dirs, grad_campos, None, None


def _SH_to_RGB_py(sh: Tensor, dirs: Tensor, campos: Optional[Tensor] = None, degree: int = 0, clamp=False) -> Tensor:
    """
    Evaluate spherical harmonics at unit directions using hardcoded SH polynomials.

    Args:
        degree: SH deg. Currently, 0-4 supported
        sh: SH coeffs [..., (deg + 1) ** 2, 3]
        dirs: unit directions [..., 3] or postions
        campos: camera positions, shape [3] or None
        clamp: max(0, rgb)
    Returns:
        [..., 3]
    """
    assert 4 >= degree >= 0
    coeff = (degree + 1) ** 2
    assert sh.ndim >= 2 and sh.shape[-2] >= coeff

    if campos is not None:
        dirs = F.normalize(dirs - campos, dim=-1, eps=1e-12)
    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = _SH_C[0] * sh[..., 0, :] + 0.5
    result = result + 0 * y  # grad for dirs
    if degree > 0:
        result = (result +
                  _SH_C[1] * sh[..., 1, :] * y +
                  _SH_C[2] * sh[..., 2, :] * z +
                  _SH_C[3] * sh[..., 3, :] * x)

    if degree > 1:
        result = (result +
                  _SH_C[4] * sh[..., 4, :] * xy +
                  _SH_C[5] * sh[..., 5, :] * yz +
                  _SH_C[6] * sh[..., 6, :] * (3.0 * zz - 1) +
                  _SH_C[7] * sh[..., 7, :] * xz +
                  _SH_C[8] * sh[..., 8, :] * (xx - yy))

    if degree > 2:
        result = (result +
                  _SH_C[9] * sh[..., 9, :] * y * (3 * xx - yy) +
                  _SH_C[10] * sh[..., 10, :] * xy * z +
                  _SH_C[11] * sh[..., 11, :] * y * (5 * zz - 1) +
                  _SH_C[12] * sh[..., 12, :] * z * (5 * zz - 3) +
                  _SH_C[13] * sh[..., 13, :] * x * (5 * zz - 1) +
                  _SH_C[14] * sh[..., 14, :] * z * (xx - yy) +
                  _SH_C[15] * sh[..., 15, :] * x * (xx - 3 * yy))

    if degree > 3:
        result = (result +
                  _SH_C[16] * sh[..., 16, :] * xy * (xx - yy) +
                  _SH_C[17] * sh[..., 17, :] * yz * (3 * xx - yy) +
                  _SH_C[18] * sh[..., 18, :] * xy * (7 * zz - 1) +
                  _SH_C[19] * sh[..., 19, :] * yz * (7 * zz - 3) +
                  _SH_C[20] * sh[..., 20, :] * (zz * (35 * zz - 30) + 3) +
                  _SH_C[21] * sh[..., 21, :] * xz * (7 * zz - 3) +
                  _SH_C[22] * sh[..., 22, :] * (xx - yy) * (7 * zz - 1) +
                  _SH_C[23] * sh[..., 23, :] * xz * (xx - 3 * yy) +
                  _SH_C[24] * sh[..., 24, :] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)))
    return result.clamp_min(0) if clamp else result


def SH_to_RGB(sh: Tensor, dirs: Tensor, campos: Optional[Tensor] = None, degree: int = 0, clamp=False) -> Tensor:
    """
    Evaluate spherical harmonics at unit directions using hardcoded SH polynomials.

    Args:
        degree: SH deg. Currently, 0-4 supported
        sh: SH coeffs [..., (deg + 1) ** 2, 3]
        dirs: unit directions [..., 3] or postions
        campos: camera positions, shape [3] or None
        clamp: max(0, rgb)
    Returns:
        [..., 3]
    """
    if sh.is_cuda and have_C_functions('SH_to_RGB_forward') and have_C_functions('SH_to_RGB_backward'):
        return _SH_to_RGB.apply(sh, dirs, campos, degree, clamp)
    else:
        return _SH_to_RGB_py(sh, dirs, campos, degree, clamp)


def RGB2SH(rgb):
    return (rgb - 0.5) / _SH_C[0]


def SH2RGB(sh):
    return sh * _SH_C[0] + 0.5


def rotation_SH(sh: Tensor, R: Tensor):
    """Reference:
        https://en.wikipedia.org/wiki/Wigner_D-matrix
        https://github.com/andrewwillmott/sh-lib
        http://filmicworlds.com/blog/simple-and-fast-spherical-harmonic-rotation/
    """
    from scipy.spatial.transform import Rotation
    import sphecerix  # noqa

    # option 1
    Robj = Rotation.from_matrix(R[..., :3, :3].cpu().numpy())
    B, N, _ = sh.shape
    sh = sh.transpose(1, 2).reshape(-1, N)
    new_sh = sh.clone()
    cnt = 0
    i = 0
    while cnt < N:
        D = sphecerix.tesseral_wigner_D(i, Robj)
        D = torch.from_numpy(D).to(sh)
        new_sh[:, cnt:cnt + D.shape[0]] = sh[:, cnt:cnt + D.shape[0]] @ D.T
        cnt += D.shape[0]
        i += 1

    # option 2
    # from e3nn import o3
    # rot_angles = o3._rotation.matrix_to_angles(R)
    # D_2 = o3.wigner_D(2, rot_angles[0], rot_angles[1], rot_angles[2])
    #
    # Y_2 = self._features_rest[:, [3, 4, 5, 6, 7]]
    # Y_2_rotated = torch.matmul(D_2, Y_2)
    # self._features_rest[:, [3, 4, 5, 6, 7]] = Y_2_rotated
    # print((sh - new_sh).abs().mean())
    return new_sh.reshape(B, 3, N).transpose(1, 2)
