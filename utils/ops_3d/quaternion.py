"""四元数 quaternion
表示为shape[-1] == 4的Tensor, 分别表示 q(x, y, z, w) = w + xi + yj + zk 

Reference:
    - https://krasjet.github.io/quaternion/quaternion.pdf
    - https://zhuanlan.zhihu.com/p/375199378
"""
from typing import Union, Any

import numpy as np
import torch
from torch import Tensor
from fast_2d_gs._C import get_C_function, try_use_C_extension


def norm(q: Tensor, keepdim=False) -> Tensor:
    return q.norm(dim=-1, keepdim=keepdim)


def normalize(q: Tensor):
    return torch.nn.functional.normalize(q, dim=-1)


def add(q1: Tensor, q2: Tensor):
    return q1 + q2


def mul(q: Tensor, s: Union[float, Tensor]):
    if isinstance(s, float):
        return q * s
    elif isinstance(s, Tensor):
        if q.ndim > s.ndim:
            return q * s[..., None]
        else:
            assert s.shape[-1] == 4
            # yapf: disable
            # b, c, d, a = q.unbind(-1)
            # f, g, h, e = q.unbind(-1)
            # return torch.stack([
            #     a * e - b * f - c * g - d * h,
            #     b * e + a * f - d * g + c * h,
            #     c * e + d * f + a * g - b * h,
            #     d * e - c * f + b * g + a * h,
            # ], dim=-1)
            # #(Graßmann Product)
            qxyz, qw = q[..., :3], q[..., -1:]
            sxyz, sw = s[..., :3], s[..., -1:]
            return torch.cat([
                sw * qxyz + qw * sxyz + torch.linalg.cross(qxyz, sxyz),
                qw * sw - torch.linalg.vecdot(qxyz, sxyz)[..., None]
            ], dim=-1)
            # yapf: enable
    else:
        raise ValueError()


def conj(q: Tensor):
    """共轭 conjugate"""
    return torch.cat([-q[..., :3], q[..., -1:]], dim=-1)


def inv(q: Tensor):
    """四元数的逆

    q是单位四元数时, 逆为conj(q)"""
    return conj(q) / norm(q)[..., None]


def cross(qa: Tensor, qb: Tensor):
    qc = torch.zeros_like(qa)
    qc[..., :3] = qa[..., 3:] * qb[..., :3] + qb[..., 3:] * qa[..., :3] + torch.cross(qa[..., :3], qb[..., :3], dim=-1)
    return qc


def from_rotate(u: Tensor, theta: Tensor):
    """ 从旋转轴u和旋转角θ构造四元数

    Args:
        u: 旋转轴, 单位向量, shape: [..., 3]
        theta: 旋转角, 弧度; shape [....]
    Returns:
        四元数 [..., 4]
    """
    theta = theta[..., None] * 0.5
    return torch.cat([theta.sin() * u, theta.cos()], dim=-1)


def to_rotate(q: Tensor):
    """
    从四元数提取 旋转轴u和旋转角θ

    Args:
        q: 单位四元数 shape [..., 4]

    Returns:
        u: 旋转轴, 单位向量, shape: [..., 3]; theta: 旋转角, 弧度; shape [....]
    """
    theta = torch.arccos(q[..., -1])
    u = q[..., :3] / torch.sin(theta)
    return u, 2. * theta


def xfm(points: Tensor, q: Tensor):
    """使用四元数旋转点"""
    points = torch.cat([points, torch.zeros_like(points[..., :1])], dim=-1)
    return mul(mul(q, points), conj(q))[..., :3]


def pow(q: Tensor, t: Tensor):
    u, theta = to_rotate(q)
    return from_rotate(u, t * theta)


def interpolation(t: Union[float, Tensor], q1: Tensor, q2: Tensor, method='slerp'):
    """插值, 插值角度较小时可使用nlerp, slerp避免插值角度接近0"""
    if method == 'slerp0':  # Spherical Linear Interpolation
        return mul(pow(mul(q2, conj(q1)), t), q1)
    elif method == 'slerp':
        theta = torch.arccos(torch.linalg.vecdot(q1, q2))
        a = torch.sin((1 - t) * theta)
        b = torch.sin(t * theta)
        c = torch.sin(theta)
        return a / c * q1 + b / c * q2
    elif method == 'nlerp':  # 正规化线性插值（Normalized Linear Interpolation）
        return normalize((1 - t) * q1 + t * q2)
    elif method == 'squad':  # 球面四边形插值 (Spherical and quadrangle)
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def standardize(quaternions: Tensor) -> Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real part is non negative.

    Args:
        quaternions: Quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


class _to_R(torch.autograd.Function):
    _forward = get_C_function('quaternion_to_R_forward')
    _backward = get_C_function('quaternion_to_R_backward')

    @staticmethod
    def forward(ctx, *args, **kwargs):
        q, = args
        R = _to_R._forward(q)
        ctx.save_for_backward(q)
        return R

    @staticmethod
    def backward(ctx, *grad_outputs):
        q, = ctx.saved_tensors
        grad_q = _to_R._backward(q, grad_outputs[0])
        return grad_q


@try_use_C_extension(_to_R.apply, "quaternion_to_R_forward", "quaternion_to_R_backward")
def toR(q: Tensor):
    """将四元数标准化并得到旋转矩阵"""
    x, y, z, w = normalize(q).unbind(-1)
    # yapf: disable
    R = torch.stack([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * w * y + 2 * x * z,
        2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x,
        2 * x * z - 2 * w * y, 2 * w * x + 2 * y * z, 1 - 2 * x * x - 2 * y * y,
    ], dim=-1).reshape(*x.shape, 3, 3)
    # yapf: enable
    return R


def weighted_avearge(qs: np.ndarray, weights: np.ndarray):
    from scipy.spatial.transform import Rotation
    # 选择参考四元数（权重最大的四元数）
    ref_index = np.argmax(weights)
    q_ref = qs[ref_index]

    # 计算相对四元数并映射到切空间
    v_avg = np.zeros(3)  # 切空间中的平均向量
    total_weight = np.sum(weights)

    for i, (q, w) in enumerate(zip(qs, weights)):
        if i == ref_index:
            continue  # 参考四元数的相对四元数是单位四元数，对切空间无贡献
        q_rel = Rotation.from_quat(q_ref).inv() * Rotation.from_quat(q)
        v_i = Rotation.from_quat(q_rel.as_quat()).as_rotvec()
        v_avg += w * v_i

    v_avg /= total_weight

    # 将平均向量映射回四元数空间
    q_avg_rel = Rotation.from_rotvec(v_avg).as_quat()
    q_avg = Rotation.from_quat(q_ref) * Rotation.from_quat(q_avg_rel)

    # 归一化
    q_avg = q_avg.as_quat()
    q_avg /= np.linalg.norm(q_avg)

    return q_avg

