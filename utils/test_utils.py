import torch
from torch import Tensor
from .torch_utils import tensor_to

__all__ = ['get_run_speed', 'get_abs_error', 'get_rel_error', 'show_max_different', 'clone_tensors']


def get_rel_error(pred: Tensor, gt: Tensor, msg=None, threshold=None, eps=1e-6):
    """get relative error"""
    if not isinstance(pred, Tensor) or not isinstance(gt, Tensor):
        print(f"\033[45m{msg}, {type(pred)}, {type(gt)}, {pred}, {gt}\033[0m")
        return msg
    assert pred.shape == gt.shape, f"{msg}: have unmatched shape {pred.shape} vs {gt.shape}"
    error = ((gt - pred).abs().max() / (gt.abs().max() + eps)).item() if gt.numel() > 0 else 0.
    bg = ''
    if error > 1e-3:
        bg = '\033[41m'
    elif error > 1e-6:
        bg = '\033[44m'
    if msg is not None:
        print(f"{msg}: \033[33m{bg}{error:.4e}\033[0m")
    if threshold is not None:
        assert error < threshold
    return msg


def get_abs_error(pred: Tensor, gt: Tensor, msg=None, threshold=None, warn_t=1e-6, error_t=1e-3):
    """get relative error"""
    assert pred.shape == gt.shape, f"{msg}: have unmatched shape {pred.shape} vs {gt.shape}"
    error = (gt - pred).abs().max().item() if gt.numel() > 0 else 0.
    bg = ''
    if error > error_t:
        bg = '\033[41m'
    elif error > warn_t:
        bg = '\033[44m'
    if msg is not None:
        print(f"{msg}: \033[33m{bg}{error:.4e}\033[0m")
    if threshold is not None:
        assert error < threshold
    return msg


def get_run_speed(inputs, grads, py_func=None, cu_func=None, cpu_func=None, num_test=100, **kwargs):
    if isinstance(inputs, Tensor):
        inputs = (inputs,)
    timer = [torch.cuda.Event(enable_timing=True) for _ in range(num_test * 3 + 3)]
    kwargs['cuda'] = (cu_func, 'cuda')
    kwargs['python'] = (py_func, 'cuda')
    kwargs['cpu'] = (cpu_func, 'cpu')
    for name, func_and_device in kwargs.items():
        if isinstance(func_and_device, (tuple, list)):
            func, device = func_and_device
        else:
            func, device = func_and_device, 'cuda'
        if func is None:
            continue
        inputs, grads = tensor_to(inputs, grads, device=torch.device(device))
        t_forward = 0
        t_backward = 0
        for step in range(num_test + 1):
            timer[step * 3 + 0].record()
            output = func(*inputs)
            timer[step * 3 + 1].record()
            if grads is not None:
                if isinstance(grads, (tuple, list)):
                    assert len(grads) == len(output), f"{len(grads)} != {len(output)}"
                    outputs = []
                    grad_outputs = []
                    for o, g in zip(output, grads):
                        if g is not None:
                            outputs.append(o)
                            grad_outputs.append(g)
                    torch.autograd.backward(outputs, grad_outputs)
                else:
                    torch.autograd.backward(output, grads)
            timer[step * 3 + 2].record()
        for step in range(1, num_test + 1):
            timer[step * 3 + 0].synchronize()
            timer[step * 3 + 1].synchronize()
            timer[step * 3 + 2].synchronize()
            t_forward += timer[step * 3 + 0].elapsed_time(timer[step * 3 + 1])
            t_backward += timer[step * 3 + 1].elapsed_time(timer[step * 3 + 2])
        t_forward = t_forward / num_test
        t_backward = t_backward / num_test
        if grads is not None:
            print(f'{name:10s} time: forward {t_forward:.3f} ms, backward {t_backward:.3f} ms, '
                  f'total: {t_backward + t_forward:.3f} ms')
        else:
            print(f'{name:10s} time: forward {t_forward:.3f} ms')


def clone_tensors(*args: Tensor, device='cuda'):
    outputs = [x.to(device).detach().clone().requires_grad_() for x in args]
    return outputs[0] if len(args) == 1 else outputs


def show_max_different(a: Tensor, b: Tensor, dim: int = None):
    assert a.shape == b.shape
    index = (a - b).abs().argmax().item()
    if dim is None:
        a = a.reshape(-1)
        b = b.reshape(-1)
        print(f"{index=}, {a[index]} vs {b[index]}")
        return
    if dim < 0:
        dim += a.ndim
    prefix = 1
    suffix = 1
    now = 1
    for i, s in enumerate(a.shape):
        if i < dim:
            prefix *= s
        elif i > dim:
            suffix *= s
        else:
            now = s
    a = a.reshape(prefix, now, suffix)
    b = b.reshape(prefix, now, suffix)
    pi = index // (now * suffix)
    si = index % suffix
    print(f"index={pi} {si}")
    print(a[pi, :, si])
    print(b[pi, :, si])
