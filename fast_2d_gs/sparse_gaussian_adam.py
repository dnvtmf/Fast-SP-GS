"""Based on https://github.com/fatPeter/mini-splatting2/blob/5ec9202c2db7d900728bfc09733e85467cff885e/submodules/diff-gaussian-rasterization_ms/diff_gaussian_rasterization_ms/__init__.py#L497 """
import torch
import torch.optim

from fast_2d_gs._C import get_C_function

__all__ = ['SparseGaussianAdam']


class SparseGaussianAdam(torch.optim.Adam):
    _update_C = get_C_function('AdamMasksedUpdated')

    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
        self.N = None
        self.visibility = None

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None or torch.prod(torch.tensor(param.grad.shape)) == 0:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]

            # compensate lr for sparse adam, (1-b2**step)**0.5/(1-b1**step)
            state['step'] += 1
            step = state['step']

            M = param.numel() // self.N
            self._update_C(param, param.grad, exp_avg, exp_avg_sq, self.visibility,
                           lr * (1 - 0.999 ** step) ** 0.5 / (1 - 0.9 ** step), 0.9, 0.999, eps, self.N, M)

    def set_before_step(self, visibility, N):
        self.visibility, self.N = visibility, N
