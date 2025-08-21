from typing import Type, Callable

import numpy as np
from torch import nn
from utils.registry import Registry

LOSSES = Registry()  # type: Registry[Type[nn.Module]]


class LossDict(nn.Module):
    """
    loss_cfg:
        name:
            lambda: v0
            _vary: type
            _steps: [s1, s2, s3]
            _valuse: [v1, v2, v3]
        name: <v0>
    weight change:
        [0, s1): vary(v0, v1)
        [s1, s2): vary(v1, v2)
        [s2, s3): vary(v2, v3)
        >= s3: v3
    """

    def __init__(self, default=0.0, **kwargs):
        super().__init__()
        self.default = default
        self.loss_functions = {}
        for name, cfg in kwargs.items():
            item = {
                'func': None,
                'lambda': default,
                'vary': 'fix',
                'steps': [],
                'values': [],
            }
            self.loss_functions[name] = item
            if isinstance(cfg, (float, int, bool)):
                item['lambda'] = cfg
                cfg = {}
            elif isinstance(cfg, dict):
                item['lambda'] = cfg.pop('lambda', default)
                item['vary'] = cfg.pop('_vary', 'fix')
                item['steps'] = cfg.pop('_steps', [])
                item['values'] = cfg.pop('_values', [])
                assert len(item['steps']) == len(item['values'])
                name = cfg.pop('_type', name)
            else:
                raise RuntimeError(f"error cfg: {cfg}, must be number or dict")
            if name in LOSSES:
                item['func'] = LOSSES[name](**cfg)  # noqa
        self._step = 0

    def forward(self, name, *inputs, **kwargs):
        w = self.w(name)
        if w <= 0:
            return 0
        if len(inputs) > 0 and isinstance(inputs[0], Callable):
            func = inputs[0]
            inputs = inputs[1:]
        else:
            func = self.loss_functions.get(name, {}).get('func', None)
        return (inputs[0] if func is None else func(*inputs, **kwargs)) * w

    def w(self, name):
        if name not in self.loss_functions:
            return self.default
        vary = self.loss_functions[name]['vary']
        steps = self.loss_functions[name]['steps']
        values = self.loss_functions[name]['values']
        if len(steps) == 0:
            return self.loss_functions[name]['lambda']
        stage = (self._step >= np.array(steps)).sum()
        if stage == len(steps):
            return max(0, values[-1])
        elif stage == 0:
            return self.loss_functions[name]['lambda']
        else:
            v1, v2 = values[stage - 1], values[stage]
            if v2 < 0:
                return 0
            s1, s2 = steps[stage - 1], steps[stage]
            ratio = (self._step - s1) / max(s2 - s1, 1)
            if isinstance(vary, list):
                vary = vary[stage]
            if vary == 'fix':
                return v2
            elif vary == 'linear':
                return v1 * (1 - ratio) + v2 * ratio
            elif vary == 'log':
                v2 = v2 if v2 > 0 else 1e-10
                return np.exp(np.log(v1) * (1 - ratio) + np.log(v2) * ratio)
            else:
                raise NotImplementedError(f"lambda vary type {vary} is not supported")

    def change_with_training_progress(self, step=0, num_steps=1, epoch=0, num_epochs=1):
        self._step = epoch * num_steps + step

    def extra_repr(self) -> str:
        fmt_s = []
        for k, v in self.loss_functions.items():
            if v['steps'] or v['values']:
                fmt_s.append(f"{k}={v['vary']}")
            elif v['lambda'] > 0:
                fmt_s.append(f"{k}={v['lambda']}")
        return ', '.join(fmt_s)


def test():
    import matplotlib.pyplot as plt
    loss_func = LossDict(
        log={'lambda': 0, '_vary': 'log', '_steps': [10, 100, 200], '_values': [1e-5, 1e-4, 1e-2]},
        fix={'lambda': 0, '_vary': 'fix', '_steps': [10, 100, 200], '_values': [1e-5, 1e-4, 1e-2]},
        linear={'lambda': 0, '_vary': 'linear', '_steps': [10, 100, 200], '_values': [1e-5, 1e-4, 1e-2]},
        test={'lambda': 0, '_vary': 'fix', '_steps': [0, 200, 200], '_values': [0, 0, 1e-3]},
    )
    x = np.arange(300)
    y1, y2, y3, y4 = [], [], [], []
    for i in range(300):
        loss_func.change_with_training_progress(i, 300, 0, 1)
        y1.append(loss_func.w('fix'))
        y2.append(loss_func.w('log'))
        y3.append(loss_func.w('linear'))
        y4.append(loss_func.w('test'))

    plt.plot(x, np.array(y1), label='fix')
    plt.plot(x, np.array(y2), label='log')
    plt.plot(x, np.array(y3), label='linear')
    plt.plot(x, np.array(y4), label='test')
    plt.yscale('log')
    plt.legend()
    plt.show()
