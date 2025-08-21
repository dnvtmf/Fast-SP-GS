from typing import Callable, Dict

import torch
from torch import Tensor
from torch.distributed import all_reduce
from .base import Metric, METRICS


@METRICS.register('AVG')
class AverageMetric(Metric):
    _names = {'sum': None, 'avg': None, 'ravg': None}

    def __init__(self, reduce=False, momentum=0, fmt: Callable[[float], str] = '{:6.3f}'.format, **kwargs):
        """

        Args:
            reduce: gather data from multi-gpu, and calcuate the sum and average during updating
            momentum: The momentum for running average
        """
        self._sum = torch.tensor([0.])
        self._cnt = torch.zeros(1, dtype=torch.int64)
        self._running_average = torch.tensor([0.])
        self.reduce = False  # reduce and get_world_size() > 1
        self.momentum = momentum
        self.fmt = fmt

    def reset(self, reduce=False, momentum=0):
        self._sum = torch.tensor([0.])
        self._cnt = torch.zeros(1, dtype=torch.int64)
        self._running_average = torch.tensor([0.])
        self.reduce = False
        self.momentum = momentum

    @torch.no_grad()
    def update(self, value: Tensor, n=1):
        """"""
        if self.reduce:
            value = value.detach().cuda() * n
            all_reduce(value)
            n_tensor = value.new_tensor(n, dtype=torch.int64)
            all_reduce(n_tensor)
        else:
            value = value.detach() * n
        if self._cnt == 0:
            self._sum = value
            self._running_average = value
            self._cnt = self._cnt.to(value.device)
        else:
            self._sum += value
            self._running_average = self.momentum * self._running_average + (1. - self.momentum) * value / n
        self._cnt += n

    # @property
    def avg(self) -> Tensor:
        return self._sum / self._cnt

    # @property
    def ravg(self) -> Tensor:
        return self._running_average

    # @property
    def sum(self) -> Tensor:
        return self._sum

    def summarize(self):
        return

    def __repr__(self):
        avg = self.avg()
        if avg.numel() == 1:
            return self.fmt(avg.item())
        elif avg.ndim == 1:
            return '[' + ', '.join(([self.fmt(x.item()) for x in avg])) + ']'
        elif avg.ndim == 2:
            return '[' + ',\n '.join(['[' + ', '.join([self.fmt(x.item()) for x in y]) + ']' for y in avg]) + ']'
        else:
            return str(avg)


class AverageDictMetirc(Metric):
    _names = {'sum': None, 'avg': None, 'ravg': None}

    def __init__(self, reduce=False, momentum=0, fmt: Callable[[float], str] = '{:6.3f}'.format):
        self.reduce = reduce
        self.momentum = momentum
        self.fmt = fmt
        self._dict = {}  # type: Dict[str, AverageMetric]

    def reset(self):
        self._dict = {}

    @torch.no_grad()
    def update(self, *args, n=1, **kwargs):
        update_dict = args[0] if len(args) > 0 else kwargs
        for key, value in update_dict.items():
            if key not in self._dict:
                self._dict[key] = AverageMetric(reduce=self.reduce, momentum=self.momentum, fmt=self.fmt)
            self._dict[key].update(value, n)

    @torch.no_grad()
    def update_batch(self, *args, **kwargs):
        update_dict = args[0] if len(args) > 0 else kwargs
        for key, value in update_dict.items():
            if key not in self._dict:
                self._dict[key] = AverageMetric(reduce=self.reduce, momentum=self.momentum, fmt=self.fmt)
            self._dict[key].update(value.mean(dim=0), value.shape[0] if value.ndim > 0 else 1)

    # @property
    def avg(self) -> Dict[str, Tensor]:
        return {k: v.avg for k, v in self._dict.items()}

    # @property
    def ravg(self) -> Dict[str, Tensor]:
        return {k: v.ravg for k, v in self._dict.items()}

    # @property
    def sum(self) -> Dict[str, Tensor]:
        return {k: v.sum for k, v in self._dict.items()}

    def summarize(self):
        for m in self._dict.values():
            m.summarize()
        return

    def __repr__(self):
        return '{' + ', '.join(f"{k}: {v}" for k, v in self._dict.items()) + '}'
