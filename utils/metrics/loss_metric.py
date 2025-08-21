from typing import Callable
from collections import defaultdict
import math

import torch
from torch import Tensor

from utils import float2str
from .base import Metric, METRICS


@METRICS.register('loss')
class LossMetirc(Metric):
    _names = None

    def __init__(self, fmt: Callable[[float], str] = float2str, momentum=0.9, items=None, **kwargs):
        if items is not None:
            self._names = {item: None for item in items}
        else:
            self._names = None
        self.momentum = momentum
        self.fmt = fmt
        self.data = defaultdict(lambda: defaultdict(float))
        self.data['loss'] = {'val': 0, 'ravg': 0, 'sum': 0, 'cnt': 0}

    def reset(self):
        self.data = defaultdict(lambda: defaultdict(float))
        self.data['loss'] = {'val': 0, 'ravg': 0, 'sum': 0, 'cnt': 0}

    @torch.no_grad()
    def update(self, losses: Tensor, n=1, **kwargs: Tensor):
        kwargs['loss'] = losses
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.data[k]['val'] = v
            self.data[k]['ravg'] = self.momentum * self.data[k]['ravg'] + (1.0 - self.momentum) * v
            self.data[k]['sum'] += v * n
            self.data[k]['cnt'] += n

    @property
    def names(self):
        return tuple(self.data.keys()) if self._names is None else tuple(self._names.keys())

    def get_result(self, name='loss'):
        return 0 if self.data[name]['cnt'] == 0 else self.data[name]['sum'] / self.data[name]['cnt']

    @property
    def value(self):
        return ', '.join(['{}={}'.format(k, self.fmt(v['val'])) for k, v in self.data.items()])

    @property
    def average(self):
        if self.data['loss']['cnt'] == 0:
            return ', '.join(['{}={}'.format(k, self.fmt(math.nan)) for k in self.data.keys()])
        else:
            return ', '.join(['{}={}'.format(k, self.fmt(v['sum'] / v['cnt'])) for k, v in self.data.items()])

    @property
    def running_average(self):
        return ', '.join(['{}={}'.format(k, self.fmt(v['ravg'])) for k, v in self.data.items()])

    @property
    def sum(self):
        return ', '.join(['{}={}'.format(k, self.fmt(v['sum'])) for k, v in self.data.items()])

    def get_average(self):
        return {k: v['sum'] / v['cnt'] for k, v in self.data.items()}

    def get_sum(self):
        return {k: v['sum'] for k, v in self.data.items()}
