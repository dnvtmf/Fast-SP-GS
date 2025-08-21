#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Created : 2021/05/19
# @Author  : Wan Diwen
# @FileName: accuracy_metric.py
import torch
from utils.metrics.base import *

__all__ = ['AccuracyMetric']


@METRICS.register('ACC')
class AccuracyMetric(Metric):
    def __init__(self, topK=(1,), device=None, **kwargs):
        self.topK = list(x - 1 for x in sorted(topK))
        self.maxK = topK[-1] + 1
        self.device = device
        self.num_samples = 0
        self.accuracies = torch.zeros(len(self.topK), device=device)

        for i, k in enumerate(self.topK):
            self.add_metric(f'top{k + 1}', lambda i=i: self.accuracies[i].item())

    def reset(self):
        self.num_samples = 0
        self.accuracies = torch.zeros(len(self.topK), device=self.device)

    @torch.no_grad()
    def update(self, outputs: torch.Tensor, labels: torch.Tensor, dim=1):
        assert outputs.ndim == labels.ndim + 1
        self.num_samples += labels.numel()
        _, prediction = outputs.topk(self.maxK, dim, True, True)
        correct = prediction.eq(labels.unsqueeze(dim)).cumsum(dim)
        result = correct.sum(dim=list(range(dim)) + list(range(dim + 1, outputs.ndim)))[self.topK]
        self.accuracies = self.accuracies + result
        return result / labels.numel()

    @torch.no_grad()
    def summarize(self):
        # self.num_samples = reduce_tensor(self.accuracies.new_tensor(self.num_samples)).item()
        # self.accuracies = reduce_tensor(self.accuracies) / self.num_samples
        return {k + 1: v.item() for k, v in zip(self.topK, self.accuracies)}

    def __repr__(self):
        return ', '.join(f'acc@{k + 1}={v.item():.2%}' for k, v in zip(self.topK, self.accuracies))
