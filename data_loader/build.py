from typing import Any
from packaging import version
import random
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import logging

from utils.config import get_parser
from utils import add_extend_list_option, add_bool_option, add_n_tuple_option, add_cfg_option, merge_dict
from .batch_collator import *
from .batch_samplers import *


def worker_init(worked_id):
    worker_seed = (torch.initial_seed() + worked_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def options(parser=None):
    group = get_parser(parser).add_argument_group('Data Loader Options ')
    group.add_argument('--num-workers', default=4, type=int, metavar='I', help="The number of data loader workers")
    add_extend_list_option(group, '-b', '--batch-size', default=[1], num=3, help='The batch size for train/val/test')
    add_bool_option(group, '--pin-memory', default=True, help='Enable pin memory for data loader')
    add_n_tuple_option(group, '--data-sampler', default=('default', 'default'))
    add_cfg_option(group, '--data-sampler-train-cfg')
    add_cfg_option(group, '--data-sampler-eval-cfg')
    add_n_tuple_option(group, '--data-collate', default=('default', 'default'))
    return group


def make(cfg, dataset, mode='train', batch_sampler: Any = 'default', collate_fn=None, batch_sampler_cfg=None, **kwargs):
    kwargs.setdefault('num_workers', cfg.num_workers)
    kwargs.setdefault('pin_memory', cfg.pin_memory)
    if version.parse(torch.__version__) >= version.parse('1.4.0'):
        kwargs.setdefault('worker_init_fn', worker_init)
    # g = torch.Generator()  # 设置样本shuffle随机种子，作为DataLoader的参数
    # g.manual_seed(0)
    # kwargs.setdefault('generator', g)

    assert mode in ['train', 'eval', 'test']
    if mode == 'train':
        batch_size = cfg.batch_size[0]
    elif mode == 'eval':
        batch_size = cfg.batch_size[1]
    else:
        batch_size = cfg.batch_size[2]
    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']

    if batch_sampler is None:
        batch_sampler = cfg.data_sampler[0 if mode == 'train' else 1]
    if isinstance(batch_sampler, str):
        if batch_sampler == 'default':
            batch_sampler = 'shuffle' if mode == 'train' else 'sequence'
        batch_sampler = BATCH_SAMPLERS[batch_sampler]

    batch_sampler_cfg = merge_dict(
        batch_sampler_cfg,
        cfg.data_sampler_train_cfg if mode == 'train' else cfg.data_sampler_eval_cfg,
        data_source=dataset, batch_size=batch_size
    )
    if isinstance(batch_sampler, IterableBatchSampler):
        batch_sampler_cfg = merge_dict(
            batch_sampler_cfg,
            length=cfg.eval_interval if cfg.eval_interval > 0 else cfg.epochs,
            num_split=1  # kwargs.get('num_split', kwargs['num_workers'])
        )
    batch_sampler = batch_sampler(**batch_sampler_cfg)

    if collate_fn is None:
        if hasattr(dataset, 'collate'):
            collate_fn = getattr(dataset, 'collate')
        else:
            collate_fn = DATA_COLLATOR[cfg.data_collate[0 if mode == 'train' else 1]]

    logging.info(f'use BatchSampler: {batch_sampler} when mode={mode}, collate_fn={collate_fn.__name__}')
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **kwargs)
    return data_loader
