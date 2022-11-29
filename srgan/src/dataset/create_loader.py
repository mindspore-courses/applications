# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Create SRGAN dataloader."""

import math
import multiprocessing

import mindspore.dataset as ds
from mindspore import context
from mindspore.context import ParallelMode

from src.dataset.data_loader import TrainDataset, TestDataset

__all__ = ["create_test_dataloader", "create_train_dataloader"]

def create_train_dataloader(batchsize, lr_path, gt_path, rank_id=0, device_num=1):
    """
    Create dataloader for training process.

    Args:
        batchsize (int): Batch size for training.
        lr_path (str): The path of low-resolution image.
        gt_path (str): The path of high-resolution image.
        rank_id (int): Identify the host. Default: 0.
        device_num (int): The number of host. Default: 1.

    Returns:
        dataloader (GeneratorDataset): Dataloader for training.
    """

    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / device_num)
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
        dataset = TrainDataset(lr_path, gt_path, in_memory=False)
        sampler = DataSampler(dataset, local_rank=rank_id, world_size=device_num)
        dataloader = ds.GeneratorDataset(dataset, column_names=['LR', 'HR'], shuffle=True,
                                         num_shards=device_num, shard_id=rank_id, sampler=sampler,
                                         python_multiprocessing=True,
                                         num_parallel_workers=min(12, num_parallel_workers)
                                         )
        dataloader = dataloader.batch(batchsize, drop_remainder=True,
                                      num_parallel_workers=min(8, num_parallel_workers))
    else:
        dataset = TrainDataset(lr_path, gt_path, in_memory=False)
        dataloader = ds.GeneratorDataset(dataset, column_names=['LR', 'HR'], shuffle=True,
                                         python_multiprocessing=True,
                                         num_parallel_workers=min(12, num_parallel_workers))
        dataloader = dataloader.batch(batchsize, drop_remainder=True, num_parallel_workers=min(8, num_parallel_workers))
    return dataloader

def create_test_dataloader(batchsize, lr_path, gt_path='', inference=False):
    """
    Create dataloader for evaluating process and inferring process.

    Args:
        batchsize (int): Batch size for evaluating or inferring.
        lr_path (str): The path of low-resolution image.
        gt_path (str): The path of high-resolution image. Default: ''.
        inference (bool): Choose inference mode or evaluation model. Default: False.

    Returns:
        dataloader (GeneratorDataset): Dataloader for evaluating or inferring.
    """
    dataset = TestDataset(lr_path, gt_path, infer=inference, in_memory=False)
    if inference:
        dataloader = ds.GeneratorDataset(dataset, column_names=["LR"], shuffle=False)
    else:
        dataloader = ds.GeneratorDataset(dataset, column_names=["LR", "HR"], shuffle=False)
    dataloader = dataloader.batch(batchsize)
    return dataloader

class DataSampler():
    """
    Object used to choose samples from the dataset.

    Args:
        dataset (Mydata): Train data file in the specified format.
        local_rank (int): Identify the host.
        world_size (int): The number of host.
    """
    def __init__(self, dataset, local_rank, world_size):
        self.__num_data = len(dataset)
        self.__local_rank = local_rank
        self.__world_size = world_size
        self.samples_per_rank = int(math.ceil(self.__num_data / float(self.__world_size)))
        self.total_num_samples = self.samples_per_rank * self.__world_size

    def __iter__(self):
        """"iter"""
        indices = list(range(self.__num_data))
        indices.extend(indices[:self.total_num_samples-len(indices)])
        indices = indices[self.__local_rank:self.total_num_samples:self.__world_size]
        return iter(indices)

    def __len__(self):
        """length"""
        return self.samples_per_rank
