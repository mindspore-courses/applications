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
""" The public API for dataset. """

import os
from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, Union, Tuple

import mindspore.dataset as ds

from utils.generator import DatasetGenerator


class Dataset:
    """
    Dataset is the base class for making dataset which are compatible with MindSpore Vision.
    """

    def __init__(self,
                 path: str,
                 split: str,
                 load_data: Union[Callable, Tuple],
                 transform: Optional[Callable],
                 target_transform: Optional[Callable],
                 batch_size: int,
                 repeat_num: int,
                 resize: Union[int, Tuple[int, int]],
                 shuffle: bool,
                 num_parallel_workers: Optional[int],
                 num_shards: int,
                 shard_id: int,
                 mr_file: Optional[str] = None,
                 columns_list: Tuple = ('image', 'label'),
                 mode: Optional[str] = None) -> None:
        ds.config.set_enable_shared_mem(False)
        self.path = os.path.expanduser(path)
        self.split = split

        if len(columns_list) == 3 and self.split != "infer":
            self.image, self.image_id, self.label = load_data()
        else:
            self.image, self.label = load_data(self.path) if self.split == "infer" else load_data()
            self.image_id = None
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.repeat_num = repeat_num
        self.resize = resize
        self.shuffle = shuffle
        self.num_parallel_workers = num_parallel_workers
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.mode = mode
        self.mr_file = mr_file
        self.columns_list = columns_list
        if self.mr_file:
            self.dataset = ds.MindDataset(mr_file,
                                          columns_list=list(self.columns_list),
                                          num_parallel_workers=num_parallel_workers,
                                          shuffle=self.shuffle,
                                          num_shards=self.num_shards,
                                          shard_id=self.shard_id)
        else:
            if self.image_id:
                self.dataset = ds.GeneratorDataset(DatasetGenerator(self.image,
                                                                    self.label,
                                                                    self.image_id,
                                                                    mode=self.mode),
                                                   column_names=list(self.columns_list),
                                                   num_parallel_workers=num_parallel_workers,
                                                   shuffle=self.shuffle,
                                                   num_shards=self.num_shards,
                                                   shard_id=self.shard_id)
            else:
                self.dataset = ds.GeneratorDataset(DatasetGenerator(self.image,
                                                                    self.label,
                                                                    mode=self.mode),
                                                   column_names=list(self.columns_list),
                                                   num_parallel_workers=num_parallel_workers,
                                                   shuffle=self.shuffle,
                                                   num_shards=self.num_shards,
                                                   shard_id=self.shard_id)

    @property
    def get_path(self):
        """Get path in imagenet dataset which will be train or val folder."""

        return os.path.join(self.path, self.split)

    def download_dataset(self):
        """Download the dataset."""
        raise NotImplementedError

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        raise NotImplementedError

    def default_transform(self):
        """Default data augmentation."""
        raise NotImplementedError

    def transforms(self):
        """Data augmentation."""
        if not self.dataset:
            raise ValueError("dataset is None")

        trans = self.transform if self.transform else self.default_transform()

        self.dataset = self.dataset.map(operations=trans,
                                        input_columns='image',
                                        num_parallel_workers=self.num_parallel_workers)
        if self.target_transform:
            self.dataset = self.dataset.map(operations=self.target_transform,
                                            input_columns='label',
                                            num_parallel_workers=self.num_parallel_workers)

    def run(self):
        """Dataset pipeline."""
        self.transforms()
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = self.dataset.repeat(self.repeat_num)

        return self.dataset


class ParseDataset(metaclass=ABCMeta):
    """
    Parse dataset.
    """

    def __init__(self, path: str):
        self.path = os.path.expanduser(path)

    @abstractmethod
    def parse_dataset(self):
        """parse dataset from internet or compression file."""
