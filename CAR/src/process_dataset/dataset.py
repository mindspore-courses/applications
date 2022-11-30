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
""" Read dataset comment file """

import os
from typing import Optional, Callable

import numpy as np
from PIL import Image
import mindspore.dataset as ds
from mindspore.dataset import vision

def imread(image, mode=None):
    """
    Read an image.

    Args:
        image (ndarray or str or Path): Ndarry, str or pathlib.Path.
        mode (str): Image mode.

    Returns:
        ndarray: Loaded image array.
    """
    image = Image.open(image)
    if mode:
        image = np.array(image.convert(mode))

    return image
class Set5Test:
    """
    A customize dataset that reads the Set5 test image.

    Args:
        path (str): The root directory of the inference image.
        split (str): The dataset split, supports "Set5", "Set14", "BSDS100" and "Urban100".

    Raises:
        ValueError: If `split` is not "Set5", "Set14", "BSDS100" and "Urban100".

    Take the dataset as an example.

    .. code-block::
        .
        └── SR_testing_datasets
             ├── Set5
             |    ├── baby.png
             |    ├── bird.png
             |    ├── ...
             ├── Set14
             |    ├── baboon.png
             |    ├── barbara.png
             |    ├── ...
             ├── BSDS100
             |    ├── 101085.png
             |    ├── 101087.png
             |    ├── ...
             ├── Urban100
             |    ├── img_001.png
             |    ├── img_002.png
             |    ├── ...
    """

    def __init__(self, path: str, split: str):
        self.path = os.path.join(path, split)
        data = os.listdir(self.path)
        self.data_list = [os.path.join(self.path, idx) for idx in data]

    def __getitem__(self, index):
        """ Get a list of datasets """
        return imread(self.data_list[index], 'RGB')

    def __len__(self):
        """ Get the length of each line """
        return len(self.data_list)


class DIV2KHR:
    """
    A customize dataset that reads the DIV2K HR dataset.

    Args:
        path (str): The root directory of the DIV2K HR dataset.
        split (str): The dataset split, supports "train" and "valid".

    Raises:
        ValueError: If `split` is not 'train' or 'valid'.

    About DIV2K dataset:

    The DIV2K dataset consists of 1000 2K resolution images, among which 800 images
    are for training, 100 images are for validation and 100 images are for testing.
    NTIRE 2017 and NTIRE 2018 include only training dataset and validation dataset.
    You can unzip the dataset files into the following directory structure

    Take the dataset as an example.

    .. code-block::
        .
        └── DIV2K
             ├── DIV2K_train_HR
             |    ├── 0001.png
             |    ├── 0002.png
             |    ├── ...
             ├── DIV2K_valid_HR
             |    ├── 000801.png
             |    ├── 000802.png
             |    ├── ...
    """

    def __init__(self, path: str, split: str):
        self.path = os.path.join(path, f"DIV2K_{split}_HR")
        data = os.listdir(self.path)
        self.data_list = [os.path.join(self.path, idx) for idx in data]

    def __getitem__(self, index):
        """ Get a list of datasets """
        return imread(self.data_list[index], 'RGB')

    def __len__(self):
        """ Get the length of each line """
        return len(self.data_list)


def default_transform(image):
    """
    Align the height and width of image and transform to Tensor.

    Args:
        image (Union[class:`PIL.Image.Image`, class:`numpy.ndarray`]): Input original image.

    Returns:
        Tensor: The image is aligned.
    """
    image = np.asarray(image)
    height, width, _ = image.shape
    image = image[:height // 8 * 8, :width // 8 * 8, :]
    image = vision.ToTensor()(image)

    return image


def build_dataset(dataset,
                  batch_size: int = 1,
                  repeat_num: int = 1,
                  shuffle: Optional[bool] = False,
                  num_parallel_workers: Optional[int] = 1,
                  num_shards: Optional[int] = None,
                  shard_id: Optional[int] = None,
                  transform: Optional[Callable] = default_transform):
    """
    Create datasets.

    Args:
        batch_size (int): The batch size of dataset. Default: 1.
        repeat_num (int): The repeat num of dataset. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default: False.
        num_parallel_workers (int, optional): The number of subprocess used to fetch the dataset
            in parallel. Default: None.
        num_shards (int, optional): The number of shards that the dataset will be divided. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        transform (callable, optional): A function transform that takes in a image. Default: default_transform.

    Returns:
        GeneratorDataset.
    """
    if shard_id is not None:
        sampler = ds.DistributedSampler(num_shards, shard_id, shuffle)
        dataset = ds.GeneratorDataset(dataset, ['image'],
                                      sampler=sampler,
                                      num_parallel_workers=num_parallel_workers)
    else:
        dataset = ds.GeneratorDataset(dataset, ['image'],
                                      num_parallel_workers=num_parallel_workers,
                                      shuffle=shuffle)

    if transform:
        dataset = dataset.map(operations=transform,
                              input_columns='image',
                              num_parallel_workers=num_parallel_workers)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(repeat_num)

    return dataset
