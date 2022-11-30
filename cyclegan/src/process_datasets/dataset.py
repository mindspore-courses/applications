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
"""Create Cycle GAN dataset."""

import os
import random
import multiprocessing
import numpy as np
from PIL import Image

import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C

from .utils.read_image import make_dataset
from .distributedsampler import DistributedSampler


class UnalignedDataset:
    """
    This dataset class can load unaligned/unpaired datasets.

    Args:
        dataroot (str): The root of image data.
        phase (str): The dataset split, supports "train", or "predict". Default: "train".
        max_dataset_size (int): Max size of dataset. Default: float("inf").
        use_random (bool): A function transform that takes in a label. Default: True.

    Raises:
        ValueError: If `phase` is not 'train', 'predict'.

    Examples:
        >>> from src.process_datasets import UnalignedDataset
        >>> UnalignedDataset(dataroot, phase, max_dataset_size=max_dataset_size, use_random=args.use_random)
    """

    def __init__(self, dataroot, phase, max_dataset_size=float("inf"), use_random=True):
        self.dir_a = os.path.join(dataroot, phase + 'A')
        self.dir_b = os.path.join(dataroot, phase + 'B')

        self.a_paths = sorted(make_dataset(self.dir_a, max_dataset_size))   # load images from '/path/to/data/trainA'
        self.b_paths = sorted(make_dataset(self.dir_b, max_dataset_size))    # load images from '/path/to/data/trainB'
        self.a_size = len(self.a_paths)  # get the size of dataset A
        self.b_size = len(self.b_paths)  # get the size of dataset B
        self.use_random = use_random

    def __getitem__(self, index):
        """
        This will return a data point and its metadata information.

        Args:
            index (int): A random integer for data indexing.

        Returns:
            a_img(array), b_img(array).
        """

        index_b = index % self.b_size
        if index % max(self.a_size, self.b_size) == 0 and self.use_random:
            random.shuffle(self.a_paths)
            index_b = random.randint(0, self.b_size - 1)
        a_path = self.a_paths[index % self.a_size]
        b_path = self.b_paths[index_b]
        a_img = np.array(Image.open(a_path).convert('RGB'))
        b_img = np.array(Image.open(b_path).convert('RGB'))

        return a_img, b_img

    def __len__(self):
        """
        This will return the total number of images in the dataset.
        """
        return max(self.a_size, self.b_size)


class ImageFolderDataset:
    """
    This dataset class can load images from image folder.

    Args:
        dataroot (str): Images root directory.
        max_dataset_size (int): Maximum number of return image paths.

    Examples:
        >>> from src.process_datasets.cyclegan_dataset import ImageFolderDataset
        >>> ImageFolderDataset(datadir, max_dataset_size=max_dataset_size)
    """

    def __init__(self, dataroot, max_dataset_size=float("inf")):
        self.dataroot = dataroot
        self.paths = sorted(make_dataset(dataroot, max_dataset_size))
        self.size = len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index % self.size]
        img = np.array(Image.open(img_path).convert('RGB'))

        return img, os.path.split(img_path)[1]

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return self.size


def create_dataset(args):
    """
    Use this to create dataset.
    This dataset class can load images for train or test.

    Args:
        dataroot (str): Images root directory.
        phase(str): Load train of test image.
        batch_size(int): Batch_size of image dataset.
        device_num(int): Device number of device.
        rank(int): Rank of the device.
        shuffle(bool): If use random.
        max_dataset_size(int): Max size of dataset.
        cores(int, optional): Cores of computer.
        num_parallel_workers(int, optional): The number of subprocess used to fetch the dataset.
        image_size(int): Size to resize image dataset.
        mean(Union[int, tuple]): args to resize images. Default: [0.5 * 255] * 3.
        std(Union[int, tuple]): args to resize images. Default: [0.5 * 255] * 3.

    Examples:
        >>> from src.process_datasets.cyclegan_dataset import create_dataset
        >>> create_dataset(args)

    Returns:
        dataset.
    """

    dataroot = args.dataroot
    phase = args.phase
    batch_size = args.batch_size
    device_num = args.device_num
    rank = args.rank
    shuffle = args.use_random
    max_dataset_size = args.max_dataset_size
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(8, int(cores / device_num))
    image_size = args.image_size
    mean = [0.5 * 255] * 3
    std = [0.5 * 255] * 3
    if phase == "train":
        dataset = UnalignedDataset(dataroot, phase, max_dataset_size=max_dataset_size, use_random=args.use_random)
        distributed_sampler = DistributedSampler(len(dataset), device_num, rank, shuffle=shuffle)
        ds = de.GeneratorDataset(dataset, column_names=["image_A", "image_B"],
                                 sampler=distributed_sampler, num_parallel_workers=num_parallel_workers)
        if args.use_random:
            trans = [
                C.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                C.RandomHorizontalFlip(prob=0.5),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW()
            ]
        else:
            trans = [
                C.Resize((image_size, image_size)),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW()
            ]
        ds = ds.map(operations=trans, input_columns=["image_A"], num_parallel_workers=num_parallel_workers)
        ds = ds.map(operations=trans, input_columns=["image_B"], num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        datadir = os.path.join(dataroot, args.data_dir)
        dataset = ImageFolderDataset(datadir, max_dataset_size=max_dataset_size)
        ds = de.GeneratorDataset(dataset, column_names=["image", "image_name"],
                                 num_parallel_workers=num_parallel_workers)
        trans = [
            C.Resize((image_size, image_size)),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
        ds = ds.map(operations=trans, input_columns=["image"], num_parallel_workers=num_parallel_workers)
        ds = ds.batch(1, drop_remainder=True)
    args.dataset_size = len(dataset)
    return ds
    