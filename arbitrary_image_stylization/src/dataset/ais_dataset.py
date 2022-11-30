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
"""Create dataset."""
import os

import numpy as np
import mindspore.dataset.vision.py_transforms as transforms
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size

class AisDataset:
    """
    Load images for Ais model.
    Content images are from ImageNet 2012 and style images are from DTD dataset.

    The generated dataset has two columns :py:obj:`[content, style]`
    The tensor of column :py:obj:`content` is a matrix of the float32 type.
    The tensor of column :py:obj:`style` is a matrix of float32 type.

    Args:
        content_path (str): Path of content images.
        style_path (str): Path of style images.

    Examples:
        >>> content_path = './data/imagenet/train/'
        >>> style_path = './data/dtd/dtd/images/'
        >>> dataset = AisDataset(content_path, style_path)

    About IMAGENET dataset:

    IMAGENET is an image dataset that spans 1000 object classes and contains 1,281,167 training images,
    50,000 validation images and 100,000 test images. Images of each object are quality-controlled and
    human-annotated.

    You can unzip the dataset files into this directory structure and read them by MindSpore Vision's API.

    .. code-block::

        .imagenet/
        ├── train/  (1000 directories and 1281167 images)
        │  ├── n04347754/
        │  │   ├── 000001.jpg
        │  │   ├── 000002.jpg
        │  │   └── ....
        │  └── n04347756/
        │      ├── 000001.jpg
        │      ├── 000002.jpg
        │      └── ....
        └── val/   (1000 directories and 50000 images)
        ├── n04347754/
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   └── ....
        └── n04347756/
            ├── 000001.jpg
            ├── 000002.jpg
            └── ....

    About DTD:

    The Describable Textures Dataset(DTD) contains 5640 texture images organized according to a list of 47
    terms (categories) inspired from human perception. There are 120 images for each category.

    You can unzip the dataset files into this directory structure and read them by MindSpore Vision's API.

    .. code-block::

        .images/ (47 directories and 5640 images)
        ├── banded/
        │   ├── banded_0002.jpg
        │   ├── banded_0004.jpg
        │   └── ....
        └── blotchy/
            ├── blotchy_0003.jpg
            ├── blotchy_0006.jpg
            └── ....
    """
    def __init__(self, content_path, style_path):
        super(AisDataset, self).__init__()
        self.content_paths = self.read_file_list(content_path)
        self.style_paths = self.read_file_list(style_path)
        self.content_size = len(self.content_paths)
        self.style_size = len(self.style_paths)

    def __getitem__(self, index):
        index1 = index % self.content_size
        index2 = index % self.style_size
        content_path = self.content_paths[index1]
        style_path = self.style_paths[index2]
        content_image = np.fromfile(content_path, dtype='int8')
        style_image = np.fromfile(style_path, dtype='int8')
        return content_image, style_image

    def read_file_list(self, path):
        """Read each image and its corresponding label from directory."""

        file_type = ['jpg', 'JPEG']
        images_path = []
        label = sorted(i.name for i in os.scandir(path) if i.is_dir())
        for item in label:
            for file_name in os.listdir(os.path.join(path, item)):
                if file_name.split('.')[-1] in file_type:
                    images_path.append(os.path.join(path, item, file_name))
        return images_path

    def __len__(self):
        return max(self.content_size, self.style_size)

def create_dataset(args):
    """
    Create dataset.

    Args:
        content_path (str): Path of content images.
        style_path (str): Path of style images.
        size (int): Size of both content and style images will be resized.
        shuffle (bool): Shuffle data or not.
        batch_size (int): Number of images of loaded at one time.

    Examples:
        >>> dataset = create_dataset(args)
        >>> dataloader = dataset.create_dict_iterator()

    Returns:
        Dataset.
    """
    if args.parallel == 0:
        dataset = ds.GeneratorDataset(
            source=AisDataset(args.content_path, args.style_path),
            column_names=['content', 'style'],
            num_parallel_workers=args.num_workers,
            shuffle=bool(args.shuffle)
        )
    else:
        rank_id = get_rank()
        rank_size = get_group_size()
        dataset = ds.GeneratorDataset(
            source=AisDataset(args.content_path, args.style_path),
            column_names=['content', 'style'],
            num_parallel_workers=args.num_workers,
            shuffle=bool(args.shuffle),
            num_shards=rank_size,
            shard_id=rank_id
        )
    scale = 32
    trans_c = [
        transforms.Decode(),
        transforms.Resize(args.size + scale),
        transforms.CenterCrop(args.size),
        transforms.RandomHorizontalFlip(prob=0.5),
        transforms.ToTensor()
    ]
    trans_s = [
        transforms.Decode(),
        transforms.RandomColorAdjust(brightness=0.8, saturation=0.5, hue=0.2),
        transforms.RandomHorizontalFlip(prob=0.5),
        transforms.RandomVerticalFlip(prob=0.5),
        transforms.Resize(args.size + scale),
        transforms.RandomCrop(args.size),
        transforms.ToTensor()
    ]
    dataset = dataset.map(operations=trans_c, input_columns='content', num_parallel_workers=args.num_workers)
    dataset = dataset.map(operations=trans_s, input_columns='style', num_parallel_workers=args.num_workers)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    return dataset
