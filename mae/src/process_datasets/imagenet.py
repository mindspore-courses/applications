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
"""
Create train or eval dataset in imagenet.
"""

import os
import json

from PIL import Image

import mindspore.dataset as de
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as C


class DataLoader:
    """
    DataLoader.

    Args:
        images_path (str): image path.
    """
    def __init__(self, images_path, data_dir=None):
        images_path = os.path.join(data_dir, images_path)
        assert os.path.realpath(images_path), "imgs_path should be real path."
        with open(images_path, 'r') as f:
            data = json.load(f)
        if data_dir is not None:
            data = [os.path.join(data_dir, item) for item in data]
        self.data = data

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        return (img,)

    def __len__(self):
        return len(self.data)


def create_dataset(args):
    """
    Create dataset of Image Net.

    Returns:
        Tensor, output tensor, the shape is (batch_size * 3).
    """
    # mean and variance of imagenet
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # Create dataset.
    dataset = de.GeneratorDataset(
        source=DataLoader(args.img_ids, data_dir=args.data_path),
        column_names="image", num_shards=args.device_num,
        shard_id=args.local_rank, shuffle=True)

    # The specific operation of transform.
    trans = [
        C.RandomResizedCrop(
            args.image_size,
            scale=(0.2, 1.0),
            ratio=(0.75, 1.333),
            interpolation=Inter.BICUBIC
        ),
        C.RandomHorizontalFlip(prob=0.5),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW(),
    ]
    ds = dataset.map(input_columns="image", num_parallel_workers=args.num_workers,
                     operations=trans, python_multiprocessing=True)
    ds = ds.shuffle(buffer_size=10)
    ds = ds.batch(args.batch_size, drop_remainder=True)
    ds = ds.repeat(1)
    return ds


def create_dataset_img_ids():
    """
        Create picture index for dataset of Image Net.

        Returns:
            None.
    """
    dataset_dir = "../imagenet2012/train"
    dic = {}
    for item in os.listdir(dataset_dir):
        for file in os.listdir(os.path.join(dataset_dir, item)):
            if str(file).find('.JPEG') > -1:
                dic['train/' + str(item) + '/' + str(file)] = str(item)
    json.dump(dic, open('../imagenet2012/tot_ids.json', 'w'), indent=4)


if __name__ == '__main__':
    create_dataset_img_ids()
