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
"""Preprocess Pix2Pix process_datasets."""

import os
import numpy as np
from PIL import Image

import mindspore
from mindspore import dataset as ds
import mindspore.dataset.vision.c_transforms as transforms


class Pix2PixDataset():
    """
    Define train process_datasets.

    Args:
        root_dir(str): train dataset path.
        config (class): Option class.

    Outputs:
        a_crop. crop image a.
        b_crop. crop image b.
    """

    def __init__(self, root_dir, config):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.load_size = config.load_size
        self.train_pic_size = config.train_pic_size

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        ab = Image.open(img_path).convert('RGB')
        w, h = ab.size
        w2 = int(w / 2)

        a = ab.crop((w2, 0, w, h))
        b = ab.crop((0, 0, w2, h))

        a = a.resize((self.load_size, self.load_size))
        b = b.resize((self.load_size, self.load_size))

        transform_params = get_params(self.load_size, self.train_pic_size)
        a_crop = crop(a, transform_params, self.load_size, self.train_pic_size)
        b_crop = crop(b, transform_params, self.load_size, self.train_pic_size)

        return a_crop, b_crop


def get_params(load_size, train_pic_size):
    """
    Get parameters from images.

    Args:
        load_size (int): Scale images to this size.
        train_pic_size (int): The train image size.

    Return:
        x,y. get image size information.
    """

    new_h = new_w = load_size  # config.load_size

    x = np.random.randint(0, np.maximum(0, new_w - train_pic_size))
    y = np.random.randint(0, np.maximum(0, new_h - train_pic_size))

    return x, y

def crop(img, pos, load_size, train_pic_size):
    """
    Crop the images.

    Args:
        img (list): image.
        pos (int): crop position.
        load_size (int): Scale images to this size.
        train_pic_size (int): The train image size.

    Return:
        img. output img.
    """

    ow = oh = load_size
    x1, y1 = pos
    tw = th = train_pic_size
    if ow > tw or oh > th:
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        return img
    return img

def sync_random_horizontal_flip(input_images, target_images):
    """
    Randomly flip the input images and the target images.

    Args:
        input_images (list): input original image.
        target_images (list): output image after random horizontal flip.

   Return:
        out_input: random horizontal flip image.
        out_target: random horizontal flip image.
    """

    seed = np.random.randint(0, 2000000000)
    mindspore.set_seed(seed)
    op = transforms.RandomHorizontalFlip(prob=0.5)
    out_input = op(input_images)
    mindspore.set_seed(seed)
    op = transforms.RandomHorizontalFlip(prob=0.5)
    out_target = op(target_images)
    return out_input, out_target


def create_train_dataset(dataset, batch_size):
    """
    Create train process_datasets.

    Args:
        dataset (Class): image processed dataset.
        batch_size (int): train dataset size.

    Return:
        train dataset parameter.
    """

    mean = [0.5 * 255] * 3
    std = [0.5 * 255] * 3

    trans = [
        transforms.Normalize(mean=mean, std=std),
        transforms.HWC2CHW()
    ]

    train_ds = ds.GeneratorDataset(dataset, column_names=["input_images", "target_images"], shuffle=False)

    train_ds = train_ds.map(operations=[sync_random_horizontal_flip], input_columns=["input_images", "target_images"])

    train_ds = train_ds.map(operations=trans, input_columns=["input_images"])
    train_ds = train_ds.map(operations=trans, input_columns=["target_images"])

    train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)

    return train_ds


class Pix2PixDatasetVal():
    """
    Define val process_datasets.

    Args:
        root_dir (str): eval dataset path.

    Outputs:
        a. crop image a.
        b. crop image b.

    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)

        ab = Image.open(img_path).convert('RGB')
        w, h = ab.size

        w2 = int(w / 2)
        a = ab.crop((w2, 0, w, h))
        b = ab.crop((0, 0, w2, h))

        return a, b


def create_val_dataset(dataset, size):
    """
    Create val process_datasets.

    Args:
        dataset (Class): image processed eval dataset.
        size (int): eval dataset size.

    Return:
        eval dataset parameter.
    """

    mean = [0.5 * 255] * 3
    std = [0.5 * 255] * 3

    trans = [
        transforms.Resize((size, size)),
        transforms.Normalize(mean=mean, std=std),
        transforms.HWC2CHW()
    ]

    val_ds = ds.GeneratorDataset(dataset, column_names=["input_images", "target_images"], shuffle=False)

    val_ds = val_ds.map(operations=trans, input_columns=["input_images"])
    val_ds = val_ds.map(operations=trans, input_columns=["target_images"])
    val_ds = val_ds.batch(1, drop_remainder=True)

    return val_ds
