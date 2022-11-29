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
"""The dataset interface of Transformer."""

import os
import random
from random import randrange
from PIL import Image

import numpy as np
import mindspore.dataset as ds

from transformer_utils.util import generate_stroke_mask


class ImageNetDatasetMask:
    """
    This dataset class can load datasets.

    Args:
        pt_dataset (str): The path of image.
        clusters (str): The path of kmeans clusters.
        mask_path (str): The path of mask.
        perm (np.array): Reshuffle pixels with any fixed permutation. Default: None
        is_train (bool): Whether the training phase. Default: False
        use_imagefolder (int): use_imagefolder for ImageNet. Default: False
        prior_size (str): The size of origin image. Default: 32
        random_stroke (bool): Whether to generate masks randomly. Default: False

    Examples:
        >>> from datasets.dataset import ImageNetDatasetMask
        >>> ImageNetDatasetMask(data_path, kmeans, mask_path=mask_path, is_train=is_train,
                                use_imagefolder=use_imagefolder, prior_size=prior_size, random_stroke=random_stroke)
    """

    def __init__(self, pt_dataset, clusters, mask_path, perm=None, is_train=False, use_imagefolder=False,
                 prior_size=32, random_stroke=False):

        self.is_train = is_train
        self.pt_dataset = pt_dataset
        self.image_id_list = []
        if not use_imagefolder:
            temp_list = getfilelist(pt_dataset)
            for x in temp_list:
                if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.JPG') or x.endswith('.PNG'):
                    self.image_id_list.append(x)
        else:
            for dirpath, _, filenames in os.walk(self.pt_dataset):
                for filename in filenames:
                    self.image_id_list.append(os.path.join(dirpath, filename))

        self.random_stroke = random_stroke
        self.clusters = clusters
        self.perm = np.arange(prior_size * prior_size) if perm is None else perm

        self.mask_dir = mask_path
        self.mask_list = os.listdir(self.mask_dir)
        self.mask_list = sorted(self.mask_list)

        self.vocab_size = clusters.shape[0]
        self.block_size = prior_size * prior_size
        self.mask_num = len(self.mask_list)

        self.prior_size = prior_size

        print("# Mask is %d, # Image is %d" % (self.mask_num, len(self.image_id_list)))

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        if not self.random_stroke:
            if self.is_train:
                selected_mask_name = random.sample(self.mask_list, 1)[0]
            else:
                selected_mask_name = self.mask_list[idx % self.mask_num]
            selected_mask_dir = os.path.join(self.mask_dir, selected_mask_name)
            mask = Image.open(selected_mask_dir).convert("L")
        else:
            mask = generate_stroke_mask([256, 256])
            mask = (mask > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask).convert("L")

        mask = mask.resize((self.prior_size, self.prior_size), resample=Image.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        mask = np.array(mask).reshape(-1)
        mask = (mask / 255.) > 0.5

        selected_img_name = self.image_id_list[idx]
        selected_img_url = selected_img_name
        x = read_img(selected_img_url, image_size=self.prior_size, is_train=self.is_train)

        x = np.array(x).reshape(-1, 3)
        x = x[self.perm]
        a = ((x[:, None, :] - self.clusters[None, :, :]) ** 2).sum(-1).argmin(1)
        return a[:], mask[:]


def read_img(img_url, image_size, is_train):
    """
    Load the image and return image np array.

    Args:
        img_url (str): The path of image.
        image_size (int): The size of image.
        is_train (bool): Whether the training phase.
    """
    img = Image.open(img_url).convert("RGB")
    x, y = img.size
    if x != y:
        if is_train:
            matrix_length = min(x, y)
            x1 = randrange(0, x - matrix_length + 1)
            y1 = randrange(0, y - matrix_length + 1)
            img = img.crop((x1, y1, x1 + matrix_length, y1 + matrix_length))

    if random.random() > 0.5 and is_train:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.array(img)


def getfilelist(path):
    """
    Get image list from file path

    Args:
        path (str): The file path.
    """
    all_file = []
    for root, _, file in os.walk(path):
        for i in file:
            t = "%s/%s" % (root, i)
            all_file.append(t)
        return all_file


def load_dataset(data_path, kmeans, mask_path=None, is_train=False, use_imagefolder=False, prior_size=32,
                 random_stroke=False, rank_id=0, rank_size=1):
    train_dataset = ds.GeneratorDataset(
        ImageNetDatasetMask(data_path, kmeans, mask_path=mask_path, is_train=is_train, use_imagefolder=use_imagefolder,
                            prior_size=prior_size, random_stroke=random_stroke),
        ["data", "mask"], num_shards=rank_size, shard_id=rank_id
    )
    return train_dataset
