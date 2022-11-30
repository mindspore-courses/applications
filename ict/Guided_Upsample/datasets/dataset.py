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
"""The dataset interface of Upsample."""

import os
import random
from random import randrange
from PIL import Image

import numpy as np
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset as ds

from upsample_utils.util import create_mask
from upsample_utils.degradation import prior_degradation, prior_degradation_2


class Dataset:
    """
    This dataset class can load datasets.

    Args:
        image_flist (str): The path of image.
        edge_flist (str): The path of edge(prior).
        mask_flist (str): The path of mask.
        image_size (int): The size of origin image.
        prior_size (int): The size of prior image from transformer.
        mask_type (int): The type of generator mask.
        kmeans (str): The path of kmeans_centers.npy
        use_degradation_2 (bool): Use the new degradation function.
        prior_random_degree (int): During training, how far deviate from.
        condition_num (int): Use how many BERT output(number of samples).
        augment (bool): The image augment.
        training (bool): The mode of training.

    Examples:
        >>> from datasets.dataset import Dataset
        >>> Dataset(image_flist, edge_flist, mask_filst, image_size, prior_size, mask_type, kmeans, augment,
        use_degradation_2, prior_random_degree, condition_num, augment, training)
    """

    def __init__(self, image_flist: str, edge_flist: str, mask_flist: str, image_size: int, prior_size: int,
                 mask_type: int, kmeans: str,
                 use_degradation_2: bool, prior_random_degree: int = 1, condition_num: int = 1, augment: bool = True,
                 training: bool = True):
        self.image_size = image_size
        self.mask_type = mask_type
        self.prior_size = prior_size
        self.augment = augment
        self.training = training
        self.use_degradation_2 = use_degradation_2
        self.prior_random_degree = prior_random_degree
        self.clusters = np.load(kmeans)
        self.image_data = self.load_flist(image_flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        # During testing, load the transformer prior.
        if not training:
            all_data = []
            all_edge_data = []
            all_mask_data = []
            for i, x in enumerate(self.image_data):
                for j in range(condition_num):
                    temp = 'condition_%d/%s' % (j + 1, os.path.basename(x))
                    all_data.append(x)
                    all_mask_data.append(self.mask_data[i])
                    all_edge_data.append(os.path.join(edge_flist, temp))
            self.image_data = all_data
            self.edge_data = all_edge_data
            self.mask_data = all_mask_data

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except FileNotFoundError:
            print('loading error: ' + self.image_data[index])
            item = self.load_item(0)
        return item

    def load_name(self, index):
        name = self.image_data[index]
        return os.path.basename(name)

    def load_item(self, index):
        """
        Load the image item include name, image, prior and mask.

        Args:
            index (int): The item index.
        """
        size = self.image_size

        # load image
        img = Image.open(self.image_data[index]).convert("RGB")
        img = np.array(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # load mask
        mask = self.load_mask(img, index)

        # load edge
        prior = self.load_prior(img, index)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            prior = prior[:, ::-1, ...]

        if self.augment and np.random.binomial(1, 0.5) > 0:
            mask = mask[:, ::-1, ...]
        if self.augment and np.random.binomial(1, 0.5) > 0:
            mask = mask[::-1, :, ...]

        return self.load_name(index), self.to_tensor(img), self.to_tensor(prior), self.to_tensor(mask)

    def load_prior(self, img, index):
        """
        Load the image prior.

        Args:
            img (array): The image numpy array.
            index (int): The item index.
        """
        if self.training:
            # Training, prior_degradation
            imgh, imgw = img.shape[0:2]
            x = Image.fromarray(img).convert("RGB")
            if self.use_degradation_2:
                prior_lr = prior_degradation_2(x, self.clusters, self.prior_size, self.prior_random_degree)
            else:
                prior_lr = prior_degradation(x, self.clusters, self.prior_size)
            prior_lr = np.array(prior_lr).astype('uint8')
            prior_lr = self.resize(prior_lr, imgh, imgw)
            return prior_lr

        # external, from transformer
        imgh, imgw = img.shape[0:2]
        edge = Image.open(self.edge_data[index]).convert("RGB")
        edge = np.array(edge)
        edge = self.resize(edge, imgh, imgw)
        return edge

    def load_mask(self, img, index):
        """
        Load the image mask.

        Args:
            img (array): The image numpy array.
            index (int): The item index.
        """
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask_type
        if mask_type == 1:
            mask = create_mask(imgw, imgh, imgw // 2, imgh // 2)
            return mask
        if mask_type == 2:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = Image.open(self.mask_data[mask_index]).convert("RGB")
            mask = np.array(mask)
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        mask = Image.open(self.mask_data[index]).convert("RGB")
        mask = np.array(mask)
        mask = self.resize(mask, imgh, imgw, center_crop=False)
        mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = py_vision.ToTensor()(img)
        return img_t

    def resize(self, img, height, width, center_crop=True):
        """
        Resize the image to the specified size.

        Args:
            img (array): The image numpy array.
            height (int): The image height.
            width (int): The image width.
            center_crop (bool): Use image crop.
        """
        imgh, imgw = img.shape[0:2]

        # While training, random crop with short side.
        if self.training:
            img = Image.fromarray(img)
            side = np.minimum(imgh, imgw)
            y1 = randrange(0, imgh - side + 1)
            x1 = randrange(0, imgw - side + 1)
            img = img.crop((x1, y1, x1 + side, y1 + side))
            img = np.array(img.resize((height, width), resample=Image.BICUBIC))
        else:
            if center_crop and imgh != imgw:
                # center crop
                side = np.minimum(imgh, imgw)
                j = (imgh - side) // 2
                i = (imgw - side) // 2
                img = img[j:j + side, i:i + side, ...]
            img = np.array(Image.fromarray(img).resize((height, width), resample=Image.BICUBIC))
        return img

    def load_flist(self, flist):
        """flist: image file path, image directory path, text file flist path"""
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = self.getfilelist(flist)
                flist.sort()
                return flist

            if os.path.isfile(flist):
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        return []

    def getfilelist(self, path):
        all_file = []
        for path_dir, _, path_file in os.walk(path):
            for i in path_file:
                t = "%s/%s" % (path_dir, i)
                if t.endswith('.png') or t.endswith('.jpg') or t.endswith('.JPG') or t.endswith('.PNG') or t.endswith(
                        '.JPEG'):
                    all_file.append(t)
        return all_file


def load_dataset(image_flist, edge_flist, mask_filst, image_size, prior_size, mask_type, kmeans,
                 use_degradation_2=False, prior_random_degree=1, condition_num=1, augment=True, training=True):
    train_dataset = ds.GeneratorDataset(
        Dataset(image_flist, edge_flist, mask_filst, image_size, prior_size, mask_type, kmeans, use_degradation_2,
                prior_random_degree, condition_num, augment, training),
        ["name", "images", "edges", "masks"]
    )
    return train_dataset
