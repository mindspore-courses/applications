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
"""Create SRGAN dataset."""

import os
import random

import numpy as np
from PIL import Image

__all__ = ["TrainDataset", "TestDataset"]

class TrainDataset:
    """
    Read the training data file in the specified format.

    Args:
        lr_path (str): The path of low-resolution image.
        gt_path (str): The path of high-resolution image.
        in_memory (bool): Choose whether to load the entire dataset into memory. Default: True.
    """
    def __init__(self, lr_path, gt_path, in_memory=True):
        """init"""
        self.lr_path = lr_path
        self.gt_path = gt_path
        self.in_memory = in_memory
        self.lr_img = sorted(os.listdir(lr_path))
        self.gt_img = sorted(os.listdir(gt_path))
        if in_memory:
            self.lr_img = [np.array(Image.open(os.path.join(self.lr_path, lr)).convert("RGB")).astype(np.float32)
                           for lr in self.lr_img]
            self.gt_img = [np.array(Image.open(os.path.join(self.gt_path, hr)).convert("RGB")).astype(np.float32)
                           for hr in self.gt_img]

    def __len__(self):
        """getlength"""
        return len(self.lr_img)

    def __getitem__(self, i):
        """getitem"""
        img_item = {}
        if self.in_memory:
            gt = self.gt_img[i].astype(np.float32)
            lr = self.lr_img[i].astype(np.float32)

        else:
            gt = np.array(Image.open(os.path.join(self.gt_path, self.gt_img[i])).convert("RGB"))
            lr = np.array(Image.open(os.path.join(self.lr_path, self.lr_img[i])).convert("RGB"))
        img_item['GT'] = (gt / 127.5) - 1.0
        img_item['LR'] = (lr / 127.5) - 1.0
        # crop
        ih, iw = img_item['LR'].shape[:2]
        ix = random.randrange(0, iw -24 + 1)
        iy = random.randrange(0, ih -24 + 1)
        tx = ix * 4
        ty = iy * 4
        img_item['LR'] = img_item['LR'][iy : iy + 24, ix : ix + 24]
        img_item['GT'] = img_item['GT'][ty : ty + (4 * 24), tx : tx + (4 * 24)]
        # augmentation
        hor_flip = random.randrange(0, 2)
        ver_flip = random.randrange(0, 2)
        rot = random.randrange(0, 2)
        if hor_flip:
            temp_lr = np.fliplr(img_item['LR'])
            img_item['LR'] = temp_lr.copy()
            temp_gt = np.fliplr(img_item['GT'])
            img_item['GT'] = temp_gt.copy()
            del temp_lr, temp_gt

        if ver_flip:
            temp_lr = np.flipud(img_item['LR'])
            img_item['LR'] = temp_lr.copy()
            temp_gt = np.flipud(img_item['GT'])
            img_item['GT'] = temp_gt.copy()
            del temp_lr, temp_gt

        if rot:
            img_item['LR'] = img_item['LR'].transpose(1, 0, 2)
            img_item['GT'] = img_item['GT'].transpose(1, 0, 2)
        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
        return  img_item['LR'], img_item['GT']

class TestDataset:
    """
    Read the testing data file in the specified format.

    Args:
        lr_path (str): The path of low-resolution image. Default: ''.
        gt_path (str): The path of high-resolution image.
        infer (bool): Choose inference mode or evaluation model. Default: False.
        in_memory (bool): Choose whether to load the entire dataset into memory. Default: True.
    """
    def __init__(self, lr_path, gt_path='', infer=False, in_memory=True):
        """init"""
        self.infer = infer
        self.lr_path = lr_path
        self.in_memory = in_memory
        self.lr_img = sorted(os.listdir(lr_path))
        if not infer:
            self.gt_img = sorted(os.listdir(gt_path))
            self.gt_path = gt_path
        if in_memory:
            self.lr_img = [np.array(Image.open(os.path.join(self.lr_path, lr)).convert("RGB")).astype(np.float32)
                           for lr in self.lr_img]
            if not infer:
                self.gt_img = [np.array(Image.open(os.path.join(self.gt_path, hr)).convert("RGB")).astype(np.float32)
                               for hr in self.gt_img]

    def __len__(self):
        """getlength"""
        return len(self.lr_img)

    def __getitem__(self, i):
        """getitem"""
        img_item = {}
        if not self.infer:
            if self.in_memory:
                lr = self.lr_img[i].astype(np.float32)
                gt = self.gt_img[i].astype(np.float32)
            else:
                gt = np.array(Image.open(os.path.join(self.gt_path, self.gt_img[i])).convert("RGB"))
                lr = np.array(Image.open(os.path.join(self.lr_path, self.lr_img[i])).convert("RGB"))
            img_item['GT'] = (gt / 127.5) - 1.0
            img_item['LR'] = (lr / 127.5) - 1.0
            img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)
            img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
            return  img_item['LR'], img_item['GT']
        if self.in_memory:
            lr = self.lr_img[i].astype(np.float32)
        else:
            lr = np.array(Image.open(os.path.join(self.lr_path, self.lr_img[i])).convert("RGB"))
        img_item['LR'] = (lr / 127.5) - 1.0
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
        return  img_item['LR']
