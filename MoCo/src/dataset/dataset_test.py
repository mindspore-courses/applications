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
""" Create the MoCo dataset."""

import os
import pickle
import numpy as np
from PIL import Image

from mindvision.dataset import Cifar10
import mindspore.dataset.vision.py_transforms as vp
import mindspore.dataset.vision.c_transforms as vc
import mindspore.dataset.transforms.c_transforms as c
from mindspore.dataset import GeneratorDataset


class CiFar10():
    """ training set or test set."""
    train_list = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
    ]

    test_list = [
        'test_batch',
    ]

    def __init__(self, root, train, transform=None, target_transform=None):

        self.root = root
        self.train = train

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2) where target is index of the target class.
        """
        img1, img2 = self.data[index], self.data[index]

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        return len(self.data)


def create_dataset(data_path):
    """ the MoCo dataset have dataset slice."""
    train_transform = c.Compose([
        vc.RandomResizedCrop(32),
        vc.RandomHorizontalFlip(0.5),
        vp.ToTensor(),
        vp.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    test_transform = c.Compose([
        vp.ToTensor(),
        vp.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    cifar10_test = CiFar10(root="Cifar10/cifar-10-batches-py", train=True, transform=train_transform)
    cifar10_test = GeneratorDataset(source=cifar10_test, column_names=["image1", "image2"])
    cifar10_test = cifar10_test.shuffle(buffer_size=2)
    train_data = cifar10_test.batch(batch_size=256, drop_remainder=True)

    dataset2 = Cifar10(path=data_path, split="train", batch_size=256, resize=32, shuffle=False,
                       download=False, transform=train_transform)
    dataset3 = Cifar10(path=data_path, split="test", batch_size=256, resize=32, shuffle=False,
                       download=False, transform=test_transform)

    memory_data = dataset2.run()
    test_data = dataset3.run()

    return train_data, memory_data, test_data
