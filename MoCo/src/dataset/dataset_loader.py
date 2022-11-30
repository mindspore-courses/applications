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

from mindvision.dataset import Cifar10
import mindspore.dataset.vision.py_transforms as vp
import mindspore.dataset.vision.c_transforms as vc
import mindspore.dataset.transforms.c_transforms as c


def create_dataset(data_path):
    """ the MoCo dataset is no dataset slice."""
    train_transform = c.Compose([
        vc.RandomResizedCrop(32),
        vc.RandomHorizontalFlip(0.5),
        vp.ToTensor(),
        vp.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    test_transform = c.Compose([
        vp.ToTensor(),
        vp.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    dataset1 = Cifar10(path=data_path, split="train", batch_size=256, resize=32, shuffle=True,
                       download=False, transform=train_transform)
    dataset2 = Cifar10(path=data_path, split="train", batch_size=256, resize=32, shuffle=False,
                       download=False, transform=train_transform)
    dataset3 = Cifar10(path=data_path, split="test", batch_size=256, resize=32, shuffle=False,
                       download=False, transform=test_transform)

    train_data = dataset1.run()
    memory_data = dataset2.run()
    test_data = dataset3.run()

    return train_data, memory_data, test_data
