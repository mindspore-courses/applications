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
"""Test MindSpore dataset"""

from datasets.scannet import ScanNetDataset
from datasets.scannet_ms import ScanNetDataset as ScanNetDataset_ms
from datasets import transforms, transforms_ms
import numpy as np

transform_list = transforms.Compose([
    transforms.ResizeImage((640, 480)),
    transforms.ToTensor(),
    transforms.RandomTransformSpace(
        [96, 96, 96], 0.04, False, False,
        0, 0, max_epoch=991),
    transforms.IntrinsicsPoseToProjection(9, 4),
])

transform_list_ms = transforms_ms.Compose([
    transforms_ms.ResizeImage((640, 480)),
    transforms_ms.ToTensor(),
    transforms_ms.RandomTransformSpace(
        [96, 96, 96], 0.04, False, False,
        0, 0, max_epoch=991),
    transforms_ms.IntrinsicsPoseToProjection(9, 4),
])

test_dataset = ScanNetDataset("data/scannet", "test", transform_list, 9, len([0, 0, 0]) - 1)
test_dataset_ms = ScanNetDataset_ms("data/scannet", "test", transform_list_ms, 9, len([0, 0, 0]) - 1)

print(np.sum((test_dataset[0]['imgs'].numpy() - test_dataset_ms[0]['imgs']) > 1e-4))
