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
# ==============================================================================
"""Set the inferring  transform for ImageNet dataset."""

import math
from mindspore.dataset.transforms import Compose
import mindspore.dataset.vision.c_transforms as c_transforms
import mindspore.dataset.vision.py_transforms as p_transforms


def infer_transform(dataset, columns_list, resize):
    """
    Implements validation transformation method (Resize --> CenterCrop --> ToTensor).
    """

    crop_ratio = 0.875
    scale_size = int(math.ceil(resize / crop_ratio))
    scale_size = (scale_size // 32) * 32

    img_transforms = Compose([
        c_transforms.Decode(),
        c_transforms.Resize([scale_size, scale_size]),
        c_transforms.CenterCrop(resize),
        c_transforms.ConvertColor(c_transforms.ConvertMode.COLOR_RGB2BGR),
        c_transforms.RandomHorizontalFlip(),
        p_transforms.ToTensor(),
    ])

    dataset = dataset.map(operations=img_transforms,
                          input_columns=columns_list[0],
                          num_parallel_workers=1)
    dataset = dataset.batch(100)
    return dataset
