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
# ==========================================================================
""" Utils """

import numpy as np


def map_func(attribute_gt):
    """
    Process the attribute column and convert it into a form where the loss can be directly calculated.

    Args:
        attribute_gt (list): Attribute column in the dataset.
        batchInfo: Required parameter.

    Returns:
        weight_attribute. Weight used in data augmentation.
    """
    attribute_gt = np.array(attribute_gt)
    batch_size = len(attribute_gt)
    attributes_w_n = attribute_gt[:, 1:6].astype(np.float32)
    mat_ratio = np.mean(attributes_w_n, 0)
    mat_ratio = np.array([1.0 / x if x > 0 else batch_size for x in mat_ratio],
                         dtype='float32')
    weight_attribute = attributes_w_n.dot(mat_ratio)

    return (weight_attribute,)
