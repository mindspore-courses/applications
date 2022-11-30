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
""" The generator dataset. """


import numpy as np
from utils.images import imread


class DatasetGenerator:
    """ Dataset generator for getting image path and its corresponding label. """

    def __init__(self, image, label, image_id=None, mode=None):
        self.image = image
        self.label = label
        self.image_id = image_id
        self.mode = mode

    def __getitem__(self, item):
        """Get the image and label for each item."""
        if isinstance(self.image, list):
            image = imread(self.image[item], self.mode) if self.mode else np.fromfile(self.image[item], dtype="int8")
        else:
            image = self.image[item]

        label = self.label[item]

        if self.image_id:
            image_id = self.image_id[item]
            return image, image_id, label

        return image, label

    def __len__(self):
        """Get the the size of dataset."""
        return len(self.image)
