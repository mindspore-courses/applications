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
# ======================================================================
""" Read dataset comment file in the specified format. """

import numpy as np
import cv2


class Datasets300W():
    """
    Read the 300W comment file in the specified format.

    Args:
        file_dir: str. File path after processing (data enhancement).
        transforms: function. Function to convert a PIL image or a numpy.ndarray
            of shape (H, W, C) to a numpy.ndarray of type (C, H, W).
    """

    def __init__(self, file_dir, transforms=None):
        self.line = None
        self.img = None
        self.landmark = None
        self.attribute = None
        self.euler_angle = None
        self.transforms = transforms
        with open(file_dir, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        """ Get a list of datasets """
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(self.line[0])

        self.landmark = np.asarray(self.line[1:137], dtype=np.float32)
        self.attribute = np.asarray(self.line[137:143], dtype=np.int32)
        self.euler_angle = np.asarray(self.line[143:146], dtype=np.float32)

        if self.transforms:
            self.img = self.transforms(self.img)

        return [self.img, self.landmark, self.attribute, self.euler_angle]

    def __len__(self):
        """ Get the length of each line """
        return len(self.lines)


class DatasetsWFLW():
    """
    Read the WFLW comment file in the specified format.

    Args:
        file_dir: str. File path after processing (data enhancement).
        transforms: function. Function to convert a PIL image or a numpy.ndarray
            of shape (H, W, C) to a numpy.ndarray of type (C, H, W).
    """

    def __init__(self, file_dir, transforms=None):
        self.line = None
        self.img = None
        self.landmark = None
        self.attribute = None
        self.euler_angle = None
        self.transforms = transforms
        with open(file_dir, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        """ Get a list of datasets """
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(self.line[0])

        self.landmark = np.asarray(self.line[1:197], dtype=np.float32)
        self.attribute = np.asarray(self.line[197:203], dtype=np.int32)
        self.euler_angle = np.asarray(self.line[203:206], dtype=np.float32)

        if self.transforms:
            self.img = self.transforms(self.img)

        return [self.img, self.landmark, self.attribute, self.euler_angle]

    def __len__(self):
        """ Get the length of each line. """
        return len(self.lines)
