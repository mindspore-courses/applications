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
"""Create CRA image dataset."""

import os
import cv2


class InpaintDataset():
    """
    Read the training image from the given path and preprocess it.

    Args:
        path(str): training image dataset path.
        args (class): option class.

    Return:
        img: Preprocessed training image.
    """

    def __init__(self, args):
        self.args = args
        self.imglist = self.get_files(args.image_dir)

    def get_files(self, path):
        ret = []
        for tuple_path in os.walk(path):
            for filespath in tuple_path[2]:
                ret.append(os.path.join(tuple_path[0], filespath))
        return ret

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        img = cv2.imread(self.imglist[index])
        h, w = self.args.IMG_SHAPE[0], self.args.IMG_SHAPE[1]
        img = cv2.resize(img, (h, w))
        img = img / 127.5 - 1
        img = img.transpose((2, 0, 1))
        return img
