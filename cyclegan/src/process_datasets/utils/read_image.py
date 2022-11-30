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
"""Read image and return list."""

import os


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff']

def is_image_file(filename):
    """
    Judge whether it is a picture.

    Args:
        filename(str): Name of picture
    """

    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir_path, max_dataset_size=float("inf")):
    """
    This will return image list in dir.

    Args:
        dir_path(str): The path of dir of picture.
        max_dataset_size(int): The max size of dataset. Default: float("inf")
    """

    images = []
    assert os.path.isdir(dir_path), '%s is not a valid directory' % dir_path

    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
