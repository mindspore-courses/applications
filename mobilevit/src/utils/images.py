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
""" Image io for read and write image. """

import os
import pathlib
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

from mindspore._checkparam import Validator

from utils import path
image_format = ('.JPEG', '.jpeg', '.PNG', '.png', '.JPG', '.jpg')
image_mode = ('1', 'L', 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV', 'I', 'F')


def imread(image, mode=None):
    """
    Read an image.

    Args:
        image (ndarray or str or Path): Ndarry, str or pathlib.Path.
        mode (str): Image mode.

    Returns:
        ndarray: Loaded image array.
    """
    Validator.check_string(mode, image_mode)

    if isinstance(image, pathlib.Path):
        image = str(image)

    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        path.check_file_exist(image)
        image = Image.open(image)
        if mode:
            image = np.array(image.convert(mode))
    else:
        raise TypeError("Image must be a `ndarray`, `str` or Path object.")

    return image


def imwrite(image, image_path, auto_mkdir=True):
    """
    Write image to file.

    Args:
        image (ndarray): Image array to be written.
        image_path (str): Image file path to be written.
        auto_mkdir (bool): `image_path` does not exist create it automatically.

    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(image_path))
        if dir_name != '':
            dir_name = os.path.expanduser(dir_name)
            os.makedirs(dir_name, mode=777, exist_ok=True)

    image = Image.fromarray(image)
    image.save(image_path)


def read_dataset(paths: str) -> Tuple[List[str], List[int]]:
    """
    Get the path list and index list of images.
    """
    img_list = list()
    id_list = list()

    idx = 0
    if os.path.isdir(paths):
        for img_name in os.listdir(paths):
            if pathlib.Path(img_name).suffix in image_format:
                img_path = os.path.join(paths, img_name)
                img_list.append(img_path)
                id_list.append(idx)
                idx += 1
    else:
        img_list.append(paths)
        id_list.append(idx)
    return img_list, id_list


def label2index(paths: str) -> Dict[str, int]:
    """
    Read images directory for getting label and its corresponding index.
    """
    label = sorted(i.name for i in os.scandir(paths) if i.is_dir())

    if not label:
        raise ValueError(f"Cannot find any folder in {paths}.")

    return dict((j, i) for i, j in enumerate(label))
