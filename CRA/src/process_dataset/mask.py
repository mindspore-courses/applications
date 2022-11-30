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
"""Create CRA mask dataset."""

import os
import random
import cv2

import mindspore
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.dataset.vision.c_transforms import Inter, RandomRotation, Resize


def get_files(path):
    """
    Obtain the relative path of each sub file.

    Args:
        path(str): training mask dataset path.

    Return:
        ret: Relative path tuple of all sub files.
        len_mask: The number of mask images.
    """

    ret = []
    for tuple_path in os.walk(path):
        for filespath in tuple_path[2]:
            ret.append(os.path.join(tuple_path[0], filespath))
    len_mask = len(ret)
    return ret, len_mask


def read_masks(file):
    """
    Read given image.

    Args:
        file(str): given path.

    Return:
        img: Read image data.
    """

    img = cv2.imread(file)
    return img


def random_rotate_image(image):
    """
    Randomly rotate the image within the range of (-90, 90) angles.

    Args:
        image(numpy): Image to be rotated.

    Return:
        image: Rotated image.
    """

    rotate = RandomRotation(90, Inter.NEAREST)
    image = rotate(image)
    return image


def random_resize_image(image, scale, height, width):
    """
    Use NEAREST interpolation to adjust the input image to the given size.

    Args:
        image(numpy): Image to be resized.
        scale(float): Image scaling scale.
        height, width(int): The size of the input image.

    Return:
        image: Resized image.
    """

    newsize = [int(height * scale), int(width * scale)]
    resize = Resize(newsize, Inter.NEAREST)
    image = resize(image)
    return image


def random_mask(args):
    """
    Processing a given mask image by randomly flipping, rotating, scaling, cropping.

    Args:
        args (class): option class.

    Return:
        mask: Preprocessed training mask.
    """

    img_shape = args.IMG_SHAPE
    height = img_shape[0]
    width = img_shape[1]
    path_list, n_masks = get_files(args.mask_template_dir)
    nd = random.randint(0, n_masks - 1)
    path_mask = path_list[nd]
    mask = read_masks(path_mask)
    mask = ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5)(mask)
    scale = random.uniform(0.8, 1.0)
    mask = random_rotate_image(mask)
    mask = random_resize_image(mask, scale, height, width)
    crop = ds.vision.c_transforms.CenterCrop((height, width))
    mask1 = crop(mask)
    mask2 = Tensor.from_numpy(mask1)
    mask3 = mask2.astype(mindspore.float32)
    mask4 = mask3[:, :, 0:1]
    mask5 = ops.ExpandDims()(mask4, 0)
    mask6 = ops.Mul()(1 / 255, mask5)
    mask = ops.Reshape()(mask6, (1, height, width, 1))
    mask = ops.Transpose()(mask, (0, 3, 1, 2))
    return mask
