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
"""Utils for Upsample."""

import os
import random
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import mindspore
import mindspore.ops.operations as P

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def postprocess(img):
    """
    Image post-processing.
    [0, 1] => [0, 255]

    Args:
        img(Tensor): Image tensor.
    """
    img = img * 255.0
    img = P.Transpose()(img, (0, 2, 3, 1))
    return P.Cast()(img, mindspore.int32)


def create_dir(path):
    """
    Create directory.

    Args:
        dir(str): Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    """
    Create a random mask.

    Args:
        width(int): The mask image width.
        height(int): The mask image height.
        mask_width(int): The mask width length.
        mask_height(int): The mask height length.
        x(int): The mask start of x. Default: None
        y(int): The mask start of y. Default: None
    """
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


def stitch_images(inputs, *outputs, img_per_row=2):
    """
    Stitch multiple images into one image.

    Args:
        inputs(Tensor): Image tensor.
        outputs(Tensor): Other image tensor..
        img_per_row(int): The number of columns per image. Default: 2
    """
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB',
                    (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array(images[cat][ix]).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    """
    Show the image.

    Args:
        img(Tensor): Image tensor.
        title(str): The show window title.
    """
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    """
    Save the image to the specified path.

    Args:
        img(Tensor): Image tensor.
        path(str): The save path
    """
    im = Image.fromarray(img.asnumpy().astype(np.uint8).squeeze())
    im.save(path)
