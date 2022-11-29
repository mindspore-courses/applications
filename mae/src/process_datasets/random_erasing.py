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
"""
Random erasing.
"""

import math
import random

import numpy as np


def _get_pixels(per_pixel, rand_color, patch_size, dtype=np.float32):
    """
    The pixels of the image are processed differently according to the different modes set.

    Args:
        per_pixel (bool): Whether it is per_pixel.
        rand_color (bool): Whether it is rand_color.
        patch_size (int): Patch size.
        dtype(numpy.dtype): Numpy data type. Default=np.float32.

    Returns:
        func: The method of data Processing.
    """
    if per_pixel:
        func = np.random.normal(size=patch_size).astype(dtype)
    elif rand_color:
        func = np.random.normal(size=(patch_size[0], 1, 1)).astype(dtype)
    else:
        func = np.zeros((patch_size[0], 1, 1), dtype=dtype)
    return func


class RandomErasing:
    """
    Randomly selects a rectangular region of the image and erases its pixels.

    This random erasure variant is intended to be applied to a batch or individual image tensor
    after it has been normalized by the mean and standard values of the dataset.

    Args:
         probability (float): The probability of performing a random erase operation. Default: 0.5.
         min_area (float): The minimum percentage of the erased area relative to the input image area. Default: 0.02.
         max_area (float): The maximum percentage of the erased area with respect to the input image area. Default: 1/3.
         min_aspect (float): The minimum aspect ratio of the erased area. Default: 0.3.
         max_aspect (float): The maximum aspect ratio of the erased area. Default: None.
         mode (str): Pixel color mode, one of 'const', 'rand' or 'pixel'. Default: 'pixel'.
             - const: The color of all channels of the erased block is constant.
             - rand: The erase block is a random (normal) color for each channel.
             - pixel: The erase block is a random (normal) color for each pixel.

         min_count (int):The minimum number of erase blocks per image, the area of each box is scaled by the count.
         max_count (int): The maximum number of erase blocks per image, the area of each box is scaled by the count.
             The count per image is chosen randomly between 1 and this value.
         num_splits (int): Skip first slice of batch if num_splits is set (for clean portion of samples).
    """
    def __init__(self, probability=0.5, min_area=0.02, max_area=1 / 3, min_aspect=0.3,
                 max_aspect=None, mode='const', min_count=1, max_count=None, num_splits=0):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True
        elif mode == 'pixel':
            self.per_pixel = True
        else:
            assert not mode or mode == 'const'

    def _erase(self, img, channel, img_h, img_w, d_type):
        """
        Random erasure operation can erase some patches of the picture and return the erased image.

        Args:
            img (str): Image location.
            channel (int): Channel num.
            img_h (int): Image height.
            img_w (int): Image width .
            d_type (type): Data type.

        Returns:
            Tensor, output tensor, the shape is (batch_size * 3).
        """

        # Set the corresponding random number to compare with the erasure likelihood threshold
        if random.random() <= self.probability:
            area = img_h * img_w

            # Set the number of pieces to erase, between minimum and maximum.
            count = self.min_count if self.min_count == self.max_count else \
                random.randint(self.min_count, self.max_count)
            for _ in range(count):
                for _ in range(10):

                    # Calculate the final masked area.
                    target_area = random.uniform(self.min_area, self.max_area) * area / count

                    # Calculate the percentage of mask.
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))

                    # Calculate the width and height of the mask.
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))

                    # Determine if the requirements are met.
                    if w < img_w and h < img_h:
                        top = random.randint(0, img_h - h)
                        left = random.randint(0, img_w - w)
                        img[:, top:top + h, left:left + w] = _get_pixels(
                            self.per_pixel, self.rand_color, (channel, h, w),
                            dtype=d_type)
                        break
        return img

    def __call__(self, x):
        """
        Execute random mask

        Args:
            x (tensor): the image before random erasing.

        Returns:
            Tensor, output tensor after random erasing.
        """
        # Call the method directly if the format matches.
        if len(x.shape) == 3:
            output = self._erase(x, *x.shape, x.dtype)

        # If the format does not match, adjust it to the correct format and then execute the method.
        else:
            output = np.zeros_like(x)
            batch_size, channel, img_h, img_w = x.shape

            # Skip first slice of batch if num_splits is set (for clean portion of samples).
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                output[i] = self._erase(x[i], channel, img_h, img_w, x.dtype)
        return output
