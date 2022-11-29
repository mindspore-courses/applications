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
"""Adjust the brightness of output image."""

import cv2
import numpy as np


def calculate_average_brightness(img):
    """
    Calculates the average brightness in the specified irregular image.

    Args:
        img (ndarray): Original image.

    Returns:
        Ndarray, grayscale average brightness image.
    """

    # Average value of three color channels
    r = img[..., 0].mean()
    g = img[..., 1].mean()
    b = img[..., 2].mean()

    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return brightness


def adjust_brightness_from_src_to_dst(dst, src, if_show=None, if_info=None):
    """
    Adjusting the average brightness of the target image to the average brightness of the source image.

    Args:
        dst (ndarray): Target image.
        src (ndarray): Source image.
        if_show (bool): Choose whether to show the image. Default: None.
        if_info (bool): Choose whether to print the image information. Default: None.

    Returns:
        Ndarray, image after brightness adjustment.
    """

    brightness1 = calculate_average_brightness(src)
    brightness2 = calculate_average_brightness(dst)
    brightness_difference = brightness1 / brightness2

    if if_info:
        print('Average brightness of original image', brightness1)
        print('Average brightness of target', brightness2)
        print('Brightness Difference between Original Image and Target', brightness_difference)

    # According to the average display brightness
    dstf = dst * brightness_difference

    # To limit the results and prevent crossing the border,
    # It must be converted to uint8, otherwise the default result is float32, and errors will occur.
    dstf = np.clip(dstf, 0, 255)
    dstf = np.uint8(dstf)

    ma, na, _ = src.shape
    mb, nb, _ = dst.shape
    result_show_img = np.zeros((max(ma, mb), 3 * max(na, nb), 3))
    result_show_img[:mb, :nb, :] = dst
    result_show_img[:ma, nb:nb + na, :] = src
    result_show_img[:mb, nb + na:nb + na + nb, :] = dstf
    result_show_img = result_show_img.astype(np.uint8)

    if if_show:
        cv2.imshow('-', cv2.cvtColor(result_show_img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return dstf
