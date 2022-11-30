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
"""result utils"""

import os

import imageio
import numpy as np

__all__ = ["to8b", "save_image", "save_video"]


def to8b(x):
    """
    Convert normalized color to 8-bit color.

    Args:
        x (Tensor): Float32 tensor.

    Returns:
        Tensor, int8 tensor.
    """
    return (255 * np.clip(x, 0.0, 1.0)).astype(np.uint8)


def save_image(j, rgb_img, save_dir):
    """
    Save images of the j-th result in a specified directory.

    Args:
        j (int): the image index.
        rgb_img (Tensor): color map. (H, W).
        save_dir (str): the path to save the result.
    """

    file_path = os.path.join(save_dir, f"{j:04d}.png")
    imageio.imwrite(file_path, to8b(rgb_img))


def save_video(i, rgb_imgs, save_dir):
    """
    Save a video of the i-th iteration in a specified directory.

    Args:
        j (int): The image index.
        rgb_imgs (Tensor): A sequence of color maps. (#imgs, H, W).
        save_dir (str): The path to save the result.
    """

    file_path = os.path.join(save_dir, f"test_{i:04d}_rgb.mp4")
    imageio.mimwrite(file_path, to8b(rgb_imgs), fps=30, quality=8)
