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
"""Image Processing"""

from enum import Enum
from typing import Optional, Dict
import cv2
import numpy as np

from utils.images import imread, imwrite


class Color(Enum):
    """An enum that defines engine colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if isinstance(color, str):
        return Color[color].value
    if isinstance(color, Color):
        return color.value
    if isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    if isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    if isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    raise TypeError(f'Invalid type for color: {type(color)}')


def imshow(img, win_name='', wait_time=0):
    """
    Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def show_result(img: str,
                result: Dict[int, float],
                text_color: str = 'green',
                font_scale: float = 0.5,
                row_width: int = 20,
                show: bool = False,
                win_name: str = '',
                wait_time: int = 0,
                out_file: Optional[str] = None) -> None:
    """Mark the results on the picture.

    Args:
        img (str): The image to be displayed.
        result (dict): The classification results to draw over `img`.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        font_scale (float): Font scales of texts.
        row_width (int): width between each row of results on the image.
        show (bool): Whether to show the image. Default: False.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param. Default: 0.
        out_file (str or None): The filename to write the image. Default: None.

    Returns:
        None
    """
    img = imread(img, mode="RGB")
    img = img.copy()

    # Write results on left-top of the image.
    x, y = 0, row_width
    text_color = color_val(text_color)
    for k, v in result.items():
        if isinstance(v, float):
            v = f'{v:.2f}'
        label_text = f'{k}: {v}'
        cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_COMPLEX,
                    font_scale, text_color)
        y += row_width

    # If out_file specified, do not show image in window.
    if out_file:
        show = False
        imwrite(img, out_file)

    if show:
        imshow(img, win_name, wait_time)
