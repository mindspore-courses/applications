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
"""Utils for RetinaFace inference."""

import os

import cv2
import numpy as np
from PIL import ImageDraw, Image


def ndarray2image(ndarray):
    """Convert numpy.ndarray image to PIL image."""
    ndarray = ndarray[:, :, ::-1]
    image = Image.fromarray(ndarray)
    return image


def image2ndarray(image):
    """Convert PIL image to numpy.ndarray image."""
    ndarray = np.array(image)
    return ndarray[:, :, ::-1]


def draw_preds(frame, bbox_list, draw_conf=False, landmark_list=None):
    """
    Draw predict boxes and landmarks for image.

    Args:
        frame (numpy.ndarray): Frame of images, usually get from cv2.imread, a [H,W,C] shape tensor.
        bbox_list (list): A list of lists, each list in bbox_list is [x, y, w, h, conf] represents a prediction box.
        draw_conf (bool): Whether draw confidence number above boxes.
        landmark_list (list): Shape [N,10], represents 5 landmark x,y pairs of N faces.

    Returns:
        Numpy ndarray, represents the image with boxes and landmarks on it.
    """
    image = ndarray2image(frame)
    thickness = int(
        max((image.size[0] + image.size[1]) // np.mean(np.array(image.size[:2])), 1))
    for i, j in enumerate(bbox_list):
        x, y, width, height, _ = j
        left, top, right, bottom = x, y, x + width, y + height
        top = max(0, int(top))
        left = max(0, int(left))
        bottom = min(image.size[1], int(bottom))
        right = min(image.size[0], int(right))
        draw = ImageDraw.Draw(image)
        for k in range(thickness):
            draw.rectangle([left + k, top + k, right - k, bottom - k], outline='red')
        if landmark_list:
            for k in range(5):
                center_x, center_y = (landmark_list[i][k * 2], landmark_list[i][k * 2 + 1])
                mult_thick = thickness * 0.2
                draw.ellipse(
                    (center_x - mult_thick, center_y - mult_thick, center_x + mult_thick, center_y + mult_thick),
                    fill='blue')
        if draw_conf:
            print('{}'.format(draw_conf))
    return image2ndarray(image)


def contain_chinese(string):
    """Check whether the string contains Chinese characters."""
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def cast_list_to_int(input_list):
    """Converts the elements of a list in a list to int."""
    processed_list = []
    for i in input_list:
        cur_list = []
        for j in i:
            cur_list.append(int(j))
        processed_list.append(cur_list)
    return processed_list


def read_input_images(image_folder):
    """
    Read images in the given folder, only read jpg images, and path should not contain any Chinese character.

    Args:
        image_folder(str): The path of image folder.

    Returns:
        A list, contains path of images in the folder.

    Raises:
        RuntimeError, if input images and folders contain Chinese characters.
    """
    read_data = []
    for i in os.listdir(image_folder):
        if i.endswith('.jpg'):
            to_add_path = os.path.join(image_folder, i)
            if contain_chinese(to_add_path):
                raise RuntimeError('Input images and folders cannot contain Chinese characters.')
            read_data.append(to_add_path)
    return read_data


def draw_image(pred_json, input_folder, save_folder, conf_threshold):
    """
    Read images in the given folder, draw predict image by predict result.

    Args:
        pred_json(dict): The predict json information contains bounding boxes and landmarks of images.
        input_folder(str): The path of input images.
        save_folder(str): The path of draw predict image result.
        conf_threshold(float): The threshold of drawing bounding boxes and landmarks.
    """
    for img_name in pred_json:
        input_path = os.path.join(input_folder, img_name + '.jpg')
        save_path = os.path.join(save_folder, img_name + '.jpg')
        bbox_to_draw = []
        landmark_to_draw = []
        processed_landmark = cast_list_to_int(pred_json[img_name]['landmarks'])
        for i, j in enumerate(pred_json[img_name]['bboxes']):
            x, y, width, height, conf = j
            if conf > conf_threshold:
                bbox_to_draw.append([int(x), int(y), int(width), int(height), conf])
                landmark_to_draw.append(processed_landmark[i])
        if not bbox_to_draw:
            continue
        frame = cv2.imread(input_path)
        img_det = draw_preds(frame, bbox_to_draw, landmark_list=processed_landmark)
        cv2.imwrite(save_path, img_det)
