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
"""Common image processing functions and tool functions."""

import os

import cv2
import numpy as np


def check_params(args):
    """
    Check dataset path, checkpoint path, image saving path and
    gan loss type. if the path do not exist, the directory will be
    created at the corresponding location according to the parameter
    settings.

    Args:
        args (namespace): training parameters.
    """

    data_path = os.path.join(args.data_dir, args.dataset)
    image_save_path = os.path.join(args.save_image_dir, args.dataset)
    ckpt_save_path = os.path.join(args.checkpoint_dir, args.dataset)
    if not os.path.exists(data_path):
        print(f'Dataset not found {data_path}')

    if not os.path.exists(image_save_path):
        print(f'* {image_save_path} does not exist, creating...')
        os.makedirs(image_save_path)

    if not os.path.exists(ckpt_save_path):
        print(f'* {ckpt_save_path} does not exist, creating...')
        os.makedirs(ckpt_save_path)

    if args.gan_loss not in {'lsgan', 'hinge', 'bce'}:
        raise ValueError(f'{args.gan_loss} is not supported')


def preprocessing(img, size):
    """
    Image processing.

    Args:
        img (ndarray): Input image.
        size (list): Image size.

    Returns:
        Ndarray, Processed image.
    """

    h, w = img.shape[:2]
    if h <= size[0]:
        h = size[0]
    else:
        x = h % 32
        h = h - x

    if w < size[1]:
        w = size[1]
    else:
        y = w % 32
        w = w - y

    # the cv2 resize func : dsize format is (W ,H)
    img = cv2.resize(img, (w, h))
    return img / 127.5 - 1.0


def denormalize_input(images):
    """
    Convert the pixel value range from 0-1 to 0-255.

    Args:
        images (ndarray / tensor): A batch of input images.

    Returns:
        Ndarray or tensor, denormalized data.
    """

    images = images * 127.5 + 127.5

    return images


def convert_image(img, img_size):
    """
    Change the channel order, transpose and resize.

    Args:
        img (ndarray): Input image.
        img_size (list): Image size.

    Returns:
        Ndarray, converted image.
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, img_size)
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img = np.asarray(img)
    return img


def inverse_image(img):
    """
    Convert the pixel value range from 0-1 to 0-255.

    Args:
        img (ndarray): Input image.

    Returns:
        Ndarray, converted image.
    """

    img = (img.squeeze() + 1.) / 2 * 255

    img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8)
    return img


def transform(fpath):
    """
    Image normalization and transpose.
    Convert the pixel value range from 0-255 to 0-1.

    Args:
        fpath (str): Path of image.

    Returns:
        Ndarray, transformed image.
    """

    image = cv2.imread(fpath)[:, :, ::-1]
    image = normalize_input(image)
    image = np.expand_dims(image.transpose(2, 0, 1), axis=0)

    return image


def compute_data_mean(data_folder):
    """
    Compute mean of R, G, B.

    Args:
        data_folder (str): Path of data.

    Returns:
        Ndarray, a list of channel means.

    Examples:
        >>> compute_data_mean('./dataset/train_photo')
    """

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f'Folder {data_folder} does not exits')

    image_files = os.listdir(data_folder)
    total = np.zeros(3)

    for img_file in image_files:
        path = os.path.join(data_folder, img_file)
        image = cv2.imread(path)
        total += image.mean(axis=(0, 1))

    channel_mean = total / len(image_files)
    mean = np.mean(channel_mean)

    return mean - channel_mean[..., ::-1]  # Convert to BGR for training


def normalize_input(images):
    """
    Convert the pixel value range from 0-255 to 0-1.

    Args:
        images (ndarray): A batch of input images.

    Returns:
        Ndarray, normalized data.
    """

    return images / 127.5 - 1.0


def inverse_transform_infer(image):
    """
    Image denormalization, transpose and change channel order.
    Convert the pixel value range from 0-1 to 0-255.
    Convert the channel order from RGB to BGR.

    Args:
        image (ndarray): Input image.

    Returns:
        Ndarray, inverse transformed image.
    """

    image = denormalize_input(image).asnumpy()
    image = cv2.cvtColor(image[0, :, :, :].transpose(
        1, 2, 0), cv2.COLOR_RGB2BGR)
    return image
