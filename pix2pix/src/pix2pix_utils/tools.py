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
# ===========================================================================
"""Tools for Pix2Pix model."""

import numpy as np
from PIL import Image as image
import matplotlib.pyplot as plt
from mindspore import Tensor


plt.switch_backend('Agg')


def save_losses(g_losses, f_losses, idx, config):
    """
    Plot loss information of iterations and save in file.

   Args:
       g_losses (float): Generator loss.
       f_losses (float): Discriminator loss.
       idx (int): Image number.
       config (class): Option class.
    """

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(f_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Losses")
    plt.legend()
    plt.savefig(config.loss_show_dir + "/{}.png".format(idx))


def save_image(img, img_path):
    """
    Save a numpy image to the disk.

    Args:
        img (numpy array / Tensor): Image to save.
        image_path (str): The path of the image.
    """

    if isinstance(img, Tensor):
        img = decode_image(img)
    elif not isinstance(img, np.ndarray):
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))

    img_pil = image.fromarray(img)
    img_pil.save(img_path + ".jpg")


def decode_image(img):
    """
    Decode a [1, transforms, H, W] Tensor to image numpy array.

    Returns:
        decode input image.
    """

    mean = 0.5 * 255
    std = 0.5 * 255

    return (img.asnumpy()[0] * std + mean).astype(np.uint8).transpose((1, 2, 0))   # ——>（256，256，3）


def get_lr(config):
    """
    Linear learning-rate generator.
    Keep the same learning rate for the first <opt.n_epochs> epochs.
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.

    Returns:
        Get learning rate.
    """

    lrs = [config.lr] * config.dataset_size * config.n_epochs
    lr_epoch = 0
    for epoch in range(config.n_epochs_decay):
        lr_epoch = config.lr * (config.n_epochs_decay - epoch) / config.n_epochs_decay
        lrs += [lr_epoch] * config.dataset_size
    lrs += [lr_epoch] * config.dataset_size * (config.epoch_num - config.n_epochs_decay - config.n_epochs)
    return Tensor(np.array(lrs).astype(np.float32))
