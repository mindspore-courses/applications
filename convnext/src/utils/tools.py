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
"""misc functions for program"""
import os
from enum import Enum
import pathlib
from typing import Dict, Optional
from PIL import Image
from scipy import io
import cv2
import numpy as np

from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import process_datasets as data
from process_datasets.data_utils.moxing_adapter import sync_data


def set_device(args):
    """Set device and ParallelMode(if device_num > 1)"""
    rank = 0
    # set context and device
    device_target = args.device_target
    device_num = int(os.environ.get("DEVICE_NUM", "1"))

    if device_num > 1:
        if device_target == "Ascend":
            context.set_context(device_id=int(os.environ["DEVICE_ID"]))
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        rank = get_rank()
    else:
        context.set_context(device_id=args.device_id)

    return rank


def pretrained(args, model):
    """"Load pretrained weights if args.pretrained is given"""
    if args.run_modelarts:
        print('Syncing data.')
        local_data_path = '/cache/weight'
        name = args.pretrained.split('/')[-1]
        path = f"/".join(args.pretrained.split("/")[:-1])
        sync_data(path, local_data_path, threads=128)
        args.pretrained = os.path.join(local_data_path, name)
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        param_dict = load_checkpoint(args.pretrained)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != args.num_classes:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        load_param_into_net(model, param_dict)
    elif os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        param_dict = load_checkpoint(args.pretrained)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != args.num_classes:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        load_param_into_net(model, param_dict)
    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))


def get_dataset(args, training=True):
    """"Get model according to args.set"""
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args, training)

    return dataset


class Color(Enum):
    """dedine enum color."""
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def check_file_exist(file_name: str):
    """check_file_exist."""
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"File `{file_name}` does not exist.")


def color_val(color):
    """color_val."""
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


def imread(image, mode=None):
    """imread."""
    if isinstance(image, pathlib.Path):
        image = str(image)

    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        check_file_exist(image)
        image = Image.open(image)
        if mode:
            image = np.array(image.convert(mode))
    else:
        raise TypeError("Image must be a `ndarray`, `str` or Path object.")

    return image


def imwrite(image, image_path, auto_mkdir=True):
    """imwrite."""
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(image_path))
        if dir_name != '':
            dir_name = os.path.expanduser(dir_name)
            os.makedirs(dir_name, mode=777, exist_ok=True)

    image = Image.fromarray(image)
    image.save(image_path)


def imshow(img, win_name='', wait_time=0):
    """imshow"""
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
    """Mark the prediction results on the picture."""
    img = imread(img, mode="RGB")
    img = img.copy()
    x, y = 0, row_width
    text_color = color_val(text_color)
    for k, v in result.items():
        if isinstance(v, float):
            v = f'{v:.2f}'
        label_text = f'{k}: {v}'
        cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_COMPLEX,
                    font_scale, text_color)
        y += row_width
    if out_file:
        show = False
        imwrite(img, out_file)

    if show:
        imshow(img, win_name, wait_time)


def index2label(args):
    """Dictionary output for image numbers and categories of the ImageNet dataset."""
    metafile = os.path.join(args.data_url, "ILSVRC2012_devkit_t12/data/meta.mat")
    meta = io.loadmat(metafile, squeeze_me=True)['synsets']

    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]

    _, wnids, classes = list(zip(*meta))[:3]
    clssname = [tuple(clss.split(', ')) for clss in classes]
    wnid2class = {wnid: clss for wnid, clss in zip(wnids, clssname)}
    wind2class_name = sorted(wnid2class.items(), key=lambda x: x[0])

    mapping = {}
    for index, (_, class_name) in enumerate(wind2class_name):
        mapping[index] = class_name[0]
    return mapping
