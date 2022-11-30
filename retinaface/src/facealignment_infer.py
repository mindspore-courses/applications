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
"""Infer with face alignment network"""

import argparse
import json
import os

import cv2
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net

from model.facealignment import Facealignment2d


def resolve_json(path, json_path, output_path):
    """
    Will use boxes to clip images and output clipped images when work with retinaface.

    .. warning::
        In File system, the input path should contain infer.json, directory infer/
        File infer.json contains bounding boxes and descriptions of pictures detected by retinaface.
        Directory infer/ contains raw images.
        See more documents about this function at facealignment.ipynb - 6.3 联合推理

    Args:
        path(string): Work folder which contains infer.json.
        output_path(string): Path to save clipped images

    Returns:
        No direct returns.
        Will generate clipped files to '{path}/infer/single'.
    """
    if output_path[-1] not in ['/', '\\']:
        output_path = output_path + '/'
    json_file = open(json_path, 'r', encoding='utf-8')
    description = json.load(json_file)

    counter = 0
    for x in range(len(description)):

        # For each Picture
        temp_key = list(description.keys())[x]
        img = description[temp_key]
        img_path = img['img_path']
        img_path = img_path.split('/')[-1]
        read_img = cv2.imread(path+"/"+img_path)
        bboxes = img['bboxes']

        for i in range(len(bboxes)):
            if bboxes[i][4] > 0.95:
                # For Each Face
                img_clipped = pic_clip(read_img, bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3])
                img_resized = cv2.resize(img_clipped, (192, 192))
                cv2.imwrite(output_path + str(counter) + ".jpg", img_resized)
                counter += 1


def pic_clip(img, x, y, width, height):
    """
    Clip image.

    Args:
        img(ndarray): Input Image.
        x(int) : Position of bounding box's left upper corner on X axis.
        y(int): Position of bounding box's left upper corner on Y axis.
        width(int): Image width.
        height(int): Image height.

    Returns:
        img_clipped(ndarray), clipped images
        img_clipped(ndarray): Clipped image

    Examples:
        >>> pic_clip(image, 29, 63, 372, 128)
    """
    if x < 0:
        t0 = 0
    else:
        t0 = x
    if y < 0:
        t1 = 0
    else:
        t1 = y
    if x + width < img.shape[1]:
        t2 = x + width
    else:
        t2 = img.shape[1]
    if y + height < img.shape[0]:
        t3 = y + height
    else:
        t3 = img.shape[0]
    img_clipped = img[int(t1):int(t3), int(t0):int(t2)]
    return img_clipped


def parse_args():
    """
    Parse configuration arguments for infer.

    .. warning::
        when 'mode' is 'standalone', args should include 'clipped_path' and 'output_path'
        when 'mode' is 'retinaface', args should include 'raw_image_path', 'clipped_path' and 'output_path'
    """
    parser = argparse.ArgumentParser(description='Face Alignment')
    parser.add_argument('--mode', type=str, default='standalone', help='Infer Work Alone / work with Retinaface')
    parser.add_argument('--pre_trained', type=str, default='./data/FaceAlignment2D.ckpt', help='ckpt path')
    parser.add_argument('--device_target', type=str, default="Ascend", help='run device_target, GPU or Ascend')
    parser.add_argument('--raw_image_path', type=str, default=None, help='Raw Img Folder Path')
    parser.add_argument('--json_path', type=str, default=None, help='json file generated bu retinaface')
    parser.add_argument('--clipped_path', type=str, default=None, help='Clipped Picture Output Path')
    parser.add_argument('--output_path', type=str, default=None, help='Predict Result Output Path')
    parser.add_argument('--device_id', type=int, default=0, help='Device id')
    args = parser.parse_args()
    return args


def read_dir(dir_path):
    """
    Read images in directory

    Args:
        dir_path(string): Target directory contain pictures.

    Returns:
        all_files(file array), contains image file paths.

    Examples:
        >>> files = read_dir('/mnt/example')

    """
    if dir_path[-1] == '/':
        dir_path = dir_path[0:-1]
    print(dir_path)
    all_files = []
    if os.path.isdir(dir_path):
        file_list = os.listdir(dir_path)
        for f in file_list:
            f = dir_path + '/' + f
            if os.path.isdir(f):
                sub_files = read_dir(f)
                all_files = sub_files + all_files
            else:
                if os.path.splitext(f)[1] in ['.jpg', '.png', '.bmp', '.jpeg']:
                    all_files.append(f)
    else:
        raise "Error,not a dir"
    return all_files


def infer(args):
    """
    Infer with face alignment net

    Will generate txt files which contains predicted annotations and jpg files to show result.

    Args:
        args(dict): Multiple arguments for eval.

    Raises:
        ValueError: Unsupported device_target, this happens when 'device_target' not in ['GPU', 'Ascend'].
    """
    if args.device_target == "GPU":
        ms.set_context(mode=ms.GRAPH_MODE,
                       device_target="GPU",
                       save_graphs=False)
    elif args.device_target == "Ascend":
        ms.set_context(mode=ms.GRAPH_MODE,
                       device_target="Ascend",
                       device_id=args.device_id,
                       save_graphs=False)
        print("Using Ascend")
    else:
        raise ValueError("Unsupported device_target.")
    print("train args: ", args)

    net = Facealignment2d(output_channel=212)
    if args.pre_trained is not None:
        param_dict = load_checkpoint(args.pre_trained)
        load_param_into_net(net, param_dict)

    images = read_dir(args.clipped_path)
    print(str(len(images)) + " images detected")

    for file in images:
        image = cv2.imread(file)
        image = np.array(image)
        image = cv2.resize(image, (192, 192))
        raw_image = image.copy()
        image = image - 127.5
        image = image * 0.0078125
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)
        img_tensor = ms.Tensor(image, ms.float32)
        expand_dims = ops.ExpandDims()
        img_tensor = expand_dims(img_tensor, 0)
        result = (np.array(net(img_tensor)) * 96 + 96).astype(int).reshape((106, 2))
        np.savetxt(args.output_path + "/" + os.path.basename(file) + "_predict.txt", result, delimiter=",")
        for idx in range(106):
            raw_image = cv2.circle(raw_image, (int(result[idx, 0]), int(result[idx, 1])), 1, (200, 160, 75), 1)
        cv2.imwrite(args.output_path + "/" + os.path.basename(file) + "_predict.jpg", raw_image)


if __name__ == '__main__':
    args_opt = parse_args()
    if args_opt.mode == 'standalone':
        infer(args_opt)
    elif args_opt.mode == 'retinaface':
        resolve_json(args_opt.raw_image_path, args_opt.json_path, args_opt.clipped_path)
        infer(args_opt)
    else:
        raise "mode not implemented"
