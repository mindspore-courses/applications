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
"""Eval Retinaface_resnet50."""

from __future__ import print_function

import os
import ast
import argparse
import numpy as np
import cv2
import mindspore as ms
from mindspore import Tensor

from model.retinaface import RetinaFace
from model.resnet50 import resnet50
from model.mobilenet025 import MobileNetV1
from process_datasets.widerface import image_transform
from utils.detection import prior_box
from utils.detection_engine import DetectionEngine
from utils.timer import Timer


def create_val_cell(backbone='mobilenet025', config=None):
    """
    Create model for evaluate.

    Args:
        backbone (str): A string represents backbone of the network, can be 'resnet50' or 'mobilenet025'.
            Default: 'resnet50'.
        config (dict): Configuration dictionary for RetinaFace with mobilenet025 backbone. Default: None.

    Returns:
        A nn.Cell network, represents the model for evaluation.

    Raises:
        RuntimeError: If backbone in the config is not supported.
    """
    if backbone == 'resnet50':
        print('resnet50')
        backbone = resnet50()
        network = RetinaFace(phase='predict', backbone=backbone)
    elif backbone == 'mobilenet025':
        print('mobilenet025')
        backbone = MobileNetV1()
        network = RetinaFace(phase='predict', backbone=backbone, cfg=config)
    else:
        raise RuntimeError('Backbone is unsupported.')
    backbone.set_train(False)
    network.set_train(False)
    return network


def read_label(label_path):
    """Read label.txt at the given path, returns the image paths in it and its length."""
    with open(label_path, 'r') as f:
        lines = f.readlines()
        test_dataset = []
        for im_path in lines:
            im_path = im_path.rstrip()
            if im_path.startswith('# '):
                test_dataset.append(im_path[2:])
    num_images = len(test_dataset)
    return test_dataset, num_images


def generate_priors(image_dataset, image_folder, min_sizes, steps, clip):
    """
    Read images from paths and generate priors for them, calculate the correct max height and width for them for
    network.

    Args:
        image_dataset (list): Images paths start from '/images' , usually is read from widerface dataset label.txt.
        image_folder (str): The base path of images, the path of widerface dataset.
        min_sizes (list): Size of prior boxes corresponding to different feature layers, which shape is [N,M], where N
            represents the number of feature layers, M represents different kind of sizes in the layer.
        steps (list): Multiple by which each feature layer is compressed, which length is N, represents the different
            multiple of N layers.
        clip (bool): Whether to restrict the output between 0 and 1.

    Returns:
        A tuple, its first element is priors, a numpy ndarray with shape [N,4], which represents generated N prior boxes
            with x,y,width and height.Its second and third element is the max height and width of images.
    """
    h_max, w_max = 0, 0
    for img_name in image_dataset:
        image_path = os.path.join(image_folder, 'images', img_name)
        read_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if read_image.shape[0] > h_max:
            h_max = read_image.shape[0]
        if read_image.shape[1] > w_max:
            w_max = read_image.shape[1]
    h_max = (int(h_max / 32) + 1) * 32
    w_max = (int(w_max / 32) + 1) * 32
    priors = prior_box(image_sizes=(h_max, w_max),
                       min_sizes=min_sizes,
                       steps=steps,
                       clip=clip)
    return priors, h_max, w_max


def val(cfg):
    """Evaluate the RetinaFace checkpoint."""

    ms.set_context(mode=ms.GRAPH_MODE, device_target=cfg['device_target'], save_graphs=False,
                   device_id=cfg['device_id'])
    if cfg['backbone'] == 'resnet50':
        backbone = resnet50()
        network = RetinaFace(phase='predict', backbone=backbone)
    elif cfg['backbone'] == 'mobilenet025':
        backbone = MobileNetV1()
        network = RetinaFace(phase='predict', backbone=backbone, cfg=cfg)
    else:
        raise RuntimeError('Backbone is unsupported.')
    backbone.set_train(False)
    network.set_train(False)
    param_dict = ms.load_checkpoint(cfg['val_model'])
    print('Load trained model done. {}'.format(cfg['val_model']))
    network.init_parameters_data()
    ms.load_param_into_net(network, param_dict)

    testset_folder = cfg['val_dataset_folder']
    testset_label_path = cfg['val_dataset_folder'] + "label.txt"
    test_dataset, num_images = read_label(testset_label_path)

    timers = {'forward_time': Timer(), 'misc': Timer()}

    if cfg['val_origin_size']:
        priors, h_max, w_max = generate_priors(test_dataset, testset_folder,
                                               min_sizes=ast.literal_eval(cfg['min_sizes']),
                                               steps=cfg['steps'],
                                               clip=cfg['clip'])
    else:
        target_size = cfg['target_size']
        max_size = cfg['max_size']
        print('size set.')
        priors = prior_box(image_sizes=(max_size, max_size),
                           min_sizes=ast.literal_eval(cfg['min_sizes']),
                           steps=cfg['steps'],
                           clip=cfg['clip'])

    detection = DetectionEngine(cfg)
    print('Predict box starting')
    for i, img_name in enumerate(test_dataset):
        image_path = os.path.join(testset_folder, 'images', img_name)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        if cfg['val_origin_size']:
            img, resize = image_transform(img, True, h_max, w_max)
        else:
            img, resize = image_transform(img, False, target_size, max_size)
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
        ldm_scale = np.array(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0], img.shape[1], img.shape[0], img.shape[1],
             img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
        img -= cfg['image_average']
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = Tensor(img)
        timers['forward_time'].start()
        boxes, confs, ldm = network(img)
        timers['forward_time'].end()
        timers['misc'].start()
        detection.detect(boxes, ldm, confs, resize, scale, ldm_scale, img_name, priors)
        timers['misc'].end()
        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images,
                                                                                     timers['forward_time'].diff,
                                                                                     timers['misc'].diff))
    print('Predict box done.')

    print('Eval starting')
    if cfg['val_save_result']:
        predict_result_path = detection.write_result()
        print('Predict result path is {}'.format(predict_result_path))
    detection.get_eval_result()
    print('Eval done.')


def parse_args():
    """Parse configuration arguments for evaluating."""
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--backbone', default='mobilenet025', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--val_model', default='./data/RetinaFace-120_1609.ckpt', type=str)
    parser.add_argument('--val_origin_size', default=False, type=bool)
    parser.add_argument('--val_dataset_folder', default='./data/widerface/valid/', type=str)
    parser.add_argument('--val_save_result', default=True, type=bool)
    parser.add_argument('--val_confidence_threshold', default=0.02, type=float)
    parser.add_argument('--val_nms_threshold', default=0.4, type=float)
    parser.add_argument('--val_iou_threshold', default=0.5, type=float)
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--val_predict_save_folder', default='./data/widerface_result', type=str)
    parser.add_argument('--val_gt_dir', default='./data/ground_truth/', type=str)
    parser.add_argument('--min_sizes', default='[[16, 32], [64, 128], [256, 512]]', type=str)
    parser.add_argument('--steps', default=[8, 16, 32], type=float, nargs='+')
    parser.add_argument('--clip', default=False, type=bool)
    parser.add_argument('--image_average', default=(104.0, 117.0, 123.0), type=float, nargs='+')
    parser.add_argument('--target_size', default=1600, type=int)
    parser.add_argument('--max_size', default=2176, type=int)
    parser.add_argument('--in_channel', default=32, type=int)
    parser.add_argument('--out_channel', default=64, type=int)
    parser.add_argument('--device_id', default=0, type=str)
    parser.add_argument('--device_target', default='Ascend', type=str)
    return vars(parser.parse_args())


if __name__ == '__main__':
    val(parse_args())
