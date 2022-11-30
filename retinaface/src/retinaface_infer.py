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
"""Load checkpoint model and inference, draw predict images."""
import os
import ast
import argparse
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor

from model.mobilenet025 import MobileNetV1
from model.resnet50 import resnet50
from model.retinaface import RetinaFace
from utils.detection_engine import DetectionEngine
from utils.timer import Timer
from utils.detection import prior_box
from utils.draw_prediction import read_input_images, draw_image


def infer(cfg):
    """
    Inference and draw predict images.

    Args:
        cfg(dict): A dictionary contains configure for inference.

    Raises:
        RuntimeError: If backbone in the config is not supported.
    """
    if not os.path.exists(cfg['img_folder']):
        raise ValueError('Input image folder not exists.')
    if not os.path.exists(cfg['draw_save_folder']):
        os.mkdir(cfg['draw_save_folder'])
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
    test_dataset = read_input_images(cfg['img_folder'])
    num_images = len(test_dataset)
    timers = {'forward_time': Timer(), 'misc': Timer()}
    if cfg['val_origin_size']:
        h_max, w_max = 0, 0
        for image_path in test_dataset:
            image_read = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_read.shape[0] > h_max:
                h_max = image_read.shape[0]
            if image_read.shape[1] > w_max:
                w_max = image_read.shape[1]
        h_max = (int(h_max / 32) + 1) * 32
        w_max = (int(w_max / 32) + 1) * 32
        priors = prior_box(image_sizes=(h_max, w_max),
                           min_sizes=ast.literal_eval(cfg['min_sizes']),
                           steps=cfg['steps'],
                           clip=cfg['clip'])
    else:
        target_size = cfg['target_size']
        max_size = cfg['max_size']
        priors = prior_box(image_sizes=(max_size, max_size),
                           min_sizes=ast.literal_eval(cfg['min_sizes']),
                           steps=cfg['steps'],
                           clip=cfg['clip'])
    detection = DetectionEngine(cfg)
    print('Predict box starting')
    image_average = cfg['image_average']
    for i, image_path in enumerate(test_dataset):
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        if cfg['val_origin_size']:
            resize = 1
            image_t = np.empty((h_max, w_max, 3), dtype=img.dtype)
            image_t[:, :] = image_average
            image_t[0:img.shape[0], 0:img.shape[1]] = img
            img = image_t
        else:
            im_size_min = np.min(img.shape[0:2])
            im_size_max = np.max(img.shape[0:2])
            resize = float(target_size) / float(im_size_min)
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
            image_t[:, :] = image_average
            image_t[0:img.shape[0], 0:img.shape[1]] = img
            img = image_t
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
        ldm_scale = np.array(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0], img.shape[1], img.shape[0], img.shape[1],
             img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
        img -= image_average
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = Tensor(img)
        timers['forward_time'].start()
        boxes, confs, ldm = network(img)
        timers['forward_time'].end()
        timers['misc'].start()
        img_name = 'infer/' + os.path.split(image_path)[-1]
        detection.detect(boxes, ldm, confs, resize, scale, ldm_scale, img_name, priors)
        timers['misc'].end()
        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images,
                                                                                     timers['forward_time'].diff,
                                                                                     timers['misc'].diff))
    print('Predict box done.')
    print('Draw image starting')
    draw_image(detection.results['infer'], cfg['img_folder'], cfg['draw_save_folder'], cfg['conf_thre'])
    print('Draw image done.')


def parse_args():
    """Parse configuration arguments for inference."""
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--backbone', default='mobilenet025', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--val_model', default='./data/RetinaFace-120_1609.ckpt', type=str)
    parser.add_argument('--val_origin_size', default=False, type=bool)
    parser.add_argument('--val_confidence_threshold', default=0.02, type=float)
    parser.add_argument('--val_nms_threshold', default=0.4, type=float)
    parser.add_argument('--val_iou_threshold', default=0.5, type=float)
    parser.add_argument('--val_predict_save_folder', default='./data/widerface_result', type=str)
    parser.add_argument('--val_gt_dir', default='./data/ground_truth/', type=str)
    parser.add_argument('--img_folder', default='./input_image', type=str)
    parser.add_argument('--draw_save_folder', default='./infer_image', type=str)
    parser.add_argument('--conf_thre', default=0.4, type=float)
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
    infer(cfg=parse_args())
