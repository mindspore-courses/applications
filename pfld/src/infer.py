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
# ==============================================================================
"""Test the performance with one picture."""

import argparse
import os
import cv2
import numpy as np

from mindspore import context, Tensor
from mindspore.dataset import vision
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

from model.pfld import pfld_1x_68, pfld_1x_98


def main(args):
    context.set_context(device_id=args.device_id, mode=context.GRAPH_MODE, device_target=args.device_target)

    # Load model
    transform = vision.py_transforms.ToTensor()
    assert args.model_type in ['98_points', '68_points']
    if args.model_type == '68_points':
        net = pfld_1x_68()
        LoadPretrainedModel(net, args.pretrain_model_path[args.model_type]).run()
    else:
        net = pfld_1x_98()
        LoadPretrainedModel(net, args.pretrain_model_path[args.model_type]).run()

    # Load image
    for filename in os.listdir(args.infer_data_root):

        origin_img = cv2.imread(args.infer_data_root + '/' + filename)
        origin_h, origin_w, _ = origin_img.shape

        # Resize to (112 112 3) and convert to tensor
        img = cv2.resize(origin_img, (112, 112))
        img = transform(img)
        img = np.expand_dims(img, axis=0)
        img = Tensor(img)

        # Infer
        _, landmarks = net(img)

        # Postprocess
        landmarks = landmarks.asnumpy()
        landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
        pre_landmark = landmarks[0] * [origin_w, origin_h]

        # Draw points
        for (x, y) in pre_landmark.astype(np.int32):
            cv2.circle(origin_img, (x, y), 1, (255, 0, 0), -1)

        # Save image
        cv2.imwrite("infer.jpg", origin_img)
        print('Done')


pretain_model = {'98_points': 'https://download.mindspore.cn/vision/pfld/PFLD1X_WFLW.ckpt',
                 '68_points': 'https://download.mindspore.cn/vision/pfld/PFLD1X_300W.ckpt'}


def parse_args():
    """
    Set network parameters.

    Returns:
        ArgumentParser. Parameter information.
    """
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--device_target', default='CPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--infer_data_root', default='../datasets/infer_image', type=str, metavar='PATH')
    parser.add_argument('--model_type', default='98_points', type=str)
    parser.add_argument('--pretrain_model_path', default=pretain_model, type=dict)
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
