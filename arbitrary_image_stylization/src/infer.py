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
"""Arbitrary image stylization infer."""
import argparse

import cv2
import mindspore
from mindspore import load_checkpoint, load_param_into_net, Tensor

from model.ais import Ais

def get_image(path):
    """
    Read image file.

    Args:
        path (str): Path of image.

    Returns:
        4-D tensor with value in [0, 1].
    """
    image = cv2.imread(path)
    image = Tensor(image, mindspore.float32)
    image = image.transpose((2, 0, 1))
    image = image / 255.0
    image = image.expand_dims(axis=0)
    return image

def main(args):
    """ Inference. """
    network = Ais(args.style_prediction_bottleneck)
    load_param_into_net(network, load_checkpoint(args.ckpt_path))
    # load images
    content = get_image(args.content_path)
    style = get_image(args.style_path)
    # predict
    stylized = network((content, style))
    print(stylized)
    stylized = stylized.asnumpy()[0].transpose((1, 2, 0)) * 255
    # save result
    cv2.imwrite(args.output, stylized)

def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description='arbitrary image stylization infer')
    parser.add_argument('--content_path', type=str, default='./content.jpg', help='Path of content image.')
    parser.add_argument('--style_path', type=str, default='./style.jpg', help='Path of style image.')
    parser.add_argument('--style_prediction_bottleneck', type=int, default=100)
    parser.add_argument('--ckpt_path', type=str, default='./model-1.0.ckpt', help='Path of checkpoint.')
    parser.add_argument('--output', type=str, default='./result.jpg', help='Path to save stylized image.')
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
