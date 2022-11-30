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
"""Test the performance in the specified directory."""

import argparse
import os

import cv2
from tqdm import tqdm
from mindspore import Tensor
from mindspore import context
from mindspore import float32 as dtype
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.model import Model

from models.generator import Generator
from animeganv2_utils.pre_process import transform, inverse_transform_infer


def parse_args():
    """Argument parsing."""
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--device_target', default='GPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--infer_dir', default='../dataset/test/real', type=str)
    parser.add_argument('--infer_output', default='../dataset/output', type=str)
    parser.add_argument('--ckpt_file_name', default='../checkpoints/Hayao/netG_30.ckpt')
    return parser.parse_args()


def main():
    """Convert real image to anime image."""
    net = Generator()
    param_dict = load_checkpoint(args.ckpt_file_name)
    load_param_into_net(net, param_dict)
    data = os.listdir(args.infer_dir)
    bar = tqdm(data)
    model = Model(net)

    if not os.path.exists(args.infer_output):
        os.mkdir(args.infer_output)

    for img_path in bar:
        img = transform(os.path.join(args.infer_dir, img_path))
        img = Tensor(img, dtype=dtype)
        output = model.predict(img)
        img = inverse_transform_infer(img)
        output = inverse_transform_infer(output)
        output = cv2.resize(output, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(args.infer_output, img_path), output)
    print('Successfully output images in ' + args.infer_output)


if __name__ == '__main__':
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    main()
