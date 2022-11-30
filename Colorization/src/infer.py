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
"""
Test the performance with one picture.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import mindspore
from mindspore import context
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import ops
from mindspore.train.model import Model
from tqdm import tqdm

from model.model import ColorizationModel
from process_datasets.data_generator import ColorizationDataset
from utils.utils import decode


def infer(opt):
    """test model"""
    net = ColorizationModel()
    param_dict = load_checkpoint(opt.ckpt_path)
    load_param_into_net(net, param_dict)
    colorizer = Model(net)
    dataset = ColorizationDataset(opt.img_path, 1, prob=0)
    data = dataset.run().create_tuple_iterator()
    iters = 0
    if not os.path.exists(opt.infer_dirs):
        os.makedirs(opt.infer_dirs)
    for images, _ in tqdm(data):
        images = ops.expand_dims(images, 1)
        img_ab_313 = colorizer.predict(images)
        out_max = np.argmax(img_ab_313[0].asnumpy(), axis=0)
        print('out_max', set(out_max.flatten()))
        color_img = decode(images, img_ab_313, opt.resource)
        plt.imsave(opt.infer_dirs+'/'+str(iters)+'_infer.png', color_img)
        iters = iters + 1


def parse_args():
    """import parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='../dataset/val')
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/net26_9600.ckpt')
    parser.add_argument('--resource', type=str, default='./resources/')
    parser.add_argument('--device_target', default='GPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=1, type=int)
    parser.add_argument('--infer_dirs', default='../dataset/output', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mindspore.context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    infer(args)
