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
from tqdm import tqdm

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import ops
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import PriorBoostLayer, NNEncLayer, NonGrayMaskLayer, decode

from model.model import ColorizationModel
from model.colormodel import ColorModel
from process_datasets.data_generator import ColorizationDataset
from losses.loss import NetLoss



def parse_args():
    """Argument parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_target', default='GPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=1, type=int)
    parser.add_argument('--image_dir', type=str, default='../dataset/train', help='path to dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='path for saving trained model')
    parser.add_argument('--test_dirs', type=str, default='../images', help='path for saving trained model')
    parser.add_argument('--resource', type=str, default='./resources/')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_parallel_workers', type=int, default=1)
    parser.add_argument('--log_path', type=str, default='./log.txt')
    parser.add_argument('--learning_rate', type=float, default=0.5e-4)
    parser.add_argument('--save_step', type=int, default=200, help='step size for saving trained models')
    return parser.parse_args()


def main(opt):
    """Build and train model."""
    encode_layer = NNEncLayer(opt)
    boost_layer = PriorBoostLayer(opt)
    non_gray_mask = NonGrayMaskLayer()
    net = ColorizationModel()
    net_opt = nn.Adam(net.trainable_params(), learning_rate=opt.learning_rate)
    net_with_criterion = NetLoss(net)
    scale_sense = nn.FixedLossScaleUpdateCell(1)
    my_train_one_step_cell_for_net = nn.TrainOneStepWithLossScaleCell(net_with_criterion, net_opt,
                                                                      scale_sense=scale_sense)
    colormodel = ColorModel(my_train_one_step_cell_for_net)
    colormodel.set_train()
    log = open(opt.log_path, mode="w", encoding="utf-8")

    dataset = ColorizationDataset(opt.image_dir, opt.batch_size, opt.shuffle, opt.num_parallel_workers)
    data = dataset.run().create_tuple_iterator()

    for epoch in range(opt.num_epochs):
        iters = 0
        for images, img_ab in tqdm(data):
            print('iter:%d' % iters, file=log)
            images = ops.expand_dims(images, 1)
            encode, max_encode = encode_layer.forward(img_ab)
            targets = mindspore.Tensor(max_encode, dtype=mindspore.int32)
            boost = mindspore.Tensor(boost_layer.forward(encode), dtype=mindspore.float32)
            mask = mindspore.Tensor(non_gray_mask.forward(img_ab), dtype=mindspore.float32)
            net_loss = colormodel(images, targets, boost, mask)
            print('[%d/%d]\tLoss_net:: %.4f' % (epoch + 1, 200, net_loss[0]), file=log)
            print('[%d/%d]\tLoss_net:: %.4f' % (epoch + 1, 200, net_loss[0]))
            if iters % opt.save_step == 0:
                if not os.path.exists(opt.checkpoint_dir):
                    os.makedirs(opt.checkpoint_dir)
                mindspore.save_checkpoint(net, os.path.join(opt.checkpoint_dir, 'net'+str(epoch+1)+'_'+str(iters)
                                                            + '.ckpt'))
                img_ab_313 = net(images)
                print(img_ab_313.shape)
                out_max = np.argmax(img_ab_313[0].asnumpy(), axis=0)
                print('out_max', set(out_max.flatten()), file=log)
                print('out_max', set(out_max.flatten()))
                color_img = decode(images, img_ab_313, opt.resource)
                if not os.path.exists(opt.test_dirs):
                    os.makedirs(opt.test_dirs)
                plt.imsave(opt.test_dirs+'/'+str(epoch + 1)+'_'+str(iters)+'%s_infer.png', color_img)
            iters = iters + 1


if __name__ == '__main__':
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    main(args)
