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
ICT Transformer distributed train.
"""

import os
import argparse

import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size

from datasets.dataset import load_dataset
from models.networks import GPT
from models.loss import TransformerWithLoss
from transformer_utils.util import AverageMeter


def main(opts):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    init('nccl')
    rank_id = get_rank()
    rank_size = get_group_size()
    context.set_auto_parallel_context(device_num=rank_size, parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True, parameter_broadcast=True)
    kmeans = np.load('../kmeans_centers.npy')
    kmeans = np.rint(127.5 * (kmeans + 1.0))

    # Define the dataset
    train_dataset = load_dataset(opts.data_path, kmeans, mask_path=opts.mask_path, is_train=True,
                                 use_imagefolder=opts.use_ImageFolder, prior_size=opts.prior_size,
                                 random_stroke=opts.random_stroke, rank_id=rank_id, rank_size=rank_size)
    train_dataset = train_dataset.batch(opts.batch_size // rank_size)
    step_size = train_dataset.get_dataset_size()

    # Define the model
    block_size = opts.prior_size * opts.prior_size
    transformer = GPT(vocab_size=kmeans.shape[0], n_embd=opts.n_embd, n_layer=opts.n_layer, n_head=opts.n_head,
                      block_size=block_size, use_gelu2=opts.GELU_2, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0)
    if os.path.exists(opts.ckpt_path):
        print('Strat loading the model parameters from %s' % (opts.ckpt_path))
        checkpoint = mindspore.load_checkpoint(opts.ckpt_path)
        mindspore.load_param_into_net(transformer, checkpoint)
        print('Finished load the model')
    model = TransformerWithLoss(backbone=transformer)

    # Define the optimizer
    optimizer = nn.Adam(model.trainable_params(), learning_rate=opts.learning_rate, beta1=opts.beta1, beta2=opts.beta2)

    train_net = nn.TrainOneStepCell(model, optimizer)
    train_loss = AverageMeter()
    best_loss = 10000000000.
    if not os.path.exists(opts.save_path):
        os.mkdir(opts.save_path)
    for epoch in range(opts.train_epoch):
        for i, sample in enumerate(train_dataset.create_dict_iterator()):
            x = sample['data']
            y = sample['mask']
            y = mindspore.ops.Cast()(y, mindspore.float32)
            loss = train_net(x, x, y)
            train_loss.update(loss, 1)
            if i % 100 == 0:
                print(f"Epoch: [{epoch} / {opts.train_epoch}], "
                      f"step: [{i} / {step_size}], "
                      f"loss: {train_loss.avg.asnumpy()}")
                if train_loss.avg < best_loss:
                    best_loss = train_loss.avg
                    mindspore.save_checkpoint(transformer,
                                              os.path.join(opts.save_path, 'ImageNet_best_{}.ckpt'.format(rank_id)))

    mindspore.save_checkpoint(transformer, os.path.join(opts.save_path, 'ImageNet_latest_{}.ckpt'.format(rank_id)))


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='', help='The path of resume ckpt')
    parser.add_argument('--save_path', type=str, default='./checkpoint', help='save checkpoints path')
    parser.add_argument('--data_path', type=str, help='Indicate where is the training set')
    parser.add_argument('--validation_path', type=str, help='Where is the validation set of ImageNet')
    parser.add_argument('--mask_path', type=str, help='Where is the mask')
    parser.add_argument('--ImageNet', action='store_true', help='Training with ImageNet')
    parser.add_argument('--batch_size', type=int, default=2 * 8, help='2*8 maybe suitable for 8*V100')
    parser.add_argument('--train_epoch', type=int, default=5, help='How many epochs')
    parser.add_argument('--random_stroke', action='store_true', help='Use the generated mask')
    parser.add_argument('--use_ImageFolder', action='store_true', help='Using the original folder for ImageNet dataset')
    parser.add_argument('--prior_size', type=int, default=32, help='Input sequence length = prior_size*prior_size')

    # transformer parameters
    parser.add_argument('--n_layer', type=int, default=35)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--GELU_2', action='store_true', help='Use the new activation function')

    # optimization parameters
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Value of learning rate.')
    parser.add_argument("--beta1", type=float, default=0.9, help="Value of beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Value of beta2")

    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    main(parse_args())
