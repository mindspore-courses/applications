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
"""Train model."""

import os
import time

import mindspore
import mindspore.dataset as ds
from mindspore import nn
from mindspore import context, save_checkpoint
from mindspore.communication import get_rank, init
from mindspore.context import ParallelMode

from process_dataset.mask import random_mask
from process_dataset.dataset import InpaintDataset
from config.config import cra_config
from models.inpainting_network import GatedGenerator, Discriminator
from models.loss import GenWithLossCell, DisWithLossCell
from models.train_one_step import TrainOneStepD, TrainOneStepG


def trainer(args):
    """Train model."""

    # Preprocess the data for training
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.run_distribute:
        if args.device_target == 'Ascend':
            context.set_context(device_id=int(os.getenv('DEVICE_ID')))
        device_num = args.device_num
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()
        rank_id = get_rank()
    else:
        if args.device_target in ['GPU', 'Ascend']:
            rank_id = 0
            if args.device_id == 0:
                if os.getenv("DEVICE_ID", "not_set").isdigit():
                    context.set_context(device_id=int(os.getenv("DEVICE_ID", "0")))
            else:
                context.set_context(device_id=args.device_id)
    dataset_generator = InpaintDataset(args)
    dataset_size = len(dataset_generator)
    total_batch = (dataset_size // args.train_batchsize) // args.device_num
    dataset = ds.GeneratorDataset(dataset_generator, ['image'], num_shards=args.device_num, shard_id=rank_id)
    dataset = dataset.batch(args.train_batchsize, drop_remainder=True)
    dataset = dataset.create_dict_iterator()

    # Network
    net_g = GatedGenerator(args)
    net_d = Discriminator()
    netg_with_loss = GenWithLossCell(net_g, net_d, args)
    netd_with_loss = DisWithLossCell(net_g, net_d, args)
    lr = nn.exponential_decay_lr(args.learning_rate, args.lr_decrease_factor, total_batch * args.epochs, total_batch,
                                 args.lr_decrease_epoch, True)
    optimizer_g = nn.Adam(filter(lambda p: p.requires_grad, net_g.trainable_params()), lr, 0.5, 0.9)
    optimizer_d = nn.Adam(net_d.trainable_params(), lr, 0.5, 0.9)
    train_discriminator = TrainOneStepD(netd_with_loss, optimizer_d)
    train_generator = TrainOneStepG(netg_with_loss, optimizer_g)

    # Train
    train_discriminator.set_train()
    train_generator.set_train()
    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        for batch_idx, image in enumerate(dataset):
            s = time.time()
            real = image['image']
            real = real.astype(mindspore.float32)
            mask = random_mask(args)
            x = real * (1 - mask)
            for _ in range(args.dis_iter):
                netd_loss = train_discriminator(real, x, mask)
            netg_loss = train_generator(real, x, mask)
            gap = time.time() - s
            # Print losses
            print('epoch{}/{}, batch{}/{}, d_loss is {:.4f}, g_loss is {:.4f}, time is {:.4f}'.format(
                epoch + 1, args.epochs, batch_idx + 1, total_batch, netd_loss.asnumpy(), netg_loss.asnumpy(), gap))
            if args.run_distribute:
                save_checkpoint_path = args.save_folder + str(get_rank())
            else:
                save_checkpoint_path = args.save_folder
            if not os.path.isdir(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)
            # Save checkpoint
            gen_name = 'generator_epoch%d_batch%d.ckpt' % (epoch + 1, batch_idx + 1)
            dis_name = 'discriminator_epoch%d_batch%d.ckpt' % (epoch + 1, batch_idx + 1)
            gen_name = os.path.join(save_checkpoint_path, gen_name)
            dis_name = os.path.join(save_checkpoint_path, dis_name)
            if (batch_idx + 1) == total_batch:
                save_checkpoint(train_generator, gen_name)
                save_checkpoint(train_discriminator, dis_name)


if __name__ == '__main__':
    trainer(cra_config)
