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
"""Train Pix2Pix model."""

import os
import datetime

import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint
from mindspore.communication.management import init, get_group_size
from mindspore.context import ParallelMode

from models.loss import WithLossCellD, LossD, WithLossCellG, LossG, TrainOneStepCell
from models.pix2pix import Pix2Pix, get_generator, get_discriminator
from process_datasets.dataset import Pix2PixDataset, create_train_dataset
from pix2pix_utils.tools import save_losses, save_image, get_lr
from config.pix2pix_config import parse_args


def train(arg):
    """Train Pix2Pix model."""

    # Preprocess the data for training
    dataset = Pix2PixDataset(root_dir=arg.train_data_dir, config=arg)
    ds = create_train_dataset(dataset, batch_size=arg.batch_size)
    print("ds:", ds.get_dataset_size())
    print("ds:", ds.get_col_names())
    print("ds.shape:", ds.output_shapes())

    steps_per_epoch = ds.get_dataset_size()
    context.set_context(mode=context.GRAPH_MODE)
    if arg.device_target == 'Ascend':
        if arg.run_distribute:
            print("Ascend distribute")
            context.set_context(device_id=arg.device_id, device_target="Ascend")
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True, device_num=arg.device_num)
            init()
            rank = arg.rank_id
        else:
            context.set_context(device_id=arg.device_id, device_target="Ascend")
    elif arg.device_target == 'GPU':
        if arg.run_distribute:
            print("GPU distribute")
            init()
            context.set_context(device_target="GPU")
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        else:
            context.set_context(device_id=arg.device_id, device_target="GPU")

    # Network
    net_generator = get_generator(arg)
    net_discriminator = get_discriminator(arg)
    pix2pix = Pix2Pix(generator=net_generator, discriminator=net_discriminator)

    d_loss_fn = LossD(arg)
    g_loss_fn = LossG(arg)
    d_loss_net = WithLossCellD(backbone=pix2pix, loss_fn=d_loss_fn)
    g_loss_net = WithLossCellG(backbone=pix2pix, loss_fn=g_loss_fn)

    d_opt = nn.Adam(pix2pix.net_discriminator.trainable_params(), learning_rate=get_lr(arg),
                    beta1=arg.beta1, beta2=arg.beta2, loss_scale=1)
    g_opt = nn.Adam(pix2pix.net_generator.trainable_params(), learning_rate=get_lr(arg),
                    beta1=arg.beta1, beta2=arg.beta2, loss_scale=1)
    # TrainOneStepCell Perform single-step training of the network and return the loss results after each training result.
    train_net = TrainOneStepCell(loss_netd=d_loss_net, loss_netg=g_loss_net, optimizerd=d_opt, optimizerg=g_opt, sens=1)
    train_net.set_train()

    if not os.path.isdir(arg.train_fakeimg_dir):
        os.makedirs(arg.train_fakeimg_dir)
    if not os.path.isdir(arg.loss_show_dir):
        os.makedirs(arg.loss_show_dir)
    if not os.path.isdir(arg.ckpt_dir):
        os.makedirs(arg.ckpt_dir)

    # Training loop
    g_losses = []
    d_losses = []

    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=arg.epoch_num)
    print("Starting Training Loop...")
    if arg.run_distribute:
        rank = arg.rank_id
    for epoch in range(arg.epoch_num):
        for i, data in enumerate(data_loader):
            start_time = datetime.datetime.now()
            input_image = Tensor(data["input_images"])
            target_image = Tensor(data["target_images"])
            dis_loss, gen_loss = train_net(input_image, target_image)
            end_time = datetime.datetime.now()
            delta = (end_time - start_time).microseconds
            if i % 100 == 0:
                print("================start===================")
                print("Date time: ", start_time)
                if arg.run_distribute:
                    print("Device ID :", str(rank))
                print("ms per step :", delta / 1000)
                print("epoch: ", epoch + 1, "/", arg.epoch_num)
                print("step: ", i, "/", steps_per_epoch)
                print("Dloss: ", dis_loss)
                print("Gloss: ", gen_loss)
                print("=================end====================")

            # Save fake_imgs
            if i == steps_per_epoch - 1:
                fake_image = net_generator(input_image)
                if arg.run_distribute:
                    fakeimg_path = arg.train_fakeimg_dir + str(rank) + '/'
                    if not os.path.isdir(fakeimg_path):
                        os.makedirs(fakeimg_path)
                    save_image(fake_image, fakeimg_path + str(epoch + 1))
                else:
                    save_image(fake_image, arg.train_fakeimg_dir + str(epoch + 1))
                print("image generated from epoch", epoch + 1, "saved")
                print("The learning rate at this point is：", get_lr(arg)[epoch * i])

            d_losses.append(dis_loss.asnumpy())
            g_losses.append(gen_loss.asnumpy())

        print("epoch", epoch + 1, "saved")
        # Save losses
        save_losses(g_losses, d_losses, epoch + 1, arg)
        print("epoch", epoch + 1, "D&G_Losses saved")
        print("epoch", epoch + 1, "finished")
        # Save checkpoint
        if (epoch + 1) == arg.epoch_num:
            if arg.run_distribute:
                save_checkpoint_path = arg.ckpt_dir + str(rank) + '/'
                if not os.path.isdir(save_checkpoint_path):
                    os.makedirs(save_checkpoint_path)
                save_checkpoint(net_generator, os.path.join(save_checkpoint_path, f"Generator_{epoch + 1}.ckpt"))
            else:
                save_checkpoint(net_generator, os.path.join(arg.ckpt_dir, f"Generator_{epoch + 1}.ckpt"))
            print("ckpt generated from epoch", epoch + 1, "saved")


if __name__ == '__main__':
    train(parse_args())
