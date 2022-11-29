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
"""train scripts"""

import sys
import os
import argparse
import time

import numpy as np
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank
from mindspore import context
from mindspore import load_checkpoint, save_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
import mindspore.ops as ops
from mindspore.common import set_seed

from src.model.generator import get_generator
from src.model.discriminator import get_discriminator
from src.dataset.create_loader import create_train_dataloader, create_test_dataloader
from src.loss.psnr_loss import PSNRLoss
from src.loss.gan_loss import DiscriminatorLoss, GeneratorLoss
from src.vgg19.define import vgg19
from src.train.train_psnr import TrainOnestepPSNR
from src.train.train_gan import TrainOneStepD, TrainOnestepG

current_dir = os.path.split(os.path.abspath(__file__))[0]
config_path = current_dir.rsplit('/', 2)[0]
sys.path.append(config_path)

def main(args):
    """Training process"""
    set_seed(2022)
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=args.platform)

    if args.run_distribute == 1:
        if args.platform == 'Ascend':
            if args.device_id == 0:
                context.set_context(device_id=int(os.getenv("DEVICE_ID", "0")))
            else:
                context.set_context(device_id=args.device_id)
        device_num = args.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()
        rank = get_rank()
    else:
        if args.platform in ['GPU', 'Ascend']:
            rank = 0
            if args.device_id == 0:
                if os.getenv("DEVICE_ID", "not_set").isdigit():
                    context.set_context(device_id=int(os.getenv("DEVICE_ID", "0")))
            else:
                context.set_context(device_id=args.device_id)

    # for srresnet
    # create dataset
    train_ds = create_train_dataloader(args.train_batch_size, args.train_LR_path, args.train_GT_path, rank,
                                       args.device_num)
    test_ds = create_test_dataloader(args.val_batch_size, args.val_LR_path, args.val_GT_path)
    train_data_loader = train_ds.create_dict_iterator()
    test_data_loader = test_ds.create_dict_iterator()

    # definition of network
    generator = get_generator(4, 0.02)

    # network with loss
    psnr_loss = PSNRLoss(generator)

    # optimizer
    psnr_optimizer = nn.Adam(generator.trainable_params(), 1e-4)

    # operation for testing
    op = ops.ReduceSum(keep_dims=False)

    # trainonestep
    train_psnr = TrainOnestepPSNR(psnr_loss, psnr_optimizer)
    train_psnr.set_train()

    bestpsnr = 0
    if not os.path.exists("./ckpt"):
        os.makedirs("./ckpt")

    # warm up generator
    for epoch in range(args.start_psnr_epoch, args.psnr_epochs):
        print("Generator: training {:d} epoch:".format(epoch + 1))
        time_begin = time.time()
        for data in train_data_loader:
            lr = data['LR']
            hr = data['HR']
            mse_loss = train_psnr(hr, lr)
        steps = train_ds.get_dataset_size()
        time_elapsed = time.time() - time_begin
        step_time = time_elapsed / steps
        print('per step needs time:{:.0f}ms'.format(step_time * 1000))
        print("mse_loss:", mse_loss)
        psnr_list = []

        # val for every epoch
        for test in test_data_loader:
            lr = test['LR']
            gt = test['HR']

            _, _, h, w = lr.shape[:4]
            gt = gt[:, :, : h * args.scale, : w * args.scale]

            output = generator(lr)
            output = op(output, 0)
            output = output.asnumpy()
            output = np.clip(output, -1.0, 1.0)
            gt = op(gt, 0)

            output = (output + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

            output = output.transpose(1, 2, 0)
            gt = gt.asnumpy()
            gt = gt.transpose(1, 2, 0)

            y_output = rgb2ycbcr(output)[args.scale: -args.scale, args.scale: -args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale: -args.scale, args.scale: -args.scale, :1]
            psnr = peak_signal_noise_ratio(y_output/255.0, y_gt/255.0, data_range=1.0)
            psnr_list.append(psnr)

        mean = np.mean(psnr_list)
        print("psnr:", mean)
        if mean > bestpsnr:
            bestpsnr = mean
            if args.run_distribute == 0:
                save_checkpoint(train_psnr, "./ckpt/best.ckpt")
            else:
                if rank == 0:
                    save_checkpoint(train_psnr, "./ckpt/best.ckpt")
        if (epoch + 1) % 200 == 0:
            if args.run_distribute == 0:
                save_checkpoint(train_psnr, './ckpt/pre_trained_model_%03d.ckpt' % (epoch + 1))
            else:
                if rank == 0:
                    save_checkpoint(train_psnr, './ckpt/pre_trained_model_%03d.ckpt' % (epoch + 1))

        print("{:d}/2000 epoch finished".format(epoch + 1))

    # for srgan
    generator = get_generator(4, 0.02)
    discriminator = get_discriminator(96, 0.02)
    if args.platform == "Ascend":
        if args.run_distribute == 0:
            ckpt = "./ckpt/best.ckpt"
        else:
            ckpt = '../train_parallel0/ckpt/best.ckpt'
    if args.platform == "GPU":
        ckpt = "./ckpt/best.ckpt"
    params = load_checkpoint(ckpt)
    load_param_into_net(generator, params)
    discriminator_loss = DiscriminatorLoss(discriminator, generator)
    vgg = vgg19(args.vgg_ckpt)
    generator_loss = GeneratorLoss(discriminator, generator, vgg)
    generator_optimizer = nn.Adam(generator.trainable_params(), 1e-4)
    discriminator_optimizer = nn.Adam(discriminator.trainable_params(), 1e-4)
    train_discriminator = TrainOneStepD(discriminator_loss, discriminator_optimizer)
    train_generator = TrainOnestepG(generator_loss, generator_optimizer)

    # trainGAN
    for epoch in range(args.start_gan_epoch, args.gan_epochs):
        print('Gan: training {:d} epoch'.format(epoch + 1))
        train_begin = time.time()
        for data in train_data_loader:
            lr = data['LR']
            hr = data['HR']
            d_loss = train_discriminator(hr, lr)
            g_loss = train_generator(hr, lr)
        time_elapsed = time.time() - train_begin
        steps = train_ds.get_dataset_size()
        step_time = time_elapsed / steps
        print('per step needs time:{:.0f}ms'.format(step_time * 1000))
        print("D_loss:", d_loss.mean())
        print("G_loss:", g_loss.mean())

        if (epoch + 1) % 100 == 0:
            if args.run_distribute == 0:
                save_checkpoint(train_generator, './ckpt/G_model_%03d.ckpt' % (epoch + 1))
                save_checkpoint(train_discriminator, './ckpt/D_model_%03d.ckpt' % (epoch + 1))
            else:
                if rank == 0:
                    save_checkpoint(train_generator, './ckpt/G_model_%03d.ckpt' % (epoch + 1))
                    save_checkpoint(train_discriminator, './ckpt/D_model_%03d.ckpt' % (epoch + 1))
        print(" {:d}/1000 epoch finished".format(epoch + 1))

def parse_args():
    """Add argument"""
    parser = argparse.ArgumentParser(description="SRGAN train")
    parser.add_argument("--train_LR_path", type=str)
    parser.add_argument("--train_GT_path", type=str)
    parser.add_argument("--val_LR_path", type=str)
    parser.add_argument("--val_GT_path", type=str)
    parser.add_argument("--vgg_ckpt", type=str)
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor')
    parser.add_argument("--image_size", type=int, default=96,
                        help="Image size of high resolution image. (default: 96)")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        metavar="N", help="batch size for training")
    parser.add_argument("--val_batch_size", default=1, type=int,
                        metavar="N", help="batch size for tesing")
    parser.add_argument("--psnr_epochs", default=2000, type=int, metavar="N",
                        help="Number of total psnr epochs to run. (default: 2000)")
    parser.add_argument("--start_psnr_epoch", default=0, type=int, metavar='N',
                        help="Manual psnr epoch number (useful on restarts). (default: 0)")
    parser.add_argument("--gan_epochs", default=1000, type=int, metavar="N",
                        help="Number of total gan epochs to run. (default: 1000)")
    parser.add_argument("--start_gan_epoch", default=0, type=int, metavar='N',
                        help="Manual gan epoch number (useful on restarts). (default: 0)")
    parser.add_argument("--init_type", type=str, default='normal', choices=("normal", "xavier"), \
                        help="network initialization, default is normal.")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--platform", type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'))
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default: 0.")
    # distribute
    parser.add_argument("--run_distribute", type=int, default=0, help="Run distribute, default: 0.")
    parser.add_argument("--device_num", type=int, default=1, help="number of device, default: 1.")
    return parser.parse_args()

if __name__ == '__main__':
    args_list = parse_args()
    main(args_list)
