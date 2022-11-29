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
"""Build and train model."""

import argparse
import os

import cv2
from tqdm import tqdm
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import float32 as dtype

from losses.loss import GeneratorLoss, DiscriminatorLoss
from models.animegan import AnimeGAN
from models.discriminator import Discriminator
from models.generator import Generator
from process_datasets.animeganv2_dataset import AnimeGANDataset
from animeganv2_utils.pre_process import denormalize_input, check_params


def parse_args():
    """Argument parsing"""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--device_target', default='GPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--dataset', default='Hayao', choices=['Hayao', 'Shinkai', 'Paprika'], type=str)
    parser.add_argument('--data_dir', default='../dataset', type=str)
    parser.add_argument('--checkpoint_dir', default='../checkpoints', type=str)
    parser.add_argument('--vgg19_path', default='../vgg.ckpt', type=str)
    parser.add_argument('--save_image_dir', default='../images', type=str)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--init_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_parallel_workers', default=1, type=int)
    parser.add_argument('--save_interval', default=1, type=int)
    parser.add_argument('--debug_samples', default=0, type=int)
    parser.add_argument('--lr_g', default=2.0e-4, type=float)
    parser.add_argument('--lr_d', default=4.0e-4, type=float)
    parser.add_argument('--init_lr', default=1.0e-3, type=float)
    parser.add_argument('--gan_loss', default='lsgan', choices=['lsgan', 'hinge', 'bce'], type=str)
    parser.add_argument('--wadvg', default=0.9, type=float, help='Adversarial loss weight for G')
    parser.add_argument('--wadvd', default=300, type=float, help='Adversarial loss weight for D')
    parser.add_argument('--wcon', default=1.8, type=float, help='Content loss weight')
    parser.add_argument('--wgra', default=2.0, type=float, help='Gram loss weight')
    parser.add_argument('--wcol', default=10.0, type=float, help='Color loss weight')
    parser.add_argument('--img_ch', default=3, type=int, help='The size of image channel')
    parser.add_argument('--ch', default=64, type=int, help='Base channel number per layer')
    parser.add_argument('--n_dis', default=3, type=int, help='The number of discriminator layer')
    return parser.parse_args()


def main():
    """Build and train model."""
    check_params(args)
    print("Init models...")
    generator = Generator()
    discriminator = Discriminator(args.ch, args.n_dis)
    optimizer_g = nn.Adam(generator.trainable_params(), learning_rate=args.lr_g, beta1=0.5, beta2=0.999)
    optimizer_d = nn.Adam(discriminator.trainable_params(), learning_rate=args.lr_d, beta1=0.5, beta2=0.999)
    net_d_with_criterion = DiscriminatorLoss(discriminator, generator, args)
    net_g_with_criterion = GeneratorLoss(discriminator, generator, args)
    my_train_one_step_cell_for_d = nn.TrainOneStepCell(net_d_with_criterion, optimizer_d)
    my_train_one_step_cell_for_g = nn.TrainOneStepCell(net_g_with_criterion, optimizer_g)
    animegan = AnimeGAN(my_train_one_step_cell_for_d, my_train_one_step_cell_for_g)
    animegan.set_train()

    data = AnimeGANDataset(args)
    data = data.run()

    size = data.get_dataset_size()

    for epoch in range(args.epochs):
        iters = 0
        for img, anime, anime_gray, anime_smt_gray in tqdm(data.create_tuple_iterator()):
            img = Tensor(img, dtype=dtype)
            anime = Tensor(anime, dtype=dtype)
            anime_gray = Tensor(anime_gray, dtype=dtype)
            anime_smt_gray = Tensor(anime_smt_gray, dtype=dtype)
            net_d_loss, net_g_loss = animegan(img, anime, anime_gray, anime_smt_gray)
            if iters % 50 == 0:
                # Output training records
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (
                    epoch + 1, args.epochs, iters, size, net_d_loss.asnumpy().min(), net_g_loss.asnumpy().min()))
            # Save training images and checkpoint
            if (epoch % args.save_interval) == 0 and (iters == size - 1):
                stylized = denormalize_input(generator(img)).asnumpy()
                no_stylized = denormalize_input(img).asnumpy()
                imgs = cv2.cvtColor(stylized[0, :, :, :].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                imgs1 = cv2.cvtColor(no_stylized[0, :, :, :].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                for i in range(1, args.batch_size):
                    imgs = np.concatenate(
                        (imgs, cv2.cvtColor(stylized[i, :, :, :].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)), axis=1)
                    imgs1 = np.concatenate(
                        (imgs1, cv2.cvtColor(no_stylized[i, :, :, :].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)), axis=1)
                cv2.imwrite(
                    os.path.join(args.save_image_dir, args.dataset, 'epoch_' + str(epoch) + '.jpg'),
                    np.concatenate((imgs1, imgs), axis=0))
                mindspore.save_checkpoint(generator, os.path.join(args.checkpoint_dir, args.dataset,
                                                                  'netG_' + str(epoch) + '.ckpt'))
            iters += 1


if __name__ == '__main__':
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    main()
