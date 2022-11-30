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
"""Parse parameters function."""

import ast
import argparse


def parse_args(phase):
    """
    Parse parameters.

    Help description for each configuration:
        phase(str): Choose train or predict mode. Default: None.
        device_id(int): Device id of GPU or Ascend. Default: 0.
        platform(str): Train on GPU ar Ascend, only support GPU and Ascend. Default: GPU.
        device_num(int): Device num support distribute. Default: 1.
        is_save_on_master(bool): Save ckpt on master or all rank, 1 for master, 0 for all ranks. Default: 1.
        rank(int): Local rank of distributed. Default: 0.
        group_size(int): World size of device. Default: 1.
        use_random(bool): If you use random when training. Default: True.
        save_graphs(bool): Whether save graphs. Default: False.
        need_profiler(bool): Whether you need profiler. Default: False.
        max_dataset_size(int): Max images pre epoch. Default: None.
        batch_size(int): Batch size for train and predict. Default: 1.
        beta1(float): Adam beta1. Default: 0.5.
        max_epoch(int): Epoch size for training. default: 200.
        n_epochs(int): Number of epochs with the initial learning rate. Default: 100.
        load_ckpt(bool): Whether load pretrained ckpt. Default: False.

        in_planes(int): The number of channels in input images. Default=3.
        ngf(int): Generator model filter numbers. Default: 64.
        gl_num(int): Generator model residual block numbers. default: 9.
        ndf(int): Discriminator model filter numbers. Default: 64.
        dl_num(int): Discriminator model residual block numbers. Default is 3.
        slope(float): Leaky Relu slope. default: 0.2.
        norm_mode(str): Norm mode. default: 'batch'.
        lambda_a(float): Weight for cycle loss (a -> b -> a). Default: 10.
        lambda_b(float): Weight for cycle loss (b -> a -> b). Default: 10.
        lambda_idt(float): Use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the
                weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times
                smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1. Default: 0.5.
        gan_mode(str): The type of GAN loss, default is 'lsgan'.
        pad_mode(str): The type of Pad. Default: 'CONSTANT'.
        init_type(str): Network initialization. Default: 'normal'.
        init_gain(float): Scaling factor for normal, xavier and orthogonal. Default: 0.02.
        need_dropout(bool): Whether you need dropout. Default: True.

        lr(float): Learning rate. Default: 0.0002.
        lr_policy(bool): Learning rate policy. Default: 'linear'.
        print_iter(int): Log print iter. Default: 100.
        pool_size(int): The size of image buffer that stores previously generated images. Default: 50.
        save_checkpoint_epochs(int): Save checkpoint epochs. Default: 10.
        save_imgs(bool): Whether save images when epoch end. Default: True.

        dataroot(str): Path of images (should have sub folders trainA, trainB, testA, testB, etc).
        Default: "./data/horse2zebra".
        image_size(int): Input image_size. Default: 256.

        data_dir(str): The translation direction of CycleGAN. Default: 'testA'.
        outputs_log(str): Logs are saved here. Default: './outputs/log'.
        outputs_ckpt(str): Ckpts are saved here. Default: './outputs/ckpt'.
        outputs_imgs(str): Images are saved here. Default: './outputs/imas'.
        outputs_dir(str): Predicted images are saved here. Default: './outputs'.
        g_a_ckpt(str): Checkpoint file path of g_a. Default: './outputs/ckpt/g_a_200.ckpt'.
        g_b_ckpt(str): Checkpoint file path of g_b. Default: './outputs/ckpt/g_b_200.ckpt'.
        d_a_ckpt(str): Checkpoint file path of d_a. Default: './outputs/ckpt/d_a_200.ckpt'.
        d_b_ckpt(str): Checkpoint file path of d_b. Default: './outputs/ckpt/d_b_200.ckpt'.

     Returns:
        parsed parameters.
    """

    parser = argparse.ArgumentParser(description='Cycle GAN')

    # train
    parser.add_argument('--phase', type=str, default=None)
    parser.add_argument('--platform', type=str, default='GPU')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--is_save_on_master', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--use_random', type=ast.literal_eval, default=True)
    parser.add_argument('--save_graphs', type=ast.literal_eval, default=False)
    parser.add_argument('--need_profiler', type=ast.literal_eval, default=False)
    parser.add_argument('--max_dataset_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--load_ckpt', type=ast.literal_eval, default=False)

    # network
    parser.add_argument('--in_planes', type=int, default=3)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--gl_num', type=int, default=9)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--dl_num', type=int, default=3)
    parser.add_argument('--slope', type=float, default=0.2)
    parser.add_argument('--norm_mode', type=str, default='batch', choices=('batch', 'instance'))
    parser.add_argument('--lambda_a', type=float, default=10.0)
    parser.add_argument('--lambda_b', type=float, default=10.0)
    parser.add_argument('--lambda_idt', type=float, default=0.5)
    parser.add_argument('--gan_mode', type=str, default='lsgan', choices=('lsgan', 'vanilla'))
    parser.add_argument('--pad_mode', type=str, default='CONSTANT', choices=('CONSTANT', 'REFLECT', 'SYMMETRIC'))
    parser.add_argument('--init_type', type=str, default='normal', choices=('normal', 'xavier'))
    parser.add_argument('--init_gain', type=float, default=0.02)
    parser.add_argument('--need_dropout', type=ast.literal_eval, default=False)

    # tools
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lr_policy', type=str, default='linear', choices=('linear', 'constant'))
    parser.add_argument('--print_iter', type=int, default=100)
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--save_checkpoint_epochs', type=int, default=10)
    parser.add_argument('--save_imgs', type=ast.literal_eval, default=True)

    # data
    parser.add_argument('--dataroot', type=str, default="./data/horse2zebra")
    parser.add_argument('--image_size', type=int, default=256)

    # infer
    parser.add_argument('--data_dir', default='testA', choices=('testA', 'testB'))
    parser.add_argument('--outputs_log', type=str, default='./outputs/log')
    parser.add_argument('--outputs_ckpt', type=str, default='./outputs/ckpt')
    parser.add_argument('--outputs_imgs', type=str, default='./outputs/imgs')
    parser.add_argument('--outputs_dir', type=str, default='./outputs')
    parser.add_argument('--g_a_ckpt', type=str, default='./outputs/ckpt/g_a_200.ckpt')
    parser.add_argument('--g_b_ckpt', type=str, default='./outputs/ckpt/g_b_200.ckpt')
    parser.add_argument('--d_a_ckpt', type=str, default='./outputs/ckpt/d_a_200.ckpt')
    parser.add_argument('--d_b_ckpt', type=str, default='./outputs/ckpt/d_b_200.ckpt')

    args = parser.parse_known_args()[0]
    args.phase = phase
    return args
