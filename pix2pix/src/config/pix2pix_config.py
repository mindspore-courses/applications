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
# ===========================================================================
"""Parse parameters function."""

import argparse


def parse_args():
    """
    Parse parameters.

     Returns:
        parsed parameters.
    """

    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--train_data_dir', default='/home/pix2pix/end/datasets/facades/train/', type=str,
                        help="The file path of training input data.")
    parser.add_argument('--epoch_num', default=200, type=int,
                        help="Epoch number for training,different datasets have different values.")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="Batch size, different size datasets have different values.")
    parser.add_argument('--dataset_size', default=400, type=int,
                        help="Training dataset size, different size datasets have different values.")
    parser.add_argument('--val_data_dir', default='/home/pix2pix/end/datasets/facades/val/', type=str,
                        help="File path of validation input data.")
    parser.add_argument('--ckpt', default='results/ckpt/Generator_200.ckpt', type=str,
                        help="File path of checking point file used in validation.")
    parser.add_argument('--device_id', default=0, type=int, help="Get device id.")
    parser.add_argument('--device_num', default=1, type=int, help="Get device num.")
    parser.add_argument('--rank_id', default=0, type=int, help="Get rank id.")

    #train
    parser.add_argument('--device_target', default='GPU', choices=['GPU', 'Ascend'], type=str,
                        help="Device id of GPU or Ascend.")
    parser.add_argument('--train_fakeimg_dir', default='results/fake_img/', type=str,
                        help="File path of stored fake img in training.")
    parser.add_argument('--loss_show_dir', default='results/loss_show', type=str,
                        help="File path of stored loss img in training.")
    parser.add_argument('--ckpt_dir', default='results/ckpt', type=str,
                        help="File Path of stored checkpoint file in training.")
    parser.add_argument('--run_distribute', default=False, type=bool, help="Whether to run distribute.")
    parser.add_argument('--beta1', default=0.5, type=float, help="Adam beta1.")
    parser.add_argument('--beta2', default=0.999, type=float, help="Adam beta2.")

    #eval
    parser.add_argument('--predict_dir', default='results/predict/', type=str,
                        help="File path of generated image in validation.")

    #dataset
    parser.add_argument('--load_size', default=286, type=int, help="Scale images to this size.")
    parser.add_argument('--train_pic_size', default=256, type=int, help="The train image size.")
    parser.add_argument('--val_pic_size', default=256, type=int, help="The eval image size.")

    #loss
    parser.add_argument('--lambda_dis', default=0.5, type=float, help="Weight for discriminator loss.")
    parser.add_argument('--lambda_gan', default=0.5, type=float, help="Weight for GAN loss.")
    parser.add_argument('--lambda_l1', default=100, type=int, help="Weight for L1 loss.")

    #tools
    parser.add_argument('--lr', default=0.0002, type=float, help="Initial learning rate.")
    parser.add_argument('--n_epochs', default=100, type=int,
                        help="The number of epochs with the initial learning rate.")
    parser.add_argument('--n_epochs_decay', default=100, type=int,
                        help="The number of epochs with the dynamic learning rate.")

    #network
    parser.add_argument('--g_in_planes', default=3, type=int, help="The number of channels in input images.")
    parser.add_argument('--g_out_planes', default=3, type=int, help="The number of channels in output images.")
    parser.add_argument('--g_ngf', default=64, type=int, help="The number of filters in the last conv layer.")
    parser.add_argument('--g_layers', default=8, type=int, help="The number of downsamplings in UNet.")
    parser.add_argument('--d_in_planes', default=6, type=int, help="The input channel.")
    parser.add_argument('--d_ndf', default=64, type=int, help="The number of filters in the last conv layer.")
    parser.add_argument('--d_layers', default=3, type=int, help="The number of ConvNormRelu blocks.")
    parser.add_argument('--alpha', default=0.2, type=float, help="LeakyRelu slope.")
    parser.add_argument('--init_gain', default=0.02, type=float,
                        help="Scaling factor for normal xavier and orthogonal.")
    parser.add_argument('--pad_mode', default='CONSTANT', type=str, help="The pad mode.")
    parser.add_argument('--init_type', default='normal', type=str, help="The network init type.")

    args = parser.parse_known_args()[0]
    return args
