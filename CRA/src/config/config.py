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

import argparse


def parse_args():
    """Parse parameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', help='Image path of training input data.')
    parser.add_argument('--mask_template_dir', help='Mask path of training input data.')
    parser.add_argument('--save_folder', help='File path of stored checkpoint file in training.')
    parser.add_argument('--device_target', default='Ascend', type=str, choices=['GPU', 'Ascend'],
                        help='Training device.')
    parser.add_argument('--device_id', default=1, type=int, help='Get device id.')
    parser.add_argument('--device_num', type=int, default=8, help='Get device num.')

    #model
    parser.add_argument('--IMG_SHAPE', default=[512, 512, 3], help='Required dimensions of the network input tensor.')
    parser.add_argument('--attention_type', default='SOFT', type=str, help='compute attention type.')

    #loss
    parser.add_argument('--coarse_alpha', default=1.2, type=float,
                        help='Proportion of coarse output in loss calculation.')
    parser.add_argument('--gan_with_mask', default=False, type=bool,
                        help='Whether to concat mask when calculating adversarial loss.')
    parser.add_argument('--gan_loss_alpha', default=0.001, type=float,
                        help='Proportion of adversarial loss of generator.')
    parser.add_argument('--in_hole_alpha', default=1.2, type=float,
                        help='The influence of the generation results in the mask area on the loss value.')
    parser.add_argument('--context_alpha', default=1.2, type=float,
                        help='The influence of the generation results outside the mask area on the loss value.')
    parser.add_argument('--wgan_gp_lambda', default=10, type=int,
                        help='The influence of WGAN-GP loss on discriminator loss value.')

    #train
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_decrease_epoch', default=2, type=int, help='Number of epochs to decay over.')
    parser.add_argument('--lr_decrease_factor', default=0.5, type=float, help='The decay rate.')
    parser.add_argument('--run_distribute', default=False, type=bool, help='Whether to run distribute.')
    parser.add_argument('--train_batchsize', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=40, help='Epoch number for training.')
    parser.add_argument('--dis_iter', type=int, default=1,
                        help='Train once generator when training dis_iter times discriminator.')
    return parser.parse_args(args=[])

cra_config = parse_args()
