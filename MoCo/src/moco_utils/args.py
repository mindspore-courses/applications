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
""" MoCo args."""

import argparse


def parse_args():
    """
    Set network parameters

    Returns:
        ArgumentParser. Parameter information
    """
    parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'])
    parser.add_argument('-a', '--arch', default='resnet18')
    parser.add_argument('--dataset', default='Cifar10', help='dataset name')
    parser.add_argument('--lr', '--learning-rate', default=0.06, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x);'
                             ' does not take effect if --cos is on')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco-m', default=0.9, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

    parser.add_argument('--bn-splits', default=8, type=int,
                        help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

    parser.add_argument('--symmetric', action='store_true',
                        help='use a symmetric loss function that backprops to both crops')

    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
    parser.add_argument('--knn-t', default=0.1, type=float,
                        help='softmax temperature in kNN monitor; could be different with moco-t')

    parser.add_argument('--resume', default='moco.ckpt', type=str, metavar='moco.ckpt',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

    return parser.parse_args()
