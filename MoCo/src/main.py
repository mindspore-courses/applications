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
""" main function script of MoCo conclude train and eval."""

import json
import os
from datetime import datetime
import pandas as pd

import mindspore
from mindspore import context, nn
import mindspore.dataset

from model.moco import ModelMoCo
from dataset.dataset_test import create_dataset
from moco_utils.args import parse_args
from train import train
from eval import test


def main():
    """MoCo train and eval."""
    args = parse_args()
    args.symmetric = False
    if args.results_dir == '':
        args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
    print(args)

    # Get the validation set
    train_data, memory_data, test_data = create_dataset(args.dataset)
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    # Create model.
    models = ModelMoCo(i=args.moco_k, m=args.moco_m,
                       t=args.moco_t, symmetric=args.symmetric)

    epoch_start = 1

    if os.path.exists(args.resume):
        checkpoint = mindspore.load_checkpoint(args.resume)
        mindspore.load_param_into_net(models, checkpoint)
        print('Loaded from: {}'.format(args.resume))

    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    for epoch in range(epoch_start, args.epochs + 1):

        exponential_decay_lr = nn.ExponentialDecayLR(0.06, 0.8, 10)
        lr = exponential_decay_lr(epoch)
        optimizer = nn.SGD(params=models.trainable_params(),
                           learning_rate=lr, weight_decay=args.wd, momentum=0.9)

        train_net = nn.TrainOneStepCell(models, optimizer)
        train_loss = train(train_net, train_data, epoch, args, lr)
        results['train_loss'].append(train_loss)

        test_acc_1 = test(models.encoder_q, memory_data, test_data, epoch, args)
        results['test_acc@1'].append(test_acc_1)

        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
        mindspore.save_checkpoint(models, "moco.ckpt")


if __name__ == '__main__':
    main()
