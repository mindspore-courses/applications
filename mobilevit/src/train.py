# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.lr_schedule.py
# org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" mobilevit training script. """

import argparse

import mindspore
import mindspore.dataset.vision.c_transforms as c_transforms
import mindspore.dataset.vision.py_transforms as p_transforms
from mindspore import nn
from mindspore import context
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor

from models.mobilevit import MobileViT
from datasets.imagenet import ImageNet
from utils.monitor import LossMonitor
from utils.CrossEntropyEsmooth import CrossEntropySmooth
set_seed(1)


def mobilevit_train(args_opt):
    """mobilevit train"""

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(enable_graph_kernel=False)

    # Data preprocessing
    if args_opt.model_type == 'small':
        img_transforms = ([
            c_transforms.Decode(),
            c_transforms.RandomResizedCrop(256),
            c_transforms.RandomHorizontalFlip(),
            c_transforms.AutoAugment(),
            p_transforms.RandomErasing(prob=0.25),
            c_transforms.ConvertColor(c_transforms.ConvertMode.COLOR_RGB2BGR),
            p_transforms.ToTensor(),
        ])
    else:
        img_transforms = ([
            c_transforms.Decode(),
            c_transforms.RandomResizedCrop(256),
            c_transforms.RandomHorizontalFlip(),
            c_transforms.ConvertColor(c_transforms.ConvertMode.COLOR_RGB2BGR),
            p_transforms.ToTensor(),
        ])

    # dataset pipline
    dataset = ImageNet(args_opt.data_url,
                       split="train",
                       shuffle=True,
                       transform=img_transforms,
                       num_parallel_workers=args_opt.num_parallel_workers,
                       resize=args_opt.resize,
                       batch_size=args_opt.batch_size)

    dataset_train = dataset.run()
    step_size = dataset_train.get_dataset_size()

    # Create model.
    network = MobileViT(model_type=args_opt.model_type, num_classes=args_opt.num_classes)

    # Define the decreasing learning rate
    lr = nn.cosine_decay_lr(min_lr=args_opt.min_lr,
                            max_lr=args_opt.max_lr,
                            total_step=args_opt.epoch_size * step_size,
                            step_per_epoch=step_size,
                            decay_epoch=args_opt.decay_epoch)

    # Define loss scale
    loss_scale = 1024.0
    loss_scale_manager = mindspore.FixedLossScaleManager(loss_scale, False)

    # Define optimizer.
    network_opt = nn.SGD(network.trainable_params(), lr, momentum=args_opt.momentum, weight_decay=args_opt.weight_decay,
                         nesterov=False, loss_scale=loss_scale)

    # Define loss function.
    network_loss = CrossEntropySmooth(sparse=True,
                                      reduction="mean",
                                      smooth_factor=0.1,
                                      classes_num=args_opt.num_classes)

    # Define checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=args_opt.model_type, directory=args_opt.ckpt_save_dir, config=ckpt_config)

    # Define metrics.
    metrics = {'acc', "loss"}

    # Define timer
    time_cb = TimeMonitor(data_size=dataset_train.get_dataset_size())

    # Init the model.
    if args_opt.device_target == "Ascend":
        model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics, amp_level="auto",
                      loss_scale_manager=loss_scale_manager)
    else:
        model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    # Begin to train.
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[time_cb, ckpt_callback, LossMonitor(lr)],
                dataset_sink_mode=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT train.')
    parser.add_argument('--epoch_size', type=int, default=200, help='Train epoch size.')
    parser.add_argument('--model_type', default='xx_small', type=str, metavar='model_type')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size.')
    parser.add_argument('--decay_epoch', type=int, default=150, help='Number of decay epochs.')
    parser.add_argument('--num_classes', type=int, default=1001, help='Number of classification.')
    parser.add_argument('--data_url', default=r"C:\Users\Administrator\Desktop\MobileViT修改版\src\dataset",
                        help='Location of data.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the moving average.')
    parser.add_argument('--device_target', type=str, default="CPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--max_lr', type=float, default=0.1, help='Number of the maximum learning rate.')
    parser.add_argument('--num_parallel_workers', type=int, default=5, help='Number of parallel workers.')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Number of the minimum learning rate.')
    parser.add_argument('--resize', type=int, default=256, help='Resize the height and weight of picture.')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='Momentum for the moving average.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=40, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./Mobilevit_Ckpt/6", help='Location of training outputs.')
    args = parser.parse_known_args()[0]
    mobilevit_train(args)
