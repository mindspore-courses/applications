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
"""Train Retinaface_resnet50."""

from __future__ import print_function
import argparse
import math

import mindspore as ms
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from model.retinaface import RetinaFace, TrainingWrapperResNet, TrainingWrapperMobileNet
from model.loss_cell import RetinaFaceWithLossCell
from model.resnet50 import resnet50
from model.mobilenet025 import MobileNetV1
from process_datasets.widerface import create_dataset
from utils.multiboxloss import MultiBoxLoss
from utils.lr_schedule import adjust_learning_rate, warmup_cosine_annealing_lr


def train(cfg, is_distributed=False):
    """Train RetinaFace network."""
    print('train config:\n', cfg)
    ms.common.seed.set_seed(cfg['seed'])
    ms.set_context(mode=ms.GRAPH_MODE, device_target=cfg['device_target'], save_graphs=False,
                   device_id=cfg['device_id'])

    if ms.get_context("device_target") == "GPU":
        ms.set_context(enable_graph_kernel=True, graph_kernel_flags="--enable_parallel_fusion")

    if is_distributed:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
        ms.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
        cfg['ckpt_path'] = cfg['ckpt_path'] + "ckpt_" + str(get_rank()) + "/"
    else:
        rank_id = 0
        device_num = 1

    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']
    training_dataset = cfg['training_dataset']
    ds_train = create_dataset(training_dataset, cfg, batch_size, multiprocessing=True, num_worker=cfg['num_workers'],
                              num_shards=device_num, shard_id=rank_id)
    initial_lr = cfg['initial_lr']
    momentum = cfg['momentum']
    lr_type = cfg['lr_type']
    weight_decay = cfg['weight_decay']
    gamma = cfg['gamma']
    stepvalues = (cfg['decay1'], cfg['decay2'])
    steps_per_epoch = math.ceil(ds_train.get_dataset_size())
    warmup_epoch = cfg['warmup_epoch']

    print('dataset size is : \n', ds_train.get_dataset_size())

    multibox_loss = MultiBoxLoss(cfg['num_classes'], cfg['num_anchor'], cfg['negative_ratio'], cfg['batch_size'])
    if cfg['backbone'] == 'resnet50':
        backbone = resnet50()
        net = RetinaFace(phase='train', backbone=backbone)
        backbone.set_train(True)
        net.set_train(True)

        loss_scale = cfg['loss_scale']
        t_max = cfg['T_max']
        eta_min = cfg['eta_min']

        if lr_type == 'dynamic_lr':
            lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch,
                                      warmup_epoch=warmup_epoch, lr_type1=lr_type)
        elif lr_type == 'cosine_annealing':
            lr = warmup_cosine_annealing_lr(initial_lr, steps_per_epoch, warmup_epoch, max_epoch, t_max, eta_min)

        if cfg['optim'] == 'momentum':
            opt = nn.Momentum(net.trainable_params(), lr, momentum, weight_decay, loss_scale)
        elif cfg['optim'] == 'sgd':
            opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
                         weight_decay=weight_decay, loss_scale=loss_scale)
        else:
            raise ValueError('optim is not define.')
    elif cfg['backbone'] == 'mobilenet025':
        backbone = MobileNetV1()
        net = RetinaFace(phase='train', backbone=backbone, cfg=cfg)
        backbone.set_train(True)
        net.set_train(True)

        lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch,
                                  warmup_epoch=warmup_epoch, lr_type1=lr_type)

        if cfg['optim'] == 'momentum':
            opt = nn.Momentum(net.trainable_params(), lr, momentum)
        elif cfg['optim'] == 'sgd':
            opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
                         weight_decay=weight_decay, loss_scale=1)
        else:
            raise ValueError('optim is not define.')
    else:
        raise ValueError('backbone is not define.')

    if cfg['pretrain']:
        pretrained_model = cfg['pretrain_path']
        param_dict_res50 = ms.load_checkpoint(pretrained_model)
        ms.load_param_into_net(backbone, param_dict_res50)
        print('Load resnet50 from [{}] done.'.format(pretrained_model))
    net = RetinaFaceWithLossCell(net, multibox_loss, cfg)
    if cfg['backbone'] == 'resnet50':
        net = TrainingWrapperResNet(net, opt)
    elif cfg['backbone'] == 'mobilenet025':
        net = TrainingWrapperMobileNet(net, opt)
    model = Model(net)

    config_ck = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_steps'],
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpoint_cb]
    print("============== Starting Training ==============")
    model.train(max_epoch, ds_train, callbacks=callback_list,
                dataset_sink_mode=True)


def parse_args():
    """Parse configuration arguments for training."""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--backbone', default='resnet50', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--clip', default=False, type=bool)
    parser.add_argument('--loc_weight', default=2.0, type=float)
    parser.add_argument('--class_weight', default=1.0, type=float)
    parser.add_argument('--landm_weight', default=1.0, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_anchor', default=29126, type=int)
    parser.add_argument('--image_size', default=840, type=int)
    parser.add_argument('--match_thresh', default=0.35, type=float)
    parser.add_argument('--in_channel', default=256, type=int)
    parser.add_argument('--out_channel', default=256, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--epoch', default=120, type=int)
    parser.add_argument('--initial_lr', default=0.0001, type=float)
    parser.add_argument('--ckpt_path', default='./ckps/', type=str)
    parser.add_argument('--save_checkpoint_steps', default=1600, type=int)
    parser.add_argument('--keep_checkpoint_max', default=3, type=int)
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', type=str)
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--pretrain_path', default='./pretrained_model/res50_pretrain.ckpt', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--negative_ratio', default=7, type=int)
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--device_id', default=0, type=str)
    parser.add_argument("--decay1", type=int, default=20)
    parser.add_argument("--decay2", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lr_type", type=str, default="dynamic_lr", choices=['dynamic_lr', 'cosine_annealing'])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_epoch", type=int, default=-1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--T_max", type=int, default=50)
    parser.add_argument("--eta_min", type=float, default=0.0)
    parser.add_argument("--loss_scale", type=int, default=1)
    parser.add_argument("--optim", type=str, default="sgd", choices=['sgd', 'momentum'])
    parser.add_argument("--num_workers", type=int, default=2)
    return vars(parser.parse_args(()))


if __name__ == '__main__':
    train(cfg=parse_args())
