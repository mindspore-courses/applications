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
"""Main script for neuralrecon training with MindSpore"""

import argparse
import os
import time
import datetime

from mindspore import Tensor
from mindspore import context
from mindspore.nn.optim.adam import Adam
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from loguru import logger

from train_utils.var_init import default_recurisive_init
from train_utils.utils import get_param_groups, TrainingWrapper, InternalCallbackParam
from train_utils.lr_scheduler import MultiStepLR
from datasets import transforms_ms, scannet_ms
from models import NeuralRecon
from config import cfg, update_config


def get_args():
    """Get args"""
    parser = argparse.ArgumentParser(description='A MindSpore Implementation of NeuralRecon')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # parse arguments and check
    args_tmp = parser.parse_args()

    return args_tmp


args = get_args()
update_config(cfg, args)

cfg.defrost()
num_gpus = get_group_size()
print('number of gpus: {}'.format(num_gpus))
cfg.DISTRIBUTED = num_gpus > 1
cfg.LOCAL_RANK = 0
if cfg.DISTRIBUTED:
    init()
    cfg.LOCAL_RANK = get_rank()
    cfg.GROUP_SIZE = get_group_size()
cfg.freeze()

set_seed(cfg.SEED)
context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=int(os.environ["DEVICE_ID"]))

context.reset_auto_parallel_context()
if cfg.DISTRIBUTED:
    parallel_mode = ParallelMode.DATA_PARALLEL
    degree = get_group_size()
else:
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1

context.set_auto_parallel_context(
    parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)
network = NeuralRecon(cfg)
default_recurisive_init(network)

# create logger
if cfg.LOCAL_RANK == 0: # main process
    if not os.path.isdir(cfg.LOGDIR):
        os.makedirs(cfg.LOGDIR)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_path = os.path.join(cfg.LOGDIR, f'{current_time_str}_{cfg.MODE}.log')
    print('creating log file', logfile_path)
    logger.add(logfile_path, format="{time} {level} {message}", level="INFO")

# Augmentation
if cfg.MODE == 'train':
    n_views = cfg.TRAIN.N_VIEWS
    random_rotation = cfg.TRAIN.RANDOM_ROTATION_3D
    random_translation = cfg.TRAIN.RANDOM_TRANSLATION_3D
    padding_xy = cfg.TRAIN.PAD_XY_3D
    padding_z = cfg.TRAIN.PAD_Z_3D
else:
    n_views = cfg.TEST.N_VIEWS
    random_rotation = False
    random_translation = False
    padding_xy = 0
    padding_z = 0

transform = []
transform += [transforms_ms.ResizeImage((640, 480)),
              transforms_ms.ToTensor(),
              transforms_ms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation, random_translation,
                  padding_xy, padding_z, max_epoch=cfg.TRAIN.EPOCHS),
              transforms_ms.IntrinsicsPoseToProjection(n_views, 4),
              ]

transforms = transforms_ms.Compose(transform)

# dataloader
data_loader, steps_per_epoch = scannet_ms.GetDataLoader(datapath=cfg.TRAIN.PATH,
                                                        transforms=transforms,
                                                        nviews=n_views,
                                                        n_scales=len(cfg.MODEL.THRESHOLDS) - 1,
                                                        per_batch_size=cfg.BATCH_SIZE,
                                                        max_epoch=cfg.TRAIN.EPOCHS,
                                                        rank=cfg.LOCAL_RANK,
                                                        group_size=cfg.GROUP_SIZE,
                                                        mode='train')

# load parameters
start_epoch = 0
if cfg.RESUME:
    saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if saved_models:
        # use the latest checkpoint file
        loadckpt = os.path.join(cfg.LOGDIR, saved_models[-1])
        logger.info("resuming " + str(loadckpt))
        state_dict = load_checkpoint(loadckpt)
        load_param_into_net(network, state_dict['model'])
        start_epoch = state_dict['epoch'] + 1
elif cfg.LOADCKPT != '':
    # load checkpoint file specified by args.loadckpt
    logger.info("loading model {}".format(cfg.LOADCKPT))
    state_dict = load_checkpoint(cfg.LOADCKPT)
    load_param_into_net(network, state_dict['model'])
    start_epoch = state_dict['epoch'] + 1
logger.info("start at epoch {}".format(start_epoch))

# scheduler
milestones = [int(epoch_idx) for epoch_idx in cfg.TRAIN.LREPOCHS.split(':')[0].split(',')]
lr_gamma = 1 / float(cfg.TRAIN.LREPOCHS.split(':')[1])
lr_fun = MultiStepLR(cfg.TRAIN.LR, milestones, lr_gamma, steps_per_epoch, cfg.TRAIN.EPOCHS)
lr = lr_fun.get_lr()

# optimizer
optimizer = Adam(params=get_param_groups(network), learning_rate=Tensor(lr),
                 beta1=0.9, beta2=0.999, weight_decay=cfg.TRAIN.WD)

network = TrainingWrapper(network, optimizer)
network.set_train()

if cfg.LOCAL_RANK == 0:
    # checkpoint save
    ckpt_max_num = cfg.TRAIN.EPOCHS * steps_per_epoch // cfg.SAVE_FREQ
    ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.SAVE_FREQ,
                                   keep_checkpoint_max=ckpt_max_num)
    ckpt_cb = ModelCheckpoint(config=ckpt_config,
                              directory=cfg.LOGDIR,
                              prefix='{}'.format(cfg.LOCAL_RANK))
    cb_params = InternalCallbackParam()
    cb_params.train_network = network
    cb_params.epoch_num = ckpt_max_num
    cb_params.cur_epoch_num = 1
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)

    print('config.steps_per_epoch = {} config.ckpt_interval ={}'.format(steps_per_epoch,
                                                                        cfg.SAVE_FREQ))

for i_batch, sample in enumerate(data_loader):
    i = i_batch % steps_per_epoch
    epoch_idx = i_batch // steps_per_epoch + 1
    # training
    start_time = time.time()
    loss, _, _ = network(sample)
    if cfg.LOCAL_RANK == 0:
        logger.info(
            'Epoch {}/{}, Iter {}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, cfg.TRAIN.EPOCHS,
                                                                              i_batch, loss,
                                                                              time.time() - start_time))
    # checkpoint
    if (epoch_idx + 1) % cfg.SAVE_FREQ == 0 and cfg.LOCAL_RANK == 0:
        # ckpt progress
        cb_params.cur_epoch_num = epoch_idx
        cb_params.cur_step_num = i + 1 + (epoch_idx-1)*steps_per_epoch
        cb_params.batch_num = i + 2 + (epoch_idx-1)*steps_per_epoch
        ckpt_cb.step_end(run_context)
