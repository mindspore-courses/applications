# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import mindspore
from mindspore import ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, Model, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from src.dataset import create_dataset
from src.unet3d_model import UNet3d, UNet3d_
from src.lr_schedule import dynamic_lr
from src.loss import SoftmaxCrossEntropyWithLogits
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
# from tensorboardX import SummaryWriter


if config.device_target == 'Ascend':
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, \
                        device_id=device_id)
else:
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
mindspore.set_seed(1)

@moxing_wrapper()
def train_net(data_path,
              run_distribute):
    data_dir = data_path + "/image/"
    seg_dir = data_path + "/seg/"
    if run_distribute:
        init()
        if config.device_target == 'Ascend':
            rank_id = get_device_id()
            rank_size = get_device_num()
        else:
            rank_id = get_rank()
            rank_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=rank_size,
                                          gradients_mean=True)
    else:
        rank_id = 0
        rank_size = 1
    # (1) create dataset
    train_dataset = create_dataset(data_path=data_dir, seg_path=seg_dir, \
                                    rank_size=rank_size, rank_id=rank_id, is_training=True)
    train_data_size = train_dataset.get_dataset_size()
    print("train dataset length is:", train_data_size)
    # (2) create network
    if config.device_target == 'Ascend':
        network = UNet3d()
    else:
        network = UNet3d_()
    # (3)define loss funtion
    loss_ce_fn = nn.CrossEntropyLoss()
    loss_dice_fn = nn.DiceLoss(smooth=1e-5)
    # (4) lr shedule and optimizor
    lr = Tensor(dynamic_lr(config, train_data_size), mstype.float32)
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=lr)
    # (5) set training mode
    network.set_train()
    # (9) Start training
    print("============== Starting Training ==============")
    # Define forward function
    def forward_fn(data, label):
        logits = network(data)
        loss_ce = loss_ce_fn(logits, label)
        loss_dice = loss_dice_fn(logits, label)
        loss = loss_dice + loss_ce
        return loss, loss_dice, loss_ce, logits
    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # Define function of one-step training
    def train_step(data, label):
        (loss, loss_dice, loss_ce, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss, loss_dice, loss_ce
    from mindspore import SummaryRecord
    summary_collect_frequency = 5
    max_epoch = 10
    with SummaryRecord('./summary_dir/summary_01') as summary_record:
        for epoch in range(max_epoch):
            for step, (data, label) in enumerate(train_dataset.create_tuple_iterator()):
                loss, loss_dice, loss_ce = train_step(data, label)

                current_step = epoch * train_data_size + step
                current_lr = optimizer.get_lr()
                # if current_step % summary_collect_frequency == 0:
                summary_record.record(current_step)
                summary_record.add_value('scalar', 'lr', current_lr)
                summary_record.add_value('scalar', 'loss_total', loss)
                summary_record.add_value('scalar', 'loss_dice', loss_dice)
                summary_record.add_value('scalar', 'loss_ce', loss_ce)

                loss, loss_dice, loss_ce = loss.asnumpy(), loss_dice.asnumpy(), loss_ce.asnumpy()
                print("Epoch: %d [%d/%d] [%d/%d] lr:%.7f Loss: %.4f Loss_dice: %.4f Loss_ce: %.4f" %
                      (epoch, step, train_data_size, current_step, train_data_size*max_epoch,
                       current_lr, loss, loss_dice, loss_ce))

            # Save checkpoint
            ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
            mindspore.save_checkpoint(network, os.path.join(ckpt_save_dir, "model.ckpt"))
            print("Saved Model to {}/model.ckpt".format(ckpt_save_dir))
    # model.train(config.epoch_size, train_dataset, callbacks=callbacks_list, dataset_sink_mode=False)
    print("============== End Training ==============")

if __name__ == '__main__':
    train_net(data_path=config.data_path,
              run_distribute=config.run_distribute)
