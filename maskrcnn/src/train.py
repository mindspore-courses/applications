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

"""train MaskRcnn and get checkpoint files."""
import os
import time

import mindspore.common.dtype as mstype
from mindspore import context, Tensor, Parameter
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Momentum
from mindspore.common import set_seed

# when use maskrcnn mobilenetv1, just change the following backbone and defined network
# from mask_rcnn_mobilenetv1 and network_define_maskrcnnmobilenetv1
from model.mask_rcnn_r50 import MaskRcnnResnet50
from utils.config import config
from utils.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from utils.lr_schedule import dynamic_lr
from dataset.dataset import create_coco_dataset, data_to_mindrecord_byte_image


set_seed(1)

def create_mindrecord_dir(prefix, mindrecord_dir):
    """Create MindRecord Direction."""
    if not os.path.isdir(mindrecord_dir):
        os.makedirs(mindrecord_dir)
    if config.dataset == "coco":
        if os.path.isdir(config.data_root):
            print("Create Mindrecord.")
            data_to_mindrecord_byte_image("coco", True, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            raise Exception("coco_root not exits.")
    else:
        if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
            print("Create Mindrecord.")
            data_to_mindrecord_byte_image("other", True, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            raise Exception("IMAGE_DIR or ANNO_PATH not exits.")
    while not os.path.exists(mindrecord_file+".db"):
        time.sleep(5)

def load_pretrained_ckpt(net, load_path, device_target):
    """
    Load pretrained checkpoint.

    Args:
        net(Cell): Used Network
        load_path(string): The path of checkpoint.
        device_target(string): device target.

    Returns:
        Cell, the network with pretrained weights.
    """
    param_dict = load_checkpoint(load_path)
    if config.pretrain_epoch_size == 0:
        for item in list(param_dict.keys()):
            if not (item.startswith('backbone') or item.startswith('rcnn_mask')):
                param_dict.pop(item)

        if device_target == 'GPU':
            for key, value in param_dict.items():
                tensor = Tensor(value, mstype.float32)
                param_dict[key] = Parameter(tensor, key)

    load_param_into_net(net, param_dict)
    return net

def train_maskrcnn():
    """construct the traning function"""
    # Allocating memory Environment
    device_target = config.device_target
    rank = 0
    device_num = 1
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

    print("Start train for maskrcnn!")

    dataset_sink_mode_flag = False
    if not config.do_eval and config.run_distribute:
        init()
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")
    # Call the interface for data processing
    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is MaskRcnn.mindrecord0, 1, ... file_num.
    prefix = "MaskRcnn.mindrecord"
    mindrecord_dir = os.path.join(config.data_root, config.mindrecord_dir)
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if rank == 0 and not os.path.exists(mindrecord_file):
        create_mindrecord_dir(prefix, mindrecord_dir)
    # When create MindDataset, using the fitst mindrecord file,
    # such as MaskRcnn.mindrecord0.

    dataset = create_coco_dataset(mindrecord_file, batch_size=config.batch_size, device_num=device_num, rank_id=rank)
    dataset_size = dataset.get_dataset_size()
    print("total images num: ", dataset_size)
    print("Create dataset done!")

    # Net Instance
    net = MaskRcnnResnet50(config=config)
    net = net.set_train()

    # load pretrained model
    load_path = config.pre_trained
    if load_path != "":
        print("Loading pretrained resnet50 checkpoint")
        net = load_pretrained_ckpt(net=net, load_path=load_path, device_target=device_target)

    loss = LossNet()
    lr = Tensor(dynamic_lr(config, rank_size=device_num, start_steps=config.pretrain_epoch_size * dataset_size),
                mstype.float32)
    opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                   weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    # wrap the loss function
    net_with_loss = WithLossCell(net, loss)
    # Use TrainOneStepCell set the training pipeline.
    net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)
    # Monitor the traning process.
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    # save the trained model
    if config.save_checkpoint:
        # set saved weights.
        ckpt_step = config.save_checkpoint_epochs * dataset_size
        ckptconfig = CheckpointConfig(save_checkpoint_steps=ckpt_step, keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
        # apply saved weights.
        ckpoint_cb = ModelCheckpoint(prefix='mask_rcnn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]
    # start training.
    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=dataset_sink_mode_flag)

if __name__ == '__main__':
    train_maskrcnn()
