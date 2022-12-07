import os
import numpy as np
import mindspore
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import math
from src.callback import LossCallBack
from src.cnn_ctc import CNNCTC_Model, ctc_loss, WithLossCell, CNNCTCTrainOneStepWithLossScaleCell
from src.dataset import ST_MJ_Generator_batch_fixed_length
from src.config import config



set_seed(1)



mindspore.set_context(mode=context.GRAPH_MODE, save_graphs=False, save_graphs_path=".")

def dataset_creator(run_distribute):
    
    st_dataset = ST_MJ_Generator_batch_fixed_length()
    ds = GeneratorDataset(st_dataset,
                          ['img', 'label_indices', 'text', 'sequence_length'],
                          num_parallel_workers=4)

    return ds




def dynamic_lr(config, steps_per_epoch):
    """dynamic learning rate generator"""
    base_lr = config.base_lr
    total_steps = steps_per_epoch * config.TRAIN_EPOCHS
    warmup_steps = int(config.warmup_step)
    decay_steps = total_steps - warmup_steps
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(base_lr) - float(base_lr * config.warmup_ratio)) / float(warmup_steps)
            learning_rate = float(base_lr * config.warmup_ratio) + lr_inc * i
            lr.append(learning_rate)
        else:
            base = float(i - warmup_steps) / float(decay_steps)
            learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr    
            lr.append(learning_rate )

    return lr

def train():
    target = config.device_target
    mindspore.set_context(device_target=target)
    if target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
        mindspore.set_context(device_id=device_id)
        if config.run_distribute:
            init()
            context.set_auto_parallel_context(parallel_mode="data_parallel")

        ckpt_save_dir = config.SAVE_PATH
    else:
        # GPU target
        device_id = int(os.getenv('DEVICE_ID', '0'))
        mindspore.set_context(device_id=device_id)
        if config.run_distribute:
            init()
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode="data_parallel",
                                              gradients_mean=False,
                                              gradient_fp32_sync=False)

            ckpt_save_dir = config.SAVE_PATH + "ckpt_" + str(get_rank()) + "/"
            print(ckpt_save_dir)
        else:
            ckpt_save_dir = config.SAVE_PATH + "ckpt_standalone/"

    ds = dataset_creator(config.run_distribute)

    net = CNNCTC_Model(config.NUM_CLASS, config.HIDDEN_SIZE, config.FINAL_FEATURE_WIDTH)
    net.set_train(True)

    if config.PRED_TRAINED:
        param_dict = load_checkpoint(config.PRED_TRAINED)
        load_param_into_net(net, param_dict)
        print('parameters loaded!')
    else:
        print('train from scratch...')

    criterion = ctc_loss()
    dataset_size = ds.get_dataset_size()
    lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)
    opt = mindspore.nn.RMSProp(params=net.trainable_params(),
                               centered=True,
                               learning_rate=lr,
                               momentum=config.MOMENTUM,
                               loss_scale=config.LOSS_SCALE)

    net = WithLossCell(net, criterion)

    if target == "Ascend":
        loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(
            config.LOSS_SCALE, False)
        net.set_train(True)
        model = Model(net, optimizer=opt, loss_scale_manager=loss_scale_manager, amp_level="O2")
    else:
        scaling_sens = Tensor(np.full((1), config.LOSS_SCALE), dtype=mstype.float32)
        net = CNNCTCTrainOneStepWithLossScaleCell(net, opt, scaling_sens)
        net.set_train(True)
        model = Model(net)

    loss_cb = LossCallBack()
    time_cb = TimeMonitor(data_size=dataset_size)
    config_ck = CheckpointConfig(save_checkpoint_steps=config.SAVE_CKPT_PER_N_STEP,
                                 keep_checkpoint_max=config.KEEP_CKPT_MAX_NUM)
    ckpoint_cb = ModelCheckpoint(prefix="CNNCTC", config=config_ck, directory=ckpt_save_dir)
    callbacks = [loss_cb, time_cb, ckpoint_cb]

    if config.run_distribute:
        if device_id == 0:
            model.train(config.TRAIN_EPOCHS,
                        ds,
                        callbacks=callbacks,
                        dataset_sink_mode=False)
        else:
            callbacks.remove(ckpoint_cb)
            model.train(config.TRAIN_EPOCHS, ds, callbacks=callbacks, dataset_sink_mode=False)
    else:
        model.train(config.TRAIN_EPOCHS,
                    ds,
                    callbacks=callbacks,
                    dataset_sink_mode=False)


if __name__ == '__main__':
    train()
