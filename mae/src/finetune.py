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
# ==============================================================================
"""
Fine-tuning operations on other data sets.
"""

import time

from mindspore import nn
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from model.vit import FineTuneVit
from process_datasets.dataset import get_dataset
from utils.logger import get_logger
from utils.loss import get_loss
from utils.monitor import StateMonitor
from utils.lr_generator import LearningRate
from utils.trainer import create_train_one_step
from utils.eval_engine import get_eval_engine
from config.config import mae_finetune_config


def main(args):
    # Initialize the environment
    local_rank = 0
    device_num = 1
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    args.logger.info("model config: {}".format(args))
    context.set_context(device_target=args.device_target, device_id=args.device_id)

    # Get the training set
    train_dataset = get_dataset(args)
    data_size = train_dataset.get_dataset_size()
    new_epochs = args.epoch
    if args.per_step_size:
        new_epochs = int((data_size / args.per_step_size) * args.epoch)
    else:
        args.per_step_size = data_size

    # Get the validation set
    eval_dataset = get_dataset(args, split="train")

    # Instantiated models
    net = FineTuneVit(batch_size=args.batch_size, patch_size=args.patch_size,
                      image_size=args.image_size, dropout=args.dropout,
                      num_classes=args.num_classes, encoder_layers=args.encoder_layers,
                      encoder_num_heads=args.encoder_num_heads, encoder_dim=args.encoder_dim,
                      mlp_ratio=args.mlp_ratio, drop_path=args.drop_path,
                      channels=args.channels)
    eval_engine = get_eval_engine(args.eval_engine, net, eval_dataset, args)

    # Define the learning rate at the beginning
    if args.start_learning_rate is None:
        args.start_learning_rate = (args.base_lr * args.device_num * args.batch_size) / 256

    # Adjust the learning rate according to the number of epoch training
    lr_schedule = LearningRate(
        args.start_learning_rate, args.end_learning_rate,
        args.epoch, args.warmup_epochs, data_size
    )

    # Define parameter groups
    group_params = net.trainable_params()

    # If using checkpoints, load the model from the checkpoint
    if args.use_ckpt:
        # layer-wise lr decay
        params_dict = load_checkpoint(args.use_ckpt)
        load_param_into_net(net, params_dict)

    # Define the optimizer, using the weight decay Adam algorithm
    optimizer = nn.AdamWeightDecay(group_params,
                                   learning_rate=lr_schedule,
                                   weight_decay=args.weight_decay,
                                   beta1=args.beta1,
                                   beta2=args.beta2)

    # Define loss
    if not args.use_label_smooth:
        args.label_smooth_factor = 0.0
    vit_loss = get_loss(loss_name=args.loss_name, args=args)

    # Build the training network
    net_with_loss = nn.WithLossCell(net, vit_loss)
    net_with_train = create_train_one_step(args, net_with_loss, optimizer, log=args.logger)

    # Perform training status monitoring
    state_cb = StateMonitor(data_size=args.per_step_size,
                            tot_batch_size=args.batch_size * device_num,
                            eval_interval=args.eval_interval,
                            eval_offset=args.eval_offset,
                            eval_engine=eval_engine,
                            logger=args.logger.info)
    callback = [state_cb]
    save_ckpt_feq = args.save_ckpt_epochs * args.per_step_size
    if local_rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=save_ckpt_feq,
                                     keep_checkpoint_max=1,
                                     integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix=args.prefix,
                                     directory=args.save_dir,
                                     config=config_ck)
        callback += [ckpoint_cb]

    # Define the model and start training
    model = Model(net_with_train, metrics=eval_engine.metric,
                  eval_network=eval_engine.eval_network)

    eval_engine.set_model(model)

    t0 = time.time()
    eval_engine.compile()
    t1 = time.time()
    args.logger.info('compile time used={:.2f}s'.format(t1 - t0))

    # Model training
    model.train(new_epochs,
                train_dataset,
                callbacks=callback,
                sink_size=args.per_step_size,
                dataset_sink_mode=args.sink_mode)


if __name__ == "__main__":
    main(mae_finetune_config)
