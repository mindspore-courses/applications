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
Pre-train operations on other data sets.
"""

from mindspore import nn
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from process_datasets.imagenet import create_dataset
from model.mae_vit import PreTrainMAEVit
from utils.monitor import LossMonitor
from utils.logger import get_logger
from utils.lr_generator import LearningRate
from utils.trainer import create_train_one_step
from config.config import mae_train_config


def main(args):
    # Initialize the environment
    local_rank = 0
    device_num = 1
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    args.logger.info("model config: {}".format(args))
    context.set_context(device_target=args.device_target, device_id=args.device_id)

    # Get the dataset
    dataset = create_dataset(args)
    data_size = dataset.get_dataset_size()
    new_epochs = args.epoch
    if args.per_step_size:
        new_epochs = int((data_size / args.per_step_size) * args.epoch)
    else:
        args.per_step_size = data_size
    args.logger.info("Will be Training epochs:{}， sink_size:{}".format(new_epochs, args.per_step_size))
    args.logger.info("Create training dataset finish, data size:{}".format(data_size))

    # Instantiated models
    net = PreTrainMAEVit(batch_size=args.batch_size, patch_size=args.patch_size, image_size=args.image_size,
                         encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                         encoder_num_heads=args.encoder_num_heads, decoder_num_heads=args.decoder_num_heads,
                         encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim,
                         mlp_ratio=args.mlp_ratio, masking_ratio=args.masking_ratio)

    # Define the learning rate at the beginning
    if args.start_learning_rate == 0.:
        args.start_learning_rate = (args.base_lr * args.device_num * args.batch_size) / 256

    # Adjust the learning rate according to the number of epoch training
    lr_schedule = LearningRate(
        args.start_learning_rate, args.end_learning_rate,
        args.epoch, args.warmup_epochs, data_size
    )

    # Define the optimizer, using the weight decay Adam algorithm
    optimizer = nn.AdamWeightDecay(net.trainable_params(),
                                   learning_rate=lr_schedule,
                                   weight_decay=args.weight_decay,
                                   beta1=args.beta1,
                                   beta2=args.beta2)

    # Load pre-trained checkpoints
    if args.use_ckpt:
        params_dict = load_checkpoint(args.use_ckpt)
        load_param_into_net(net, params_dict)
        load_param_into_net(optimizer, params_dict)

    # Build the training network
    train_model = create_train_one_step(args, net, optimizer, log=args.logger)

    # Perform training status monitoring
    callback = [LossMonitor(log=args.logger)]

    # Model configuration
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
    model = Model(train_model)
    model.train(new_epochs, dataset, callbacks=callback,
                dataset_sink_mode=args.sink_mode, sink_size=args.per_step_size)


if __name__ == "__main__":
    main(mae_train_config)
