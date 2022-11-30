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
"""Buile and train model."""

import argparse

from mindspore import dataset as ds
from mindspore import context, nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train import Model
from mindvision.engine.callback import LossMonitor

from src.dataset.data import Mydata
from src.model.ecg_cnn import ECGCNNNet


def main(args):
    """Main."""
    # Set environment
    mode = context.GRAPH_MODE if args.graph_mode else context.PYNATIVE_MODE
    context.set_context(mode=mode, device_target=args.device_target)
    # Generator dataset
    train_dataset_generator = Mydata(data_path=args.data_path, label_path=args.label_path, splits="train")
    train_dataset = ds.GeneratorDataset(train_dataset_generator, ["data", "label"], shuffle=True)
    dataset = train_dataset.batch(batch_size=args.batch_size, drop_remainder=True)
    # Get other components of the model
    ecg_net = ECGCNNNet()
    optimizer = nn.Momentum(params=ecg_net.trainable_params(),
                            momentum=args.momentum,
                            learning_rate=args.lr,
                            weight_decay=args.weight_decay)
    loss = nn.loss.SoftmaxCrossEntropyWithLogits()
    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_steps,
                                 keep_checkpoint_max=args.keep_checkpoint_max)
    # 应用模型保存参数
    ckpoint = ModelCheckpoint(prefix="ecg_net", directory=args.checkpoint_dir, config=config_ck)
    # 初始化模型参数
    model = Model(ecg_net, loss_fn=loss, optimizer=optimizer, metrics={'accuracy'})
    # 训练网络模型
    model.train(10, dataset, callbacks=[ckpoint, LossMonitor(args.lr, args.per_print_times)])


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--graph_mode', default=True, type=bool)
    parser.add_argument('--device_target', default='GPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--data_path', default="/home/hujingsong/my_ecg/src/dataset/30X_eu_MLIII.csv", type=str)
    parser.add_argument('--label_path', default="/home/hujingsong/my_ecg/src/dataset/30Y_eu_MLIII.csv", type=str)
    parser.add_argument('--batch_size', default=128, type=str)
    parser.add_argument('--momentum', default=0.7, type=str)
    parser.add_argument('--lr', default=0.003, type=str)
    parser.add_argument('--weight_decay', '--wd', default=1e-6, type=float)
    parser.add_argument('--save_checkpoint_steps', default=1000, type=str)
    parser.add_argument('--keep_checkpoint_max', default=10, type=str)
    parser.add_argument('--checkpoint_dir', default="/home/hujingsong/my_ecg/save/", type=str)
    parser.add_argument('--per_print_times', default=100, type=str)
    return parser.parse_args(args=[])


if __name__ == "__main__":
    main(parse_args())
