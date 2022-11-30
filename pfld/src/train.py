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

import mindspore
import mindspore.dataset as ds
from mindspore import context, nn
from mindspore.dataset import vision
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindvision.engine.callback import LossMonitor

from model.pfld import pfld_1x_68, pfld_1x_98
from model.auxiliarynet import AuxiliaryNet
from pfld_utils.loss_cell import CustomWithLossCell
from pfld_utils.loss import PFLDLoss
from pfld_utils.utils import map_func
from datasets.data_preprocess import data_preprocess
from datasets.data_loader import DatasetsWFLW, Datasets300W


def main(args):
    # Set environment
    context.set_context(device_id=args.device_id, mode=context.GRAPH_MODE, device_target=args.device_target)

    # Data preprocess
    if args.data_preprocess_flag == 'True':
        data_preprocess(target_dataset=args.target_dataset, dataset_file_path=args.dataset_file_path)

    # Create dataset, 300W dataset is for 68 points, WFLW is for 98 points
    transform = vision.py_transforms.ToTensor()
    assert args.model_type in ['98_points', '68_points']

    # Generate dataset
    if args.model_type == '68_points':
        dataset_generator = Datasets300W(args.dataset_file_path + '/train_data/list.txt', transform)
        net = pfld_1x_68()

    else:
        dataset_generator = DatasetsWFLW(args.dataset_file_path + '/train_data/list.txt', transform)
        net = pfld_1x_98()
    dataset = ds.GeneratorDataset(list(dataset_generator),
                                  ["img", "landmark", "attributes", "angle"],
                                  num_parallel_workers=args.workers,
                                  shuffle=True)
    dataset = dataset.batch(args.train_batchsize,
                            input_columns=["attributes"],
                            output_columns=["weight_attribute"],
                            per_batch_map=map_func)

    if args.resume == 'True':
        LoadPretrainedModel(net, args.pretrain_model_path[0][args.model_type]).run()

    # Get other components of the model
    net_auxiliary = AuxiliaryNet()

    dataset_size = len(dataset_generator.lines)
    lr = nn.inverse_decay_lr(
        learning_rate=args.base_lr,
        decay_rate=0.4,
        total_step=((dataset_size + args.train_batchsize - 1) // args.train_batchsize) * args.end_epoch,
        step_per_epoch=(dataset_size + args.train_batchsize - 1) // args.train_batchsize,
        decay_epoch=1)

    optimizer = nn.Adam(params=net.get_parameters(),
                        learning_rate=lr,
                        weight_decay=args.weight_decay)

    loss = PFLDLoss()
    net_with_loss = CustomWithLossCell(net, net_auxiliary, loss)

    model = mindspore.Model(network=net_with_loss, optimizer=optimizer)

    config_ck = CheckpointConfig(
        save_checkpoint_steps=(dataset_size + args.train_batchsize - 1) // args.train_batchsize,
        keep_checkpoint_max=100)
    ckpoint = ModelCheckpoint(prefix="checkpoint",
                              directory=args.checkpoint_path,
                              config=config_ck)

    # Start to train
    model.train(args.end_epoch, dataset, callbacks=[ckpoint, LossMonitor(lr)], dataset_sink_mode=False)


petrain_model = {'98_points': 'https://download.mindspore.cn/vision/pfld/PFLD1X_WFLW.ckpt',
                 '68_points': 'https://download.mindspore.cn/vision/pfld/PFLD1X_300W.ckpt'},


def parse_args():
    """
    Set network parameters.

    Returns:
        ArgumentParser. Parameter information.
    """
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-j', '--workers', default=1, type=int)
    parser.add_argument('--device_target', default='CPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--base_lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', '--wd', default=1e-6, type=float)
    parser.add_argument('--end_epoch', default=100, type=int)
    parser.add_argument('--resume', default='True', type=str)
    parser.add_argument('--model_type', default='68_points', type=str)
    parser.add_argument('--target_dataset', default='300W', type=str)
    parser.add_argument('--data_preprocess_flag', default='True', type=str)
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--checkpoint_path', default='../checkpoint', type=str)
    parser.add_argument('--dataset_file_path', default='../datasets/300W', type=str)
    parser.add_argument('--pretrain_model_path', default=petrain_model, type=dict)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
