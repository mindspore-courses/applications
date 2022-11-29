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
# ===========================================================================
"""Parse parameters function."""
import ast
import argparse


def parse_args():
    """
    Parse parameters.

     Returns:
        parsed parameters.
    """

    parser = argparse.ArgumentParser(description='config')
    parser.add_argument("--seed", default=0, type=int, help="seed for initializing training. ")
    parser.add_argument("--device_target", default="Ascend", choices=["GPU", "Ascend"], type=str)
    parser.add_argument("--device_id", default=0, type=int, help="device id")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="convnext_tiny", help="model architecture")
    parser.add_argument("--in_chans", default=3, type=int)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--drop_path_rate", default=0.1, type=float)
    parser.add_argument("--amp_level", default="O1", choices=["O0", "O1", "O2", "O3"], help="AMP Level")
    parser.add_argument("--label_smoothing", type=float, help="label smoothing to use, default 0.1", default=0.1)
    parser.add_argument("--mix_up", default=0.8, type=float, help="mix up")
    parser.add_argument("--cutmix", default=1.0, type=float, help="cutmix")
    parser.add_argument("--run_modelarts", type=ast.literal_eval, default=False, help="whether run on modelarts")
    parser.add_argument("--pretrained", dest="pretrained", default=None, type=str, help="use pre-trained model")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument('--data_url', default="./data", help='location of data.')
    parser.add_argument("--image_size", default=224, help="image Size.", type=int)
    parser.add_argument('--interpolation', type=str, default="bicubic")
    parser.add_argument('--auto_augment', type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("-j", "--num_parallel_workers", default=16, type=int, metavar="N",
                        help="number of data loading workers (default: 16)")
    parser.add_argument("--batch_size", default=300, type=int, metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                             "batch size of all Devices on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--re_prob", default=0.0, type=float, help="re prob")
    parser.add_argument('--re_mode', type=str, default="pixel")
    parser.add_argument("--re_count", default=1, type=int, help="re count")
    parser.add_argument("--mixup_prob", default=1., type=float, help="mixup prob")
    parser.add_argument("--switch_prob", default=0.5, type=float, help="switch prob")
    parser.add_argument("--mixup_mode", default='batch', type=str, help="mixup_mode")
    parser.add_argument("--optimizer", help="Which optimizer to use", default="adamw")
    parser.add_argument("--lr_scheduler", default="cosine_lr", help="schedule for the learning rate.")
    parser.add_argument("--warmup_length", default=20, type=int, help="number of warmup iterations")
    parser.add_argument("--warmup_lr", default=0.00000007, type=float, help="warm up learning rate")
    parser.add_argument("--base_lr", default=0.004, type=float, help="base learning rate")
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--min_lr", default=0.0000006, type=float, help="min learning rate")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--accumulation_step", default=1, type=int, help="accumulation step")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--beta", default=[0.9, 0.999], type=lambda x: [float(a) for a in x.split(",")],
                        help="beta for optimizer")
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--wd", "--weight_decay", default=0.05, type=float, metavar="W",
                        help="weight decay (default: 0.05)", dest="weight_decay")
    parser.add_argument("--is_dynamic_loss_scale", default=True, type=bool, help="is_dynamic_loss_scale ")
    parser.add_argument("--loss_scale", default=1024, type=int, help="loss_scale")
    parser.add_argument("--with_ema", default=False, type=ast.literal_eval, help="training with ema")
    parser.add_argument("--save_every", default=20, type=int, help="save every ___ epochs(default:20)")
    parser.add_argument('--train_url', default="./", help='location of training outputs.')

    return parser.parse_args()
