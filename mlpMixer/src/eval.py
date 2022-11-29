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
""" MLP-Mixer training script. """

import argparse
import os

from mindspore import nn
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.train import Model
import mindspore.dataset.vision.c_transforms as transforms
from mindvision import dataset
from mindvision.dataset import Cifar10
from mindvision.engine.loss import CrossEntropySmooth

from model.mlp_mixer import mixer_b_16, mixer_b_32, mixer_s_16, mixer_s_8, mixer_s_32

set_seed(777)

def mixer_eval(args_opt):
    """MLP-Mixer eval."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    normalize = transforms.Normalize(mean=(126.1, 123.7, 114.6),
                                     std=(62.9, 61.9, 66.7))

    transform = [
        transforms.Resize(args_opt.resize),
        normalize,
        transforms.HWC2CHW()
    ]

    dataset_eval = Cifar10(path=args_opt.data_url,
                           shuffle=True,
                           batch_size=args_opt.batch_size,
                           split='test',
                           transform=transform).run()

    if args_opt.finetune:
        param_path = "./pretrain_mixer"
        if not os.path.exists(param_path):
            os.makedirs(param_path)

        # Create Mixer model.
        if args_opt.model == "Mixer_B_16":
            network = mixer_b_16(num_classes=args_opt.num_classes, image_size=args_opt.resize)
            param_url = "https://download.mindspore.cn/vision/mlpmixer/gsam_Mixer-B_16.ckpt"
        elif args_opt.model == "Mixer_B_32":
            network = mixer_b_32(num_classes=args_opt.num_classes, image_size=args_opt.resize)
            param_url = "https://download.mindspore.cn/vision/mlpmixer/gsam_Mixer-B_32.ckpt"
        elif args_opt.model == "Mixer_S_16":
            network = mixer_s_16(num_classes=args_opt.num_classes, image_size=args_opt.resize)
            param_url = "https://download.mindspore.cn/vision/mlpmixer/gsam_Mixer-S_16.ckpt"
        elif args_opt.model == "Mixer_S_32":
            network = mixer_s_32(num_classes=args_opt.num_classes, image_size=args_opt.resize)
            param_url = "https://download.mindspore.cn/vision/mlpmixer/gsam_Mixer-S_32.ckpt"
        elif args_opt.model == "Mixer_S_8":
            network = mixer_s_8(num_classes=args_opt.num_classes, image_size=args_opt.resize)
            param_url = "https://download.mindspore.cn/vision/mlpmixer/gsam_Mixer-S_8.ckpt"

        # Download Mixers' ckpt.
        param = dataset.DownLoad()
        param.download_url(url=param_url, path=param_path)
        param_dict = load_checkpoint("./pretrain_mixer" + "/gsam_Mixer-B_16.ckpt")
        load_param_into_net(network, param_dict)

    else:
        # Create Mixer model.
        if args_opt.model == "Mixer_b_16":
            network = mixer_b_16(num_classes=args_opt.num_classes, image_size=args_opt.resize)
        elif args_opt.model == "Mixer_B_32":
            network = mixer_b_32(num_classes=args_opt.num_classes, image_size=args_opt.resize)
        elif args_opt.model == "Mixer_S_16":
            network = mixer_s_16(num_classes=args_opt.num_classes, image_size=args_opt.resize)
        elif args_opt.model == "Mixer_S_32":
            network = mixer_s_32(num_classes=args_opt.num_classes, image_size=args_opt.resize)
        elif args_opt.model == "Mixer_S_8":
            network = mixer_s_8(num_classes=args_opt.num_classes, image_size=args_opt.resize)

        # import local Mixers' ckpt.
        net_dir = args_opt.ckpt_save_dir
        ckpt_file_name = net_dir + args_opt.model + '.ckpt'
        param_dict = load_checkpoint(ckpt_file_name)
        load_param_into_net(network, param_dict)

     # Define loss function.
    network_loss = CrossEntropySmooth(sparse=True,
                                      reduction="mean",
                                      smooth_factor=args_opt.smooth_factor,
                                      classes_num=args_opt.num_classes)

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    # Init the model.
    model = Model(network, network_loss, metrics=eval_metrics)

    # Begin to eval.
    result = model.eval(dataset_eval)
    print(result)

def parse_args():
    """Set parse."""
    parser = argparse.ArgumentParser(description="ViT eval.")
    parser.add_argument("--ckpt_save_dir", type=str, default="./mixer", help="Location of training outputs.")
    parser.add_argument("--finetune", type=bool, default=False, help="Load pretrained model.")
    parser.add_argument("--model", required=False, default="Mixer_B_16",
                        choices=["Mixer_B_16",
                                 "Mixer_B_32",
                                 "Mixer_S_16",
                                 "Mixer_S_32",
                                 "Mixer_S_8"])
    parser.add_argument("--device_target", type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument("--data_url", default='./data/', help="Location of data.")
    parser.add_argument("--num_parallel_workers", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of batch size.")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classification.")
    parser.add_argument("--smooth_factor", type=float, default=0.1, help="The smooth factor.")
    parser.add_argument("--resize", type=int, default=224, help="Resize the image.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    mixer_eval(args)
