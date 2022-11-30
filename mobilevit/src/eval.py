# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.lr_schedule.py
# org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" mobiletvit eval script. """

import math
import argparse

import mindspore.dataset.vision.c_transforms as c_transforms
import mindspore.dataset.vision.py_transforms as p_transforms
from mindspore import context
from mindspore.train import Model
from mindspore.dataset.vision import Inter
from mindspore import nn, load_checkpoint, load_param_into_net

from datasets.imagenet import ImageNet
from models.mobilevit import MobileViT
from utils.CrossEntropyEsmooth import CrossEntropySmooth


def mobilevit_eval(args_opt):
    """mobilevit eval."""

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data pipeline.
    img_transforms = ([
        c_transforms.Decode(),
        c_transforms.Resize((int(math.ceil(256 / 0.875)) // 32) * 32, interpolation=Inter.BILINEAR),
        c_transforms.CenterCrop(256),
        c_transforms.ConvertColor(c_transforms.ConvertMode.COLOR_RGB2BGR),
        c_transforms.RandomHorizontalFlip(),
        p_transforms.ToTensor(),
    ])

    dataset = ImageNet(args_opt.data_url,
                       split="val",
                       transform=img_transforms,
                       num_parallel_workers=args_opt.num_parallel_workers,
                       resize=args_opt.resize,
                       shuffle=True,
                       batch_size=args_opt.batch_size)

    dataset_eval = dataset.run()

    # Create model.
    network = MobileViT(model_type=args_opt.model_type, num_classes=args_opt.num_classes)

    # load pertain model
    param_dict = load_checkpoint(args_opt.pretrained_model)
    load_param_into_net(network, param_dict)

    # Define loss function.
    network_loss = CrossEntropySmooth(sparse=True,
                                      reduction="mean",
                                      smooth_factor=0.1,
                                      classes_num=args_opt.num_classes)

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    # Init the model.
    model = Model(network, network_loss, metrics=eval_metrics)
    result = model.eval(dataset_eval)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT eval.')
    parser.add_argument("--data_url", default=None, help="Location of data.")
    parser.add_argument('--pretrained_model', default=None, type=str, metavar='PATH')
    parser.add_argument('--model_type', default='xx_small', type=str, metavar='model_type')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of batch size.')
    parser.add_argument('--smooth_factor', type=float, default=0.1, help='The smooth factor.')
    parser.add_argument('--num_classes', type=int, default=1001, help='Number of classification.')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--resize', type=int, default=256, help='Resize the height and weight of picture.')
    args = parser.parse_known_args()[0]
    mobilevit_eval(args)
