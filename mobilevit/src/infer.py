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
""" mobilevit infer script. """

import argparse
import numpy as np

import mindspore
from mindspore import context, Tensor, load_checkpoint, load_param_into_net
from mindspore.train import Model
import mindspore.dataset as ds

from utils.imageprocess import show_result
from utils.images import read_dataset
from utils.generator import DatasetGenerator
from utils.infer_tansform import infer_transform
from datasets.imagenet import ImageNet
from models.mobilevit import MobileViT


def mobilevit_infer(args_opt):
    """mobilevit infer"""

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data pipeline.
    dataset_analyse = ImageNet(args_opt.data_url,
                               split="val",
                               num_parallel_workers=8,
                               resize=args_opt.resize,
                               batch_size=args_opt.batch_size)

    # Create model.
    network = MobileViT(model_type=args_opt.model_type, num_classes=args_opt.num_classes)

    # load pertain model
    param_dict = load_checkpoint(args_opt.pretrained_model)
    load_param_into_net(network, param_dict)

    # Init the model.
    model = Model(network)

    # read inference picture
    image_list, image_label = read_dataset(args_opt.infer_url)
    columns_list = ('image', 'label')
    dataset_infer = ds.GeneratorDataset(DatasetGenerator(image_list, image_label),
                                        column_names=list(columns_list),
                                        num_parallel_workers=args_opt.num_parallel_workers,
                                        python_multiprocessing=False)
    dataset_infer = infer_transform(dataset_infer, columns_list, args_opt.resize)

    # read data for inference
    for i, image in enumerate(dataset_infer.create_dict_iterator(output_numpy=True)):
        image = image["image"]
        image = Tensor(image, mindspore.float32)
        prob = model.predict(image)
        label = np.argmax(prob.asnumpy(), axis=1)
        predict = dataset_analyse.index2label[int(label)]
        output = {int(label): predict}
        print(output)
        show_result(img=image_list[i], result=output, out_file=image_list[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT infer.')
    parser.add_argument("--resize", type=int, default=256, help="Image resize.")
    parser.add_argument("--data_url", default="./dataset/", help="Location of data.")
    parser.add_argument('--model_type', default='xx_small', type=str, metavar='model_type')
    parser.add_argument("--batch_size", type=int, default=100, help="Number of batch size.")
    parser.add_argument('--num_classes', type=int, default=1001, help='Number of classification.')
    parser.add_argument("--infer_url", default='./dataset/infer', help="Location of inference data.")
    parser.add_argument('--device_target', type=str, default="CPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--pretrained_model', default='./mobilevit_xxs.ckpt', type=str, metavar='PATH')
    parser.add_argument("--num_parallel_workers", type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_known_args()[0]
    mobilevit_infer(args)
