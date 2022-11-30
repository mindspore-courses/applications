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

"""Evaluating the model"""

import numpy as np
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose

from configs.dataset_config import parse_args
from configs.net_configs import get_network
from dataset.mindspore_coco import COCODataset
from utils.utils import get_final_preds


def evaluate_model(model, dataset, output_path):
    """
    Evaluate and Print indicators.

    Args:
        model (nn.Cell): Lite-HRNet network.
        dataset (COCODataset): Dataset for evaluating.
        output_path (str): The path for storing evaluation json files.
    """

    num_samples = len(dataset)
    all_preds = np.zeros(
        (num_samples, 17, 3),
        dtype=np.float32
        )

    all_boxes = np.zeros((num_samples, 6))
    image_path = []

    for i, data in enumerate(dataset):
        input_data, target, meta = data[0], data[1], data[3]
        input_height, input_width = input_data.shape[1], input_data.shape[2]
        input_data = ms.Tensor(input_data, ms.float32).reshape(1, 3, input_height, input_width)
        shit = model(input_data).asnumpy()
        target = target.reshape(shit.shape)
        c = meta['center'].reshape(1, 2)
        s = meta['scale'].reshape(1, 2)
        score = meta['score']
        preds, maxvals = get_final_preds(shit, c, s)
        all_preds[i:i + 1, :, 0:2] = preds[:, :, 0:2]
        all_preds[i:i + 1, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[i:i + 1, 0:2] = c[:, 0:2]
        all_boxes[i:i + 1, 2:4] = s[:, 0:2]
        all_boxes[i:i + 1, 4] = np.prod(s*200, 1)
        all_boxes[i:i + 1, 5] = score
        image_path.append(meta['image'])

    dataset.evaluate(0, all_preds, output_path, all_boxes, image_path)

def main(args):
    """
    Main evaluating function

    Args:
        args (argparse.ArgumentParser): Main configs argparser.
    """

    model_path = args.checkpoint_path
    net_type = args.model_type
    res_type = args.image_size
    root_dir = args.root

    #Define a training dataset
    trans = Compose([py_vision.ToTensor(),
                     py_vision.Normalize(
                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    eval_ds = COCODataset(args, root_dir, "val2017", False, transform=trans)

    #Constructing network model
    res = str(res_type[1]) + "x" + str(res_type[0])
    if "lite" in net_type:
        model_path += "litehrnet_" + str(net_type[-2:]) + "_" + "coco" + "_" + res + ".ckpt"
    else:
        model_path += net_type + "_litehrnet_18_" + "coco" + "_" + res + ".ckpt"
    net = get_network(net_type, "COCO")
    param_dict = load_checkpoint(model_path)
    load_param_into_net(net, param_dict)

    #Evaluating
    evaluate_model(net, eval_ds, args.output_dir)


if __name__ == "__main__":
    eval_args = parse_args()
    main(eval_args)
