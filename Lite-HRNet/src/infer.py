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

"""Infer"""

import os

import cv2
import numpy as np
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
import mindspore.dataset.vision.py_transforms as py_vision

from configs.dataset_config import parse_args
from configs.net_configs import get_network
from utils.utils import get_max_preds


def main(args):
    """
    Main inference function
    Args:
        args (argparse.ArgumentParser): Main configs argparser.
    """

    model_path = args.checkpoint_path
    net_type = args.model_type
    res_type = args.image_size
    trans = py_vision.ToTensor()

    #Constructing network model
    res = str(res_type[1]) + "x" + str(res_type[0])
    if "lite" in net_type:
        model_path += "litehrnet_" + str(net_type[-2:]) + "_" + "coco" + "_" + res + ".ckpt"
    else:
        model_path += net_type + "_litehrnet_18_" + "coco" + "_" + res + ".ckpt"
    net = get_network(net_type, "COCO")
    param_dict = load_checkpoint(model_path)
    load_param_into_net(net, param_dict)

    #Start infer
    for filename in os.listdir(args.infer_data_root):
        origin_img = cv2.imread(args.infer_data_root + '/' + filename)
        origin_h, origin_w, _ = origin_img.shape
        scale_factor = [origin_w/res_type[0], origin_h/res_type[1]]

        # resize to given shape and convert to tensor
        img = cv2.resize(origin_img, res_type)
        print(img.shape)
        img = trans(img)
        img = np.expand_dims(img, axis=0)
        img = ms.Tensor(img)

        # Infer
        heatmap_pred = net(img).asnumpy()
        pred, _ = get_max_preds(heatmap_pred)

        # Postprocess
        pred = pred.reshape(pred.shape[0], -1, 2)
        print(pred[0])
        pre_landmark = pred[0] * 4 * scale_factor
        # Draw points
        for (x, y) in pre_landmark.astype(np.int32):
            cv2.circle(origin_img, (x, y), 3, (255, 255, 255), -1)

        # Save image
        cv2.imwrite(os.path.join(args.out_data_root, filename), origin_img)


if __name__ == "__main__":
    infer_args = parse_args()
    main(infer_args)
