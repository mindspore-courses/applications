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
"""file for inferring"""

import argparse
import time

import numpy as np
from PIL import Image
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore import context
import mindspore.ops as ops

from src.model.generator import Generator
from src.dataset.create_loader import create_test_dataloader

def main(args):
    """Inferring process"""
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id, save_graphs=False)
    test_ds = create_test_dataloader(1, args.test_LR_path, inference=True)
    data_size = test_ds.get_dataset_size()
    test_data_loader = test_ds.create_dict_iterator()
    generator = Generator(args.scale)
    params = load_checkpoint(args.generator_path)
    print("======load checkpoint")
    load_param_into_net(generator, params)
    op = ops.ReduceSum(keep_dims=False)
    print("=======starting test=====")
    time_total = 0
    i = 0
    for data in test_data_loader:
        time_begin = time.time()
        lr = data['LR']
        output = generator(lr)
        time_total += time.time() - time_begin
        output = op(output, 0).asnumpy()
        output = np.clip(output, -1.0, 1.0)
        output = ((output + 1.0) / 2.0).transpose(1, 2, 0)
        result = Image.fromarray((output * 255.0).astype(np.uint8))
        # save the output image
        result.save(f"./output/{i}.jpg")
        i += 1
    print("Total %d images need %.0fms, per image needs %.0fms." % (data_size, time_total * 1000, \
        (time_total / data_size) * 1000))
    print("Inference End.")

def parse_args():
    """Add argument"""
    parser = argparse.ArgumentParser(description="SRGAN infer")
    parser.add_argument("--test_LR_path", type=str)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--generator_path", type=str)
    parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'))
    return parser.parse_args()

if __name__ == '__main__':
    args_list = parse_args()
    main(args_list)
