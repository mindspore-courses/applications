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
"""infer"""
import os
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.train import Model
import mindspore.dataset as ds
from mindspore.common import set_seed
from mindspore import context
import mindspore.dataset.vision.c_transforms as transforms
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision.utils import Inter

from utils.cell import cast_amp
from utils.tools import pretrained, show_result, index2label
from models.convnext import convnext_tiny
from models.loss import get_criterion, NetWithLoss
from config.convnext_config import parse_args


def infer(args):
    """"infer ConvNext model."""
    set_seed(args.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    # get model
    net = convnext_tiny(args)
    cast_amp(net, args)
    criterion = get_criterion(args)
    NetWithLoss(net, criterion)
    if args.pretrained:
        pretrained(args, net)
    # Read data for inference
    dataset_infer = ds.ImageFolderDataset(os.path.join(args.data_url, "infer"), shuffle=True)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    # test transform complete
    transform_img = [
        transforms.Decode(),
        transforms.Resize(int(256 / 224 * 224), interpolation=Inter.PILCUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=mean, std=std),
        transforms.HWC2CHW()
        ]
    transform_label = C.TypeCast(mstype.int32)
    dataset_infer = dataset_infer.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                                      operations=transform_img)
    dataset_infer = dataset_infer.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                      operations=transform_label)
    one_hot = C.OneHot(num_classes=args.num_classes)
    dataset_infer = dataset_infer.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                      operations=one_hot)
    # apply batch operations
    dataset_infer = dataset_infer.batch(1, drop_remainder=True,
                                        num_parallel_workers=args.num_parallel_workers)
    model = Model(net)
    for i, image in enumerate(dataset_infer.create_dict_iterator(output_numpy=True)):
        print('i is : ', i)
        image = image["image"]
        image = ms.Tensor(image)
        prob = model.predict(image)
        print("predict is finished.")
        label = np.argmax(prob.asnumpy(), axis=1)
        mapping = index2label(args)
        output = {int(label): mapping[int(label)]}
        print(output)
        show_result(img="/home/ma-user/work/imagenet2012/infer/n01440764/ILSVRC2012_test_00000293.JPEG",
                    result=output,
                    out_file="/home/ma-user/work/imagenet2012/infer/ILSVRC2012_test_00000293.JPEG")


if __name__ == '__main__':
    infer(parse_args())
