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
# ======================================================================
"""infer code"""
import argparse
import random

import cv2
import matplotlib.pyplot as plt
import mindspore
from mindspore import nn
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore import context, ops
from mindspore.common.tensor import Tensor
from mindspore import load_checkpoint, load_param_into_net

from models import detr
from models.segmentation import DETRsegm


color_list = ['r', 'rosybrown', 'lightcoral', 'salmon', 'darksalmon', 'coral',
              'y', 'darkorange', 'burlywood', 'tan', 'moccasin', 'orange', 'wheat',
              'g', 'yellowgreen', 'greenyellow', 'lawngreen', 'darkseagreen',
              'b', 'springgreen', 'aquamarine', 'turquoise', 'lightseagreen',
              'c', 'm', 'skyblue', 'lime', 'pink', 'purple', 'indigo', 'plum',
              'orchid', 'hotpink', 'deeppink', 'palevioletred', 'crimson'
              ]

cats_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
             6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
             11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
             16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
             22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
             28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
             35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
             40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
             44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
             50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
             55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
             60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
             65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
             74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
             79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
             85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
             }


def parse_args():
    """Inference parameters"""
    parser = argparse.ArgumentParser(description='infer DTER')
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--img_path', default='../images/000000056288.jpg', type=str)
    parser.add_argument('--resume_path', default='../resume/resnet50.ckpt', type=str)
    parser.add_argument('--resnet', default='resnet50', choices=['resnet50', 'resnet101'], type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--is_segmentation', action='store_true')
    return parser.parse_args()


def im_save(outs, origin_img):
    """
    Save image.

    Args:
        outs (Tensor): The detr outputs.
        origin_img (numpy.ndarray): The original images for inference.
    """
    origin_h, origin_w, _ = origin_img.shape
    softmax = nn.Softmax()
    scores = softmax(outs['pred_logits'][0])
    tgts = []
    for idx, score in enumerate(scores):
        s_max = score.max(-1)
        if s_max > 0.9:
            img_id = ops.Argmax(-1)(score)
            if img_id < 91:
                box = outs['pred_boxes'][0][idx]
                tgts.append([int(img_id), round(float(s_max), 2), box.asnumpy()])
    cat2color = dict()
    plt.imshow(origin_img)
    for tgt in tgts:
        name = 'unk'
        if tgt[0] in cats_dict:
            name = cats_dict[tgt[0]]
        score = tgt[1]
        box = tgt[2].copy()
        box[0] = box[0] - 0.5 * box[2]
        box[1] = box[1] - 0.5 * box[3]
        if not (box > 0).all():
            continue
        box = list(box * [origin_w, origin_h, origin_w, origin_h])
        print(name, score, box)
        if name not in cat2color:
            c = random.sample(color_list, 1)[0]
            while c in cat2color.values():
                c = random.sample(color_list, 1)[0]
            cat2color[name] = c
        else:
            c = cat2color[name]
        plt.gca().add_patch(plt.Rectangle(xy=(box[0], box[1]), width=box[2],
                                          height=box[3], edgecolor=c, fill=False, linewidth=2))
        plt.text(box[0], box[1], f'{name}:{score}', fontsize=8, bbox=dict(fc=c, ec=c, pad=0.5))
    plt.savefig('img_detr.png')


def infer(args):
    """
    Inference code.

    Args:
        args: Inference parameters.
    """
    print('args:', args)
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device)
    is_segmentation = args.is_segmentation
    print('Build detr network......')
    if is_segmentation:
        net = detr.bulid_detr(resnet=args.resnet, return_interm_layers=is_segmentation,
                              num_classes=250, is_dilation=args.dilation)
        net = DETRsegm(net, freeze_detr=False)
    else:
        net = detr.bulid_detr(resnet=args.resnet, return_interm_layers=False,
                              num_classes=91, is_dilation=args.dilation)
    param_dict = load_checkpoint(args.resume_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    origin_img = cv2.imread(args.img_path)
    b, g, r = cv2.split(origin_img)
    origin_img = cv2.merge([r, g, b])
    origin_h, origin_w, _ = origin_img.shape
    normalize_op = c_vision.Normalize([123.7, 116.3, 103.5], [58.4, 57.1, 57.4])
    img = normalize_op(origin_img)
    img = Tensor(img)
    transpose = ops.Transpose()
    img = transpose(img, (2, 0, 1))
    img_mask = ops.Zeros()((1, origin_h, origin_w), mindspore.float32)
    expand_dims = ops.ExpandDims()
    img = expand_dims(img, 0)
    outs = net(img, img_mask)
    im_save(outs, origin_img)


if __name__ == '__main__':
    infer(parse_args())
