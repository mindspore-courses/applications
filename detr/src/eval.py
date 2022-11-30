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
"""eval"""
import argparse
import json
import os
import time

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net

from models import detr
from datasets import coco
from util import post_process


def parse_args():
    """Evaluation parameters"""
    parser = argparse.ArgumentParser(description='infer DTER')
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--coco_dir', default='./coco', type=str)
    parser.add_argument('--result_dir', default='./detr/result', type=str)
    parser.add_argument('--resume_dir', default='./detr/resume', type=str)
    parser.add_argument('--resnet', default='resnet50', choices=['resnet50', 'resnet101'], type=str)
    parser.add_argument('--dilation', action='store_true')
    return parser.parse_args()


def gen_json(args):
    """
    Generate the resulting json file.

    Args:
        args: Evaluation parameters.
    """
    if args.device == 'CPU':
        context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=args.device_id)
    print('Build detr network......')
    net = detr.bulid_detr(resnet=args.resnet, return_interm_layers=False,
                          num_classes=91, is_dilation=args.dilation)
    ckpt_file = args.resnet
    res_file = f'result_{args.resnet}'
    if args.dilation:
        ckpt_file += '_dc'
        res_file += '_dc'
    ckpt_file += '.ckpt'
    res_file += '.json'
    ckpt_file = os.path.join(args.resume_dir, ckpt_file)
    res_path = os.path.join(args.result_dir, res_file)
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    dataset = coco.build(img_set='val', batch=1, shuffle=False, coco_dir=args.coco_dir)
    result = []
    step = 0
    infer_time = 0
    for data in dataset.create_dict_iterator():
        img = data['img']
        mask = data['img_mask']
        len_list = data['len_list']
        orig_size = data['orig_size']
        print('step:', step, 'img:', img.shape, 'mask:', mask.shape, 'len_list:', len_list)
        tgts = []
        begin = 0
        bs = img.shape[0]
        for i in range(bs):
            target = {}
            if len_list[i][0] < 2:
                end = begin + int(len_list[i][0])
            else:
                end = begin + len_list[i][0]
            target = {'labels': data['cats'][0][begin:end],
                      'boxes': data['bbox'][0][begin:end],
                      'masks': data['masks'][0][begin:end],
                      'img_id': data['img_id'][i][0]}
            begin = end
            tgts.append(target)
        t0 = time.time()
        out = net(img, mask)
        t1 = time.time()
        infer_time += (t1 - t0)
        results = post_process(out, orig_size)
        for bs_id, r in enumerate(results):
            img_id = tgts[bs_id]['img_id']
            for idx, j in enumerate(r['labels']):
                if j in [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
                    continue
                category_id = j
                box = r['boxes'][idx]
                score = r['scores'][idx]
                result.append({
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "bbox": list(box.asnumpy().astype(np.float64)),
                    "score": float(score)
                })
        step += 1
        print('infer_time:', t1 - t0, 'infer_time_avg:', infer_time / step)
    print('infer_time_all:', infer_time, 'infer_time_avg:', infer_time / step)
    with open(res_path, 'w', encoding='u8') as f:
        json.dump(result, f)


def coco_eval(args):
    """
    Evaluation of target detection results.

    Args:
        args: Evaluation parameters.
    """
    if not os.path.exists(args.result_dir):
        print(f'{args.result_dir} does not exist! Make new Folder!')
        os.makedirs(args.result_dir)
    res_file = f'result_{args.resnet}'
    if args.dilation:
        res_file += '_dc'
    res_file += '.json'
    res_path = os.path.join(args.result_dir, res_file)
    if res_file not in os.listdir(args.result_dir):
        gen_json(args)
    coco_gt_json = f'{args.coco_dir}/annotations/instances_val2017.json'
    cocogt = COCO(coco_gt_json)
    cocodt = cocogt.loadRes(res_path)
    cocoeval = COCOeval(cocogt, cocodt, "bbox")
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()


if __name__ == "__main__":
    coco_eval(parse_args())
