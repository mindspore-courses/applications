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

import cv2
import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from panopticapi.evaluation import pq_compute
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net

from models.segmentation import DETRsegm
from models import detr
from datasets import cocopanoptic
from util import post_process, post_process_seg, post_process_pano


def parse_args():
    """Evaluation parameters"""
    parser = argparse.ArgumentParser(description='infer DTER')
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--coco_dir', default='./coco', type=str)
    parser.add_argument('--panoptic_dir', default='./coco_panoptic', type=str)
    parser.add_argument('--result_dir', default='./detr/result', type=str)
    parser.add_argument('--resume_dir', default='./detr/resume', type=str)
    parser.add_argument('--resnet', default='resnet50', choices=['resnet50', 'resnet101'], type=str)
    parser.add_argument('--dilation', action='store_true')
    return parser.parse_args()


def get_res(args):
    """
    Get the result for evaluation.

    Args:
        args: Evaluation parameters.
    """
    if args.device == 'CPU':
        context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=args.device_id)
    net = detr.bulid_detr(resnet=args.resnet, return_interm_layers=True, num_classes=250, is_dilation=args.dilation)
    net = DETRsegm(net, freeze_detr=False)
    res_file = f'result_{args.resnet}'
    if args.dilation:
        res_file += '_dc'
        checkpoint = f'{args.resnet}_dc_seg.ckpt'
    else:
        checkpoint = f'{args.resnet}_seg.ckpt'
    seg_file = res_file + '_seg.json'
    box_file = res_file + '_box.json'
    pano_file = res_file + '_pano.json'
    seg_file_path = os.path.join(args.result_dir, seg_file)
    box_file_path = os.path.join(args.result_dir, box_file)
    pano_file_path = os.path.join(args.result_dir, pano_file)
    output_dir = os.path.join(args.result_dir, res_file)
    ckpt_path = os.path.join(args.resume_dir, checkpoint)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    dataset = cocopanoptic.build(img_set='val', batch=1, shuffle=False,
                                 coco_dir=args.coco_dir, pano_dir=args.panoptic_dir)
    result_box = []
    result_seg = []
    result_pano = {'annotations': []}
    step = 0
    infer_time = 0
    for data in dataset.create_dict_iterator():
        img = data['img']
        mask = data['img_mask']
        len_list = data['len_list']
        orig_size = data['orig_size']
        size = data['size']
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
        res_pano = post_process_pano(out, size, orig_size)
        results = post_process(out, orig_size)
        results = post_process_seg(results, out, orig_size, size)
        for bs_id, r in enumerate(results):
            img_id = tgts[bs_id]['img_id']
            for idx, j in enumerate(r['labels']):
                if j in [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
                    continue
                category_id = j
                box = r['boxes'][idx]
                score = r['scores'][idx]
                seg_mask = np.array(r["masks"][idx].asnumpy()[:, :, np.newaxis], dtype=np.uint8, order="F")
                seg = mask_util.encode(seg_mask)[0]
                seg["counts"] = seg["counts"].decode("utf-8")
                result_box.append({
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "bbox": list(box.asnumpy().astype(np.float64)),
                    "score": float(score)
                })
                result_seg.append({
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "segmentation": seg,
                    "score": float(score)
                })
            image_id = int(img_id)
            file_name = f"{image_id:012d}.png"
            file_path = os.path.join(output_dir, file_name)
            cv2.imwrite(file_path, res_pano[bs_id]["file_name"])
            res_pano[bs_id]["image_id"] = image_id
            res_pano[bs_id]["file_name"] = file_name
            result_pano['annotations'].extend(res_pano)
        step += 1
        print('infer_time:', t1 - t0, 'infer_time_avg:', infer_time / step)
    print('infer_time_all:', infer_time, 'infer_time_avg:', infer_time / step)
    with open(seg_file_path, 'w', encoding='u8') as f:
        json.dump(result_seg, f)
    with open(box_file_path, 'w', encoding='u8') as f:
        json.dump(result_box, f)
    with open(pano_file_path, 'w', encoding='u8') as f:
        json.dump(result_pano, f)


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
    seg_file = res_file + '_seg.json'
    box_file = res_file + '_box.json'
    pano_file = res_file + '_pano.json'
    seg_file_path = os.path.join(args.result_dir, seg_file)
    box_file_path = os.path.join(args.result_dir, box_file)
    pano_file_path = os.path.join(args.result_dir, pano_file)
    output_dir = os.path.join(args.result_dir, res_file)
    if not os.path.exists(output_dir):
        print(f'{output_dir} does not exist! Make new Folder!')
        os.makedirs(output_dir)
    for i in [seg_file, box_file, pano_file]:
        if i not in os.listdir(args.result_dir):
            get_res(args)
            break
    coco_gt_json = f'{args.coco_dir}/annotations/instances_val2017.json'
    cocogt = COCO(coco_gt_json)
    cocodt = cocogt.loadRes(seg_file_path)
    cocoeval = COCOeval(cocogt, cocodt, "segm")
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()
    cocodt = cocogt.loadRes(box_file_path)
    cocoeval = COCOeval(cocogt, cocodt, "bbox")
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()
    pano_gt_json = f'{args.panoptic_dir}/annotations/panoptic_val2017.json'
    pano_gt_folder = f'{args.panoptic_dir}/panoptic_val2017'
    pq_compute(pano_gt_json, pano_file_path, pano_gt_folder, output_dir)


if __name__ == "__main__":
    coco_eval(parse_args())
