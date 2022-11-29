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
# ===========================================================================
"""
Network config setting, will be used in train.py, eval.py and infer.py
"""
import argparse
import ast
from easydict import EasyDict as ed


def parse_args():
    """
    Parse parameters.

     Returns:
        parsed parameters.
    """

    parser = argparse.ArgumentParser()

    # Device type
    parser.add_argument('--device_target', default='Ascend', choices=['CPU', 'GPU', 'Ascend'], type=str,
                        help="Device id of CPU, GPU or Ascend.")
    parser.add_argument('--enable_profiling', default=False, type=ast.literal_eval, help="Whether to enable profiling.")

    # Dataset path
    parser.add_argument('--data_root', default='../../coco2017bk', type=str,
                        help="File path of dataset in training.")

    # MaskRcnn training
    parser.add_argument('--only_create_dataset', default=False, type=ast.literal_eval,
                        help="Whether to create dataset.")
    parser.add_argument('--run_distribute', default=False, type=ast.literal_eval, help="Whether to run distribute.")
    parser.add_argument('--do_train', default=True, type=ast.literal_eval, help="Whether to do train.")
    parser.add_argument('--do_eval', default=False, type=ast.literal_eval, help="Whether to do eval.")
    parser.add_argument('--dataset', default='coco', type=str, help="Dataset name")
    parser.add_argument('--pre_trained', default='../../maskrcnnr5/checkpoint/resnet50.ckpt',
                        type=str, help="File path of pretrained checkpoint in training.")
    parser.add_argument('--device_id', default=0, type=int, help="Target device id.")
    parser.add_argument('--device_num', default=1, type=int, help="Target device number.")
    parser.add_argument('--rank_id', default=0, type=int, help="Target device rank id.")

    # MaskRcnn evaluation
    parser.add_argument('--ann_file', default='../../coco2017bk/annotations/instances_val2017.json',
                        type=str, help="File path of cocodataset annotations.")
    parser.add_argument('--checkpoint_path', default='../checkpoint/maskrcnn_coco2017_acc32.9.ckpt',
                        type=str, help="File path of pretrained checkpoint in evaluation.")


    # MaskRcnn ResNet50 export"
    parser.add_argument('--batch_size_export', default=1, type=int, help="Choose batch size to export.")
    parser.add_argument('--ckpt_file', default='./checkpoint/maskrcnn_gpu_coco.ckpt',
                        type=str, help="File path of pretrained checkpoint to export.")
    parser.add_argument('--file_name', default='./checkpoint/maskrcnn_coco2017_acc32.9.ckpt',
                        type=str, help="File path of pretrained checkpoint in evaluation.")

    # MaskRcnn ResNet50 inference
    parser.add_argument('--img_path', default='../../coco2017bk/val2017',
                        type=str, help="File path of dataset in inference.")
    parser.add_argument('--ann_path', default='../../coco2017bk/annotations/instances_val2017.json',
                        type=str, help="File path of annotations in inference.")
    parser.add_argument('--result_path', default='./results',
                        type=str, help="File path of detected results in inference.")

    # ==============================================================================

    parser.add_argument('--img_width', default=1280, type=int, help="The input image width.")
    parser.add_argument('--img_height', default=768, type=int, help="The input image height.")
    parser.add_argument('--keep_ratio', default=True, type=ast.literal_eval,
                        help="Whether to keep the same image scaling ratio.")
    parser.add_argument('--flip_ratio', default=0.5, type=float, help="The flip ratio.")
    parser.add_argument('--expand_ratio', default=1.0, type=float, help="The expand ratio.")

    parser.add_argument('--max_instance_count', default=128, type=int, help="The maximum instance count to detect.")
    parser.add_argument('--mask_shape', default=[28, 28], type=list, help="The mask shape.")

    # anchor
    parser.add_argument('--feature_shapes', default=[(192, 320), (96, 160), (48, 80), (24, 40), (12, 20)],
                        type=list, help="The feature shape.")
    parser.add_argument('--anchor_scales', default=[8], type=list, help="The anchor scales.")
    parser.add_argument('--anchor_ratios', default=[0.5, 1.0, 2.0], type=list, help="The anchor ratios.")
    parser.add_argument('--anchor_strides', default=[4, 8, 16, 32, 64], type=list, help="The anchor strides.")
    parser.add_argument('--num_anchors', default=3, type=int, help="The number of anchors.")

    # resnet
    parser.add_argument('--resnet_block', default=[3, 4, 6, 3],
                        type=list, help="The resnet bolck shape.")
    parser.add_argument('--resnet_in_channels', default=[64, 256, 512, 1024],
                        type=list, help="The resnet input channels.")
    parser.add_argument('--resnet_out_channels', default=[256, 512, 1024, 2048],
                        type=list, help="The resnet output channels.")

    # fpn
    parser.add_argument('--fpn_in_channels', default=[256, 512, 1024, 2048],
                        type=list, help="The fpn input channels.")
    parser.add_argument('--fpn_out_channels', default=256, type=int, help="The fpn output channels.")
    parser.add_argument('--fpn_num_outs', default=5, type=int, help="The number of fpn outputs.")

    # rpn
    parser.add_argument('--rpn_in_channels', default=256, type=int, help="The rpn input channels.")
    parser.add_argument('--rpn_feat_channels', default=256, type=int, help="The rpn feature channels.")
    parser.add_argument('--rpn_loss_cls_weight', default=1.0, type=float, help="The rpn loss classification weight.")
    parser.add_argument('--rpn_loss_reg_weight', default=1.0, type=float, help="The rpn loss regression weight.")
    parser.add_argument('--rpn_cls_out_channels', default=1, type=int, help="The rpn classification output channels.")
    parser.add_argument('--rpn_target_means', default=[0., 0., 0., 0.], type=list, help="The rpn target means.")
    parser.add_argument('--rpn_target_stds', default=[1.0, 1.0, 1.0, 1.0], type=list, help="The rpn target standards.")

    # bbox_assign_sampler
    parser.add_argument('--neg_iou_thr', default=0.3, type=float, help="The negative iou threshold.")
    parser.add_argument('--pos_iou_thr', default=0.7, type=float, help="The positive iou threshold.")
    parser.add_argument('--min_pos_iou', default=0.3, type=float, help="The minimum positive iou.")
    parser.add_argument('--num_bboxes', default=245520, type=int, help="The number of bboxes.")
    parser.add_argument('--num_gts', default=128, type=int, help="The number pf ground truth.")
    parser.add_argument('--num_expected_neg', default=256, type=int, help="The number of expected negative.")
    parser.add_argument('--num_expected_pos', default=128, type=int, help="The number of expected positive.")

    # proposal
    parser.add_argument('--activate_num_classes', default=256, type=int, help="The activate number of classes.")
    parser.add_argument('--use_sigmoid_cls', default=True, type=ast.literal_eval,
                        help="Whether to use sigmoid for classification.")

    # roi_align
    parser.add_argument('--roi_layer', default=ed(type='RoIAlign', out_size=7, mask_out_size=14, sample_num=2),
                        type=dict, help="The roi layer.")
    parser.add_argument('--roi_align_out_channels', default=256, type=int, help="The roi align output channels.")
    parser.add_argument('--roi_align_featmap_strides', default=[4, 8, 16, 32], type=int,
                        help="The strides of roi align feature map.")
    parser.add_argument('--roi_align_finest_scale', default=56, type=int, help="The roi align finest scale.")
    parser.add_argument('--roi_sample_num', default=640, type=int, help="The number of roi sample.")

    # bbox_assign_sampler_stage2
    parser.add_argument('--neg_iou_thr_stage2', default=0.5, type=float, help="The negative iou threshold for stage2.")
    parser.add_argument('--pos_iou_thr_stage2', default=0.5, type=float, help="The positive iou threshold for stage2.")
    parser.add_argument('--min_pos_iou_stage2', default=0.5, type=float, help="The minimum positive iou for stage2.")
    parser.add_argument('--num_bboxes_stage2', default=2000, type=int, help="The number of bboxes for stage2.")
    parser.add_argument('--num_expected_pos_stage2', default=128, type=int,
                        help="The number of expected positive in stage2.")
    parser.add_argument('--num_expected_neg_stage2', default=512, type=int,
                        help="The number of expected negative in stage2.")
    parser.add_argument('--num_expected_total_stage2', default=512, type=int,
                        help="The number of expected total for stage2.")

    # rcnn
    parser.add_argument('--rcnn_num_layers', default=2, type=int, help="The rcnn number of layers.")
    parser.add_argument('--rcnn_in_channels', default=256, type=int, help="The rcnn input channels.")
    parser.add_argument('--rcnn_fc_out_channels', default=1024, type=int, help="The rcnn fc output channels.")
    parser.add_argument('--rcnn_mask_out_channels', default=256, type=int, help="The rcnn mask output channels.")
    parser.add_argument('--rcnn_loss_cls_weight', default=1, type=int, help="The weight of rcnn loss classification.")
    parser.add_argument('--rcnn_loss_reg_weight', default=1, type=int, help="The weight of rcnn loss regression.")
    parser.add_argument('--rcnn_loss_mask_fb_weight', default=1, type=int,
                        help="The weight of rcnn loss mask feedback.")
    parser.add_argument('--rcnn_target_means', default=[0., 0., 0., 0.],
                        type=list, help="The rcnn target means.")
    parser.add_argument('--rcnn_target_stds', default=[0.1, 0.1, 0.2, 0.2],
                        type=list, help="The rcnn target stds.")

    # train proposal
    parser.add_argument('--rpn_proposal_nms_across_levels', default=False, type=ast.literal_eval,
                        help="Whether to do rpn proposal nms across levels.")
    parser.add_argument('--rpn_proposal_nms_pre', default=2000, type=int, help="The rpn proposal nms preparation.")
    parser.add_argument('--rpn_proposal_nms_post', default=2000, type=int, help="The rpn proposal nms post.")
    parser.add_argument('--rpn_proposal_max_num', default=2000, type=int, help="The rpn proposal max number.")
    parser.add_argument('--rpn_proposal_nms_thr', default=0.7, type=float, help="The rpn proposal nms threshold.")
    parser.add_argument('--rpn_proposal_min_bbox_size', default=0, type=int,
                        help="The rpn proposal min bounding box size.")

    # test proposal
    parser.add_argument('--rpn_nms_across_levels', default=False, type=ast.literal_eval,
                        help="Whether to do rpn nms across levels.")
    parser.add_argument('--rpn_nms_pre', default=1000, type=int, help="The rpn nms preparation.")
    parser.add_argument('--rpn_nms_post', default=1000, type=int, help="The rpn nms post.")
    parser.add_argument('--rpn_max_num', default=1000, type=int, help="The rpn max number.")
    parser.add_argument('--rpn_nms_thr', default=0.7, type=float, help="The rpn nms threshold.")
    parser.add_argument('--rpn_min_bbox_min_size', default=0, type=int, help="The min size of rpn min bounding box.")
    parser.add_argument('--test_score_thr', default=0.05, type=float, help="The test score threshold.")
    parser.add_argument('--test_iou_thr', default=0.5, type=float, help="The test iou threshold.")
    parser.add_argument('--test_max_per_img', default=100, type=int, help="The test maximum per img.")
    parser.add_argument('--test_batch_size', default=2, type=int, help="The test batch size.")

    parser.add_argument('--rpn_head_use_sigmoid', default=True, type=ast.literal_eval,
                        help="Whether to use sigmoid for rpn heads.")
    parser.add_argument('--rpn_head_weight', default=1.0, type=float, help="The rpn head weight.")
    parser.add_argument('--mask_thr_binary', default=0.5, type=float, help="The binary mask threshold.")

    # LR
    parser.add_argument('--base_lr', default=0.02, type=float, help="The basic learning rate.")
    parser.add_argument('--base_step', default=59633, type=int, help="The basic step.")
    parser.add_argument('--total_epoch', default=13, type=int, help="The total epoch.")
    parser.add_argument('--warmup_step', default=500, type=int, help="The warmup step.")
    parser.add_argument('--warmup_ratio', default=1/3.0, type=float, help="The warmup ratio.")
    parser.add_argument('--sgd_momentum', default=0.9, type=float, help="The sgd momentum.")

    # train
    parser.add_argument('--batch_size', default=2, type=int,
                        help="Batch size, different size datasets have different values.")
    parser.add_argument('--loss_scale', default=1, type=int, help="The loss scale.")
    parser.add_argument('--momentum', default=0.91, type=float, help="The momentum.")
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="The weight decay.")
    parser.add_argument('--pretrain_epoch_size', default=0, type=int, help="The pretrained epoch size.")
    parser.add_argument('--epoch_size', default=12, type=int, help="The epoch size.")
    parser.add_argument('--save_checkpoint', default=True, type=ast.literal_eval, help="Whether to save checkpoint.")
    parser.add_argument('--save_checkpoint_epochs', default=1, type=int, help="Set the epoch to save checkpoint.")
    parser.add_argument('--keep_checkpoint_max', default=12, type=int, help="The max checkpoint to keep.")
    parser.add_argument('--save_checkpoint_path', default='./', type=str,
                        help="File path of pretrained checkpoint to save.")

    # cocodataset
    parser.add_argument('--mindrecord_dir', default='./MindRecord_COCO/MindRecord_COCO', type=str,
                        help="File path of MindRecord to save/read.")
    parser.add_argument('--train_data_type', default='train2017', type=str,
                        help="The data type for training (it is not necessary for other dataset.).")
    parser.add_argument('--val_data_type', default='val2017', type=str,
                        help="The data type for validation (it is not necessary for other dataset.).")
    parser.add_argument('--instance_set', default='annotations/instances_{}.json', type=str,
                        help="The instance set for cocodataset (it is not necessary for other dataset.).")
    parser.add_argument('--data_classes', default=('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                                                   'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                                                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                                                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                                                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                                                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                                                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                                                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                                                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                                                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                                                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                                                   'refrigerator', 'book', 'clock', 'vase', 'scissors',
                                                   'teddy bear', 'hair drier', 'toothbrush'), type=list,
                        help="The data classes for cocodataset (it is not necessary for other dataset.).")
    parser.add_argument('--num_classes', default=81, type=int,
                        help="The number of classes for cocodataset (it is not necessary for other dataset.).")
    return parser.parse_args(args=[])

config = parse_args()
