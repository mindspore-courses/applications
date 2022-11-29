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
"""Parse parameters function."""

import argparse

from mindspore.dataset.vision.utils import Inter


def common_parse_args(parser):
    """
    Common parse parameters.

     Returns:
        parsed parameters.
    """
    # train
    parser.add_argument('--seed', default=2022, type=int, help='the random seed number.')
    parser.add_argument('--use_parallel', default=False, type=bool, help='whether to use parallel.')
    parser.add_argument('--device_target',
                        default='Ascend',
                        choices=['CPU', 'GPU', 'Ascend'],
                        type=str, help='device need of training input data.d of CPU, GPU or Ascend.')
    parser.add_argument('--mode', default='GRAPH_MODE', type=str, help='model run mode.')
    parser.add_argument('--device_id', default=0, type=int, help='the number of device id.')
    parser.add_argument('--parallel_mode', default='DATA_PARALLEL', type=str, help='the mode of parallel.')
    parser.add_argument('--save_dir', default='/home/ma-user/work/output/', type=str, help='the save dir name.')
    parser.add_argument('--num_workers', default=8, type=int, help='the workers number.')
    parser.add_argument('--use_ckpt', default='/home/ma-user/work/ckpt/mae-base.ckpt', type=str,
                        help='the ckpt model path.')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam beta1.')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam beta1.')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='the weight decay.')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='the warmup epoch when finetune.')

    # eval
    parser.add_argument('--eval_engine', default='imagenet', type=str, help='the name of eval engine.')
    parser.add_argument('--eval_path', default='/home/ma-user/work/imagenet2012/val', type=str,
                        help='the path of eval dataset.')
    parser.add_argument('--eval_interval', default=1, type=int, help='the eval interval.')
    parser.add_argument('--eval_offset', default=100, type=int, help='the eval offset.')

    # dataset
    parser.add_argument('--dataset_name', default='imagenet', type=str, help='the name of train dataset.')

    # loss
    parser.add_argument('--use_label_smooth', default=1, type=int, help='whether to use label smooth.')
    parser.add_argument('--label_smooth_factor', default=0.1, type=float, help='the label smooth factor.')
    parser.add_argument('--loss_name', default='soft_ce', type=str, help='the name of loss.')
    parser.add_argument('--use_dynamic_loss_scale', default=False, type=bool, help='whether to use dynamic loss scale.')
    parser.add_argument('--loss_scale', default=1024, type=int, help='loss scale.')

    # tools
    parser.add_argument('--interpolation', default=Inter.BICUBIC, type=int, help='the type of interpolation.')
    parser.add_argument('--per_step_size', default=0, type=int, help='per step size.')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout.')
    parser.add_argument('--drop_path', default=0.1, type=float, help='drop path.')
    parser.add_argument('--use_ema', default=False, type=bool, help='whether to use ema.')
    parser.add_argument('--ema_decay', default=0.9999, type=float, help='ema decay.')
    parser.add_argument('--use_global_norm', default=True, type=bool, help='whether to use global norm.')
    parser.add_argument('--clip_gn_value', default=1.0, type=float, help='clip gn value.')

    # network
    parser.add_argument('--encoder_layers', default=12, type=int, help='the number of encoder layer.')
    parser.add_argument('--encoder_num_heads', default=12, type=int, help='the number of encoder head.')
    parser.add_argument('--encoder_dim', default=768, type=int, help='the number of encoder dimension.')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='the ratio between mlp and encoder dimension.')
    parser.add_argument('--image_size', default=224, type=int, help='the number of image size.')
    parser.add_argument('--batch_size', default=32, type=int, help='the number of patch size.')
    parser.add_argument('--patch_size', default=16, type=int, help='the random seed number.')
    parser.add_argument('--num_classes', default=1000, type=int, help='the number of classes.')
    parser.add_argument('--channels', default=3, type=int, help='the number of channels in input images.')

    return parser


def train_parse_args():
    """
    Parse parameters.

     Returns:
        parsed parameters.
    """
    # train
    parser = argparse.ArgumentParser(description='pretrain')
    parser = common_parse_args(parser)
    parser.add_argument('--save_ckpt_epochs', default=1, type=int, help='the number epoch of save ckpt when train.')
    parser.add_argument('--prefix', default='MaeFintuneViT-B', type=str, help='the path prefix when save model.')
    parser.add_argument('--epoch', default=800, type=float, help='the pretrain epoch number.')

    # dataset
    parser.add_argument('--data_path', default='/home/ma-user/work/imagenet2012', type=str,
                        help='the path of train dataset.')
    parser.add_argument('--img_ids', default='tot_ids.json', type=str, help='the image id json file.')

    # loss
    parser.add_argument('--norm_pixel_loss', default=True, type=bool, help='whether to use norm pixel loss.')

    # tools
    parser.add_argument('--base_lr', default=0.0000005, type=float, help='initial learning rate.')
    parser.add_argument('--start_learning_rate', default=0.0000005, type=float, help='the start of learning rate.')
    parser.add_argument('--end_learning_rate', default=0.0000000001, type=float, help='the end of learning rate.')
    parser.add_argument('--sink_mode', default=True, type=bool, help='sink mode.')

    # network
    parser.add_argument('--decoder_layers', default=8, type=int, help='the number of encoder layer.')
    parser.add_argument('--decoder_num_heads', default=16, type=int, help='the number of encoder head.')
    parser.add_argument('--decoder_dim', default=512, type=int, help='the number of encoder dimension.')
    parser.add_argument('--masking_ratio', default=0.75, type=float, help='the mask radio.')
    parser.add_argument('--cb_size', default=1, type=int, help='the clip size.')

    return parser.parse_args()


def finetune_parse_args():
    """
    Finetune parse parameters.

     Returns:
        parsed parameters.
    """
    # train
    parser = argparse.ArgumentParser(description='finetune')
    parser = common_parse_args(parser)
    parser.add_argument('--save_ckpt_epochs', default=1, type=int, help='the number epoch of save ckpt when finetune.')
    parser.add_argument('--prefix', default='MaeFintuneViT-B', type=str, help='the path prefix when save model.')
    parser.add_argument('--layer_decay', default=0.75, type=float, help='the layer decay.')
    parser.add_argument('--epoch', default=300, type=float, help='the finetune epoch number.')

    # dataset
    parser.add_argument('--dataset_path', default='/home/ma-user/work/imagenet2012/train', type=str,
                        help='the path of train dataset.')

    # loss
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='the label smoothing.')

    # tools
    parser.add_argument('--base_lr', default=0.00015, type=float, help='initial learning rate.')
    parser.add_argument('--start_learning_rate', default=0.00015, type=float, help='the start of learning rate.')
    parser.add_argument('--end_learning_rate', default=0.0000000001, type=float, help='the end of learning rate.')
    parser.add_argument('--auto_augment', default='rand-m9-mstd0.5-inc1', type=str,
                        help='number of epochs with the dynamic learning rate.')
    parser.add_argument('--re_prop', default=0.5, type=float, help='re prop.')
    parser.add_argument('--re_mode', default='pixel', type=str, help='re mode type.')
    parser.add_argument('--re_count', default=1, type=int, help='re count.')
    parser.add_argument('--crop_min', default=0.2, type=float, help='mix up.')
    parser.add_argument('--mixup', default=0.8, type=float, help='cutmix.')
    parser.add_argument('--cutmix', default=1.0, type=float, help='mix up prob.')
    parser.add_argument('--mixup_prob', default=1.0, type=float, help='mix up prob.')
    parser.add_argument('--switch_prob', default=0.5, type=float, help='switch prob.')
    parser.add_argument('--sink_mode', default=True, type=bool, help='data sink mode.')

    return parser.parse_args()


def eval_parse_args():
    """
    Eval parse parameters.

     Returns:
        parsed parameters.
    """
    parser = argparse.ArgumentParser(description='eval')
    parser = common_parse_args(parser)

    return parser.parse_args()


mae_train_config = train_parse_args()
mae_finetune_config = finetune_parse_args()
mae_eval_config = eval_parse_args()
