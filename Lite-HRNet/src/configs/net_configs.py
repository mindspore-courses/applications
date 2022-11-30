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

"""Configs of LiteHRNet"""

from backbone.LiteHRNetbackbone import LiteHRNet

def litehrnet_18_coco():

    """
    LiteHRNet 18 version for coco, which is suit for 17 keypoints.

    Returns:
        LiteHRNet. One version of LiteHRNet.
    """

    extra_lite_18_coco = dict(
        stem=dict(
            stem_channels=32,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('LITE', 'LITE', 'LITE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320))),
        with_head=True,
        final_cfg=dict(num_channels=40, joints=17, final_conv_kernel=1))
    backbone = dict(in_channels=3, extra=extra_lite_18_coco)
    return LiteHRNet(**backbone)

def litehrnet_30_coco():
    """
    LiteHRNet 30 version for coco, which is suit for 17 keypoints.

    Returns:
        LiteHRNet. One version of LiteHRNet.
    """

    extra_lite_30_coco = dict(
        stem=dict(
            stem_channels=32,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(3, 8, 3),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('LITE', 'LITE', 'LITE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320))),
        with_head=True,
        final_cfg=dict(num_channels=40, joints=17, final_conv_kernel=1))
    backbone = dict(in_channels=3, extra=extra_lite_30_coco)
    return LiteHRNet(**backbone)

def naive_litehrnet_18_coco():
    """
    LiteHRNet naive version for coco, which is suit for 17 keypoints.

    Returns:
        LiteHRNet. One version of LiteHRNet.
    """

    extra_naive_coco = dict(
        stem=dict(
            stem_channels=32,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('NAIVE', 'NAIVE', 'NAIVE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (30, 60),
                (30, 60, 120),
                (30, 60, 120, 240))),
        with_head=True,
        final_cfg=dict(num_channels=30, joints=17, final_conv_kernel=1))
    backbone = dict(in_channels=3, extra=extra_naive_coco)
    return LiteHRNet(**backbone)

def wider_naive_litehrnet_18_coco():
    """
    LiteHRNet wider naive version for coco, which is suit for 17 keypoints.

    Returns:
        LiteHRNet. One version of LiteHRNet.
    """

    extra_wider_naive_coco = dict(
        stem=dict(
            stem_channels=32,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('NAIVE', 'NAIVE', 'NAIVE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320))),
        with_head=True,
        final_cfg=dict(num_channels=40, joints=17, final_conv_kernel=1))
    backbone = dict(in_channels=3, extra=extra_wider_naive_coco)
    return LiteHRNet(**backbone)

def litehrnet_18_mpii():
    """
    LiteHRNet 18 version for mpii, which is suit for 16 keypoints.

    Returns:
        LiteHRNet. One version of LiteHRNet.
    """

    extra_lite_18_mpii = dict(
        stem=dict(
            stem_channels=32,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('LITE', 'LITE', 'LITE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320))),
        with_head=True,
        final_cfg=dict(num_channels=40, joints=16, final_conv_kernel=1))
    backbone = dict(in_channels=3, extra=extra_lite_18_mpii)
    return LiteHRNet(**backbone)

def litehrnet_30_mpii():
    """
    LiteHRNet 30 version for mpii, which is suit for 16 keypoints.

    Returns:
        LiteHRNet. One version of LiteHRNet.
    """

    extra_lite_30_mpii = dict(
        stem=dict(
            stem_channels=32,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(3, 8, 3),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('LITE', 'LITE', 'LITE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320))),
        with_head=True,
        final_cfg=dict(num_channels=40, joints=16, final_conv_kernel=1))
    backbone = dict(in_channels=3, extra=extra_lite_30_mpii)
    return LiteHRNet(**backbone)

def naive_litehrnet_18_mpii():
    """
    LiteHRNet naive version for mpii, which is suit for 16 keypoints.

    Returns:
        LiteHRNet. One version of LiteHRNet.
    """

    extra_naive_mpii = dict(
        stem=dict(
            stem_channels=32,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('NAIVE', 'NAIVE', 'NAIVE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (30, 60),
                (30, 60, 120),
                (30, 60, 120, 240))),
        with_head=True,
        final_cfg=dict(num_channels=30, joints=16, final_conv_kernel=1))
    backbone = dict(in_channels=3, extra=extra_naive_mpii)
    return LiteHRNet(**backbone)

def wider_naive_litehrnet_18_mpii():
    """
    LiteHRNet wider naive version for mpii, which is suit for 16 keypoints.

    Returns:
        LiteHRNet. One version of LiteHRNet.
    """

    extra_wider_naive_mpii = dict(
        stem=dict(
            stem_channels=32,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('NAIVE', 'NAIVE', 'NAIVE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320))),
        with_head=True,
        final_cfg=dict(num_channels=40, joints=16, final_conv_kernel=1))
    backbone = dict(in_channels=3, extra=extra_wider_naive_mpii)
    return LiteHRNet(**backbone)

def get_network(net_type, dataset_type):
    """
    Get the corresponding network

    Args:
        net_type (str): The type of Lite-HRNet
        dataset_type (str): COCO or MPII

    Returns:
        LiteHRNet. One version of LiteHRNet.
    """

    if dataset_type == "mpii":
        if "naive" in net_type:
            if "wider" in net_type:
                return wider_naive_litehrnet_18_mpii()

            return naive_litehrnet_18_mpii()

        if "30" in net_type:
            return litehrnet_30_mpii()

        return litehrnet_18_mpii()

    if "naive" in net_type:
        if "wider" in net_type:
            return wider_naive_litehrnet_18_coco()

        return naive_litehrnet_18_coco()

    if "30" in net_type:
        return litehrnet_30_coco()

    return litehrnet_18_coco()
