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
"""RetinaFace Network."""

import math

from mindspore import context
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import functional, operations, composite
from mindspore.communication.management import get_group_size
from mindspore.parallel._auto_parallel_context import auto_parallel_context

from model.head import ClassHead, BboxHead, LandmarkHead
from model.mobilenet025 import MobileNetV1
from utils.initialize import init_kaiming_uniform


class ConvBNReLU(nn.SequentialCell):
    """Convolution,batch normalization and leaky ReLU with Kaiming uniform initialize."""

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, norm_layer, use_leaky=False,
                 leaky=0):
        weight_shape = (out_planes, in_planes, kernel_size, kernel_size)
        kaiming_weight, _ = init_kaiming_uniform(weight_shape, a=math.sqrt(5))
        activation = nn.LeakyReLU(alpha=leaky) if use_leaky else nn.ReLU()
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=groups,
                      has_bias=False, weight_init=kaiming_weight),
            norm_layer(out_planes),
            activation
        )


class ConvBN(nn.SequentialCell):
    """Convolution and batch normalization with Kaiming uniform initialize."""

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, norm_layer):
        weight_shape = (out_planes, in_planes, kernel_size, kernel_size)
        kaiming_weight, _ = init_kaiming_uniform(weight_shape, a=math.sqrt(5))

        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=groups,
                      has_bias=False, weight_init=kaiming_weight),
            norm_layer(out_planes),
        )


class SSH(nn.Cell):
    """
    SSH feature pyramid structure.

    Args:
        in_channel(int): Channel number of input tensor.
        out_channel(int): Channel number of output tensor.

    Raises:
        RuntimeError: If the number of output channels is not a multiple of 4.

    Inputs:
        - **x** (Tensor) - An input tensor, which shape is :math:`(B, in_channel, H, W)`.

    Outputs:
        Tensor of the output of the feature pyramid, which shape is :math:`(B, out_channel, H, W)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> from src.model.retinaface import SSH
        >>> zeros = ops.Zeros()
        >>> output = zeros((1, 8, 1, 1), mindspore.float32)
        >>> ssh = SSH(8, 4)
        >>> print(ssh(output))
        [[[[0.]]
          [[0.]]
          [[0.]]
          [[0.]]]]
    """

    def __init__(self, in_channel, out_channel, is_mobilenet=False):
        super(SSH, self).__init__()
        if out_channel % 4 != 0:
            raise RuntimeError('The number of output channels must be a multiple of 4.')
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1

        norm_layer = nn.BatchNorm2d
        self.conv3x3 = ConvBN(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1, groups=1,
                              norm_layer=norm_layer)

        self.conv5x5_1 = ConvBNReLU(in_channel, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                    norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)
        self.conv5x5_2 = ConvBN(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                norm_layer=norm_layer)

        self.conv7x7_2 = ConvBNReLU(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                    norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)
        self.conv7x7_3 = ConvBN(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                norm_layer=norm_layer)

        self.cat = ops.Concat(axis=1)
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass."""
        conv3x3 = self.conv3x3(x)

        conv5x5_1 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5_1)

        conv7x7_2 = self.conv7x7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)

        out = self.cat((conv3x3, conv5x5, conv7x7))
        out = self.relu(out)

        return out


class FPN(nn.Cell):
    """
    FPN feature pyramid structure.

    Args:
        in_channel(int): The input channel number of FPN. Default: None.
        out_channel(int): The output channel number of FPN. Default: None.

    Inputs:
        - **input1** (Tensor) - Input tensor of the top layer, which shape is :math:`(B, 512, H1, W1)`.
        - **input2** (Tensor) - Input tensor of the medium layer, which shape is :math:`(B, 1024, H2, W2)`.
        - **input3** (Tensor) - Input tensor of the bottom layer, which shape is :math:`(B, 2048, H3, W3)`.

    Outputs:
        Tuple of 3 Tensor, which contains output Tensor from top, middle and bottom layer.

        - **tensor1** (Tensor) - Tensor, its shape is :math:`(B, 256, H1, W1)`.
        - **tensor2** (Tensor) - Tensor, its shape is :math:`(B, 256, H2, W2)`.
        - **tensor3** (Tensor) - Tensor, its shape is :math:`(B, 256, H3, W3)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> from src.model.retinaface import FPN
        >>> fpn = FPN()
        >>> zeros = ops.Zeros()
        >>> f1 = zeros((1, 512, 4, 4), mindspore.float32)
        >>> f2 = zeros((1, 1024, 2, 2), mindspore.float32)
        >>> f3 = zeros((1, 2048, 1, 1), mindspore.float32)
        >>> f1, f2, f3 = fpn(f1, f2, f3)
    """

    def __init__(self, in_channel=None, out_channel=None, is_mobilenet=False):
        super(FPN, self).__init__()
        norm_layer = nn.BatchNorm2d
        leaky = 0
        if in_channel is None or out_channel is None:
            self.output1 = ConvBNReLU(512, 256, kernel_size=1, stride=1, padding=0, groups=1,
                                      norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)
            self.output2 = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0, groups=1,
                                      norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)
            self.output3 = ConvBNReLU(2048, 256, kernel_size=1, stride=1, padding=0, groups=1,
                                      norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)

            self.merge1 = ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1, groups=1,
                                     norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)
            self.merge2 = ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1, groups=1,
                                     norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)
        else:
            out_channels = out_channel
            if out_channels <= 64:
                leaky = 0.1
            self.output1 = ConvBNReLU(in_channel * 2, out_channel, kernel_size=1, stride=1,
                                      padding=0, groups=1, norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)
            self.output2 = ConvBNReLU(in_channel * 4, out_channel, kernel_size=1, stride=1,
                                      padding=0, groups=1, norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)
            self.output3 = ConvBNReLU(in_channel * 8, out_channel, kernel_size=1, stride=1,
                                      padding=0, groups=1, norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)

            self.merge1 = ConvBNReLU(out_channel, out_channel, kernel_size=3, stride=1, padding=1,
                                     groups=1,
                                     norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)
            self.merge2 = ConvBNReLU(out_channel, out_channel, kernel_size=3, stride=1, padding=1,
                                     groups=1,
                                     norm_layer=norm_layer, leaky=leaky, use_leaky=is_mobilenet)

    def construct(self, input1, input2, input3):
        """Forward pass."""
        output1 = self.output1(input1)
        output2 = self.output2(input2)
        output3 = self.output3(input3)

        up3 = ops.ResizeNearestNeighbor([ops.Shape()(output2)[2], ops.Shape()(output2)[3]])(output3)
        output2 = up3 + output2
        output2 = self.merge2(output2)

        up2 = ops.ResizeNearestNeighbor([ops.Shape()(output1)[2], ops.Shape()(output1)[3]])(output2)
        output1 = up2 + output1
        output1 = self.merge1(output1)

        return output1, output2, output3


class RetinaFace(nn.Cell):
    """
    RetinaFace network.

    Args:
        phase(string):Can be 'train' or else.If not train,classification result output will be after softmax.
            Default: 'train'.
        backbone(nn.Cell):One of resnet50 or mobilenet025,backbone of RetinaFace. Default: None.
        cfg(dict):A configuration. Default: None.

    Inputs:
        - **inputs** (Tensor) - images,which shape is :math:`(B,C,H,W)`.

    Outputs:
        Tuple of 3 Tensor, which represents bbox_regressions, classifications and ldm_regressions if train, else
        bbox_regressions, ops.Softmax(-1)(classifications), ldm_regressions.

        - **tensor1** (Tensor) - Tensor, its shape is :math:`(B, num_anchor, 4)`.
        - **tensor2** (Tensor) - Tensor, its shape is :math:`(B, num_anchor, 2)`.
        - **tensor3** (Tensor) - Tensor, its shape is :math:`(B, num_anchor, 10)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> from src.model.resnet50 import resnet50
        >>> from src.model.retinaface import RetinaFace
        >>> backbone = resnet50()
        >>> network = RetinaFace(phase='train', backbone=backbone)
        >>> zeros = ops.Zeros()
        >>> x = zeros((1, 3, 16, 16), mindspore.float32)
        >>> x = network(x)
    """

    def __init__(self, phase='train', backbone=None, cfg=None):

        super(RetinaFace, self).__init__()
        self.phase = phase
        self.base = backbone
        is_mobilenet = isinstance(backbone, MobileNetV1)
        if cfg is None:
            self.fpn = FPN(is_mobilenet=is_mobilenet)
            self.ssh1 = SSH(256, 256, is_mobilenet=is_mobilenet)
            self.ssh2 = SSH(256, 256, is_mobilenet=is_mobilenet)
            self.ssh3 = SSH(256, 256, is_mobilenet=is_mobilenet)
            self.class_head = self._make_class_head(fpn_num=3, inchannels=[256, 256, 256], anchor_num=[2, 2, 2])
            self.bbox_head = self._make_bbox_head(fpn_num=3, inchannels=[256, 256, 256], anchor_num=[2, 2, 2])
            self.landmark_head = self._make_landmark_head(fpn_num=3, inchannels=[256, 256, 256], anchor_num=[2, 2, 2])
        else:
            self.fpn = FPN(in_channel=cfg['in_channel'], out_channel=cfg['out_channel'], is_mobilenet=is_mobilenet)
            self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'], is_mobilenet=is_mobilenet)
            self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'], is_mobilenet=is_mobilenet)
            self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'], is_mobilenet=is_mobilenet)
            self.class_head = self._make_class_head(fpn_num=3, inchannels=[cfg['out_channel'], cfg['out_channel'],
                                                                           cfg['out_channel']], anchor_num=[2, 2, 2])
            self.bbox_head = self._make_bbox_head(fpn_num=3, inchannels=[cfg['out_channel'], cfg['out_channel'],
                                                                         cfg['out_channel']], anchor_num=[2, 2, 2])
            self.landmark_head = self._make_landmark_head(fpn_num=3, inchannels=[cfg['out_channel'],
                                                                                 cfg['out_channel'],
                                                                                 cfg['out_channel']],
                                                          anchor_num=[2, 2, 2])

        self.cat = ops.Concat(axis=1)

    def _make_class_head(self, fpn_num, inchannels, anchor_num):
        """Construct class head of network."""
        classhead = nn.CellList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels[i], anchor_num[i]))
        return classhead

    def _make_bbox_head(self, fpn_num, inchannels, anchor_num):
        """Construct box head of network."""
        bboxhead = nn.CellList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels[i], anchor_num[i]))
        return bboxhead

    def _make_landmark_head(self, fpn_num, inchannels, anchor_num):
        """Construct landmark head of network."""
        landmarkhead = nn.CellList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels[i], anchor_num[i]))
        return landmarkhead

    def construct(self, inputs):
        """Forward pass."""
        f1, f2, f3 = self.base(inputs)
        f1, f2, f3 = self.fpn(f1, f2, f3)
        f1 = self.ssh1(f1)
        f2 = self.ssh2(f2)
        f3 = self.ssh3(f3)
        features = [f1, f2, f3]
        bbox = ()
        for i, feature in enumerate(features):
            bbox = bbox + (self.bbox_head[i](feature),)
        bbox_regressions = self.cat(bbox)
        cls = ()
        for i, feature in enumerate(features):
            cls = cls + (self.class_head[i](feature),)
        classifications = self.cat(cls)
        landm = ()
        for i, feature in enumerate(features):
            landm = landm + (self.landmark_head[i](feature),)
        ldm_regressions = self.cat(landm)
        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, ops.Softmax(-1)(classifications), ldm_regressions)

        return output


class TrainingWrapperMobileNet(nn.Cell):
    """
    Wrap for training with mobilenet025 backbone.

    Args:
        network(nn.Cell): The RetinaFace network with loss calculation cell.
        optimizer(Optimizer): The optimizer for gradient descent.
        sens(float): Sensitivity, means gradient with respect to output. Default: 1.0.

    Inputs:
        - **args** (tuple) - Represents the tensor of input images and their box, conf and landmark ground truth.

    Outputs:
        Tensor, represents the loss after forward pass, its shape is :math:`(1)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> from src.model.mobilenet025 import MobileNetV1
        >>> from src.model.retinaface import RetinaFace
        >>> from src.model.retinaface import TrainingWrapperMobileNet
        >>> from src.model.loss_cell import RetinaFaceWithLossCell
        >>> from src.utils.multiboxloss import MultiBoxLoss
        >>> backbone = MobileNetV1()
        >>> network = RetinaFace(phase='train', backbone=backbone)
        >>> multibox_loss = MultiBoxLoss(10, 16800, 7, 8)
        >>> zeros = ops.Zeros()
        >>> net = RetinaFaceWithLossCell(net, multibox_loss, cfg)
        >>> opt = ms.nn.Adam(net.trainable_params(), 1e-3)
        >>> net = TrainingWrapperMobileNet(net, opt)
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapperMobileNet, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        class_list = [ms.ParallelMode.DATA_PARALLEL, ms.ParallelMode.HYBRID_PARALLEL]
        if self.parallel_mode in class_list:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = ms.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = ms.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


clip_grad = composite.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Args:
        clip_type(int): Can be 0,1 or else, means choose to use clip by value, clip by norm or no clip.
        clip_value(float): The value used for clip.
        grad(Tensor): Gradient tensor from forward pass result.

    Returns:
        Tensor, represents clipped gradients, its shape and data type is same as `grad`.
    """
    if clip_type not in (0, 1):
        return grad
    dt = functional.dtype(grad)
    if clip_type == 0:
        new_grad = composite.clip_by_value(grad, functional.cast(functional.tuple_to_array((-clip_value,)), dt),
                                           functional.cast(functional.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, functional.cast(functional.tuple_to_array((clip_value,)), dt))
    return new_grad


class TrainingWrapperResNet(nn.Cell):
    """
    Wrap for training with resnet backbone.

    Args:
        network(nn.Cell): The RetinaFace network with loss calculation cell.
        optimizer(Optimizer): The optimizer for gradient descent.
        sens(float): Sensitivity, means gradient with respect to output. Default: 1.0.

    Inputs:
        - **args** (tuple) - Represents the tensor of input images and their box, conf and landmark ground truth.

    Outputs:
        Tensor, represents the loss after forward pass, its shape is :math:`(1)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> from src.model.resnet50 import resnet50
        >>> from src.model.retinaface import RetinaFace
        >>> from src.model.retinaface import TrainingWrapperResNet
        >>> from src.model.loss_cell import RetinaFaceWithLossCell
        >>> from src.utils.multiboxloss import MultiBoxLoss
        >>> backbone = resnet50()
        >>> network = RetinaFace(phase='train', backbone=backbone)
        >>> multibox_loss = MultiBoxLoss(10, 16800, 7, 8)
        >>> zeros = ops.Zeros()
        >>> net = RetinaFaceWithLossCell(net, multibox_loss, cfg)
        >>> opt = ms.nn.Adam(net.trainable_params(), 1e-3)
        >>> net = TrainingWrapperResNet(net, opt)
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapperResNet, self).__init__(auto_prefix=False)
        self.gradient_clip_type = 1
        self.gradient_clip_value = 1.0
        self.network = network
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = composite.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        class_list = [ms.context.ParallelMode.DATA_PARALLEL, ms.context.ParallelMode.HYBRID_PARALLEL]
        if self.parallel_mode in class_list:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = ms.ops.HyperMap()

    def construct(self, *args):
        """construct"""
        weights = self.weights
        loss = self.network(*args)
        sens = operations.Fill()(operations.DType()(loss), operations.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        grads = self.hyper_map(functional.partial(clip_grad, self.gradient_clip_type, self.gradient_clip_value), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return functional.depend(loss, self.optimizer(grads))
