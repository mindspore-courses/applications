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
"""MaskRcnn training network wrapper."""

import time

import numpy as np
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import ParameterTuple
from mindspore.train.callback import Callback
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from utils.config import config

TIME_STAMP_INIT = False
TIME_STAMP_FIRST = 0

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Args:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Returns:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    mf_cast = F.cast(F.tuple_to_array((-clip_value,)), dt)
    pf_cast = F.cast(F.tuple_to_array((clip_value,)), dt)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, mf_cast, pf_cast)
    else:
        new_grad = nn.ClipByNorm()(grad, pf_cast)
    return F.cast(new_grad, dt)


class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.count = 0
        self.loss_sum = 0
        self.rank_id = rank_id

        global TIME_STAMP_INIT, TIME_STAMP_FIRST
        if not TIME_STAMP_INIT:
            TIME_STAMP_FIRST = time.time()
            TIME_STAMP_INIT = True

    def step_end(self, run_context):
        """set the end of step"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        self.count += 1
        self.loss_sum += float(loss)

        if self.count >= 1:
            global TIME_STAMP_FIRST
            time_stamp_current = time.time()
            total_loss = self.loss_sum/self.count

            print("%lu epoch: %s step: %s total_loss: %.5f" %
                  (time_stamp_current - TIME_STAMP_FIRST,
                   cb_params.cur_epoch_num, cur_step_in_epoch, total_loss))
            loss_file = open("./loss_{}.log".format(self.rank_id), "a+")
            loss_file.write("%lu epoch: %s step: %s total_loss: %.5f" %
                            (time_stamp_current - TIME_STAMP_FIRST,
                             cb_params.cur_epoch_num, cur_step_in_epoch, total_loss))
            loss_file.write("\n")
            loss_file.close()

            self.count = 0
            self.loss_sum = 0


class LossNet(nn.Cell):
    """MaskRcnn loss sum"""
    def construct(self, x1, x2):
        return x1 + x2


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.

    Inputs:
        - **x** (Tensor) - Input variant.
        - **img_shape** (Tensor) - Img shape.
        - **gt_bboxe** (Tensor) - Ground truth bounding boxes.
        - **gt_label** (Tensor) - Ground truth labels.
        - **gt_num** (int) - The number of ground truth.
        - **gt_mask** (Tensor) - Ground truth mask.

    Outputs:
        Loss network, Cell

    Support Platform:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> from src.utils.config import config
        >>> from src.model.mask_rcnn_r50 import MaskRcnnResnet50
        >>> net = MaskRcnnMobilenetV1(config=config)
        >>> loss = LossNet()
        >>> net_with_loss = WithLossCell(net, loss)
    """
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num, gt_mask):
        loss1, loss2, _, _, _, _, _ = \
            self._backbone(x, img_shape, gt_bboxe, gt_label, gt_num, gt_mask)
        return self._loss_fn(loss1, loss2)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone

class TrainOneStepCell(nn.Cell):
    """
    Network training package class.

    Append an optimizer to the training network
    after that the construct function.
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        reduce_flag (bool): The reduce flag. Default: False.
        mean (bool): Allreduce method. Default: False.
        degree (int): Device number. Default: None.

    Inputs:
        - **x** (Tensor) - Input variant.
        - **img_shape** (Tensor) - Img shape.
        - **gt_bboxe** (Tensor) - Ground truth bounding boxes.
        - **gt_label** (Tensor) - Ground truth labels.
        - **gt_num** (int) - The number of ground truth.
        - **gt_mask** (Tensor) - Ground truth mask.

    Outputs:
        Float, loss result.

    Support Platform:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> from src.utils.config import config
        >>> from src.model.mask_rcnn_r50 import MaskRcnnResnet50
        >>> net = MaskRcnnResnet50(config=config)
        >>> loss = LossNet()
        >>> net_with_loss = WithLossCell(net, loss)
        >>> lr = Tensor(dynamic_lr(config, rank_size=1, start_steps=0), mstype.float32)
        >>> opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.91,
        ...                weight_decay=1e-4, loss_scale=1)
        >>> net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)
    """
    def __init__(self, network, optimizer, sens=1.0, reduce_flag=False, mean=True, degree=None):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        if config.device_target == "Ascend":
            self.sens = Tensor((np.ones((1,)) * sens).astype(np.float16))
        else:
            self.sens = Tensor((np.ones((1,)) * sens).astype(np.float32))
        self.reduce_flag = reduce_flag
        self.hyper_map = C.HyperMap()
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num, gt_mask):
        """Construct Network training package class."""
        weights = self.weights
        loss = self.network(x, img_shape, gt_bboxe, gt_label, gt_num, gt_mask)
        grads = self.grad(self.network, weights)(x, img_shape, gt_bboxe, gt_label, gt_num, gt_mask, self.sens)
        if self.reduce_flag:
            grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        self.optimizer(grads)
        return loss
