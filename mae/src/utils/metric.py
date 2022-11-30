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
# ==============================================================================
"""
Metric is used to evaluate the quality of the model.
"""

import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.communication.management import GlobalComm


class ClassifyCorrectWithCache(nn.Cell):
    """
    Calculate the correct total number of classifications on the validation set.

    Args:
        network (net): Using network.
        eval_dataset (str): Eval dataset.
    """

    def __init__(self, network, eval_dataset):
        super(ClassifyCorrectWithCache, self).__init__(auto_prefix=False)
        self._network = network
        self.argmax = P.Argmax()
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()

        # Statute the Tensor data of all devices in the communication group using the specified method,
        # and all devices will get the same result
        # op: the specific operation of the statute, such as "sum", "max", and "min". Default value: ReduceOp.SUM.
        # group: The communication group to work with. Default value: "GlobalComm.WORLD_COMM_GROUP"
        # (i.e. "hccl_world_group" for Ascend platform, "nccl_world_group" for GPU platform) nccl_world_group"
        # for Ascend platform and " nccl_world_group" for GPU platform).
        self.allreduce = P.AllReduce(P.ReduceOp.SUM, GlobalComm.WORLD_COMM_GROUP)
        self.assign_add = P.AssignAdd()
        self.assign = P.Assign()
        self._correct_num = Parameter(Tensor(0.0, mstype.float32), name="correct_num",
                                      requires_grad=False)
        pdata = []
        plabel = []
        step_num = 0
        for batch in eval_dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
            # Create dictionary iterators for the validation dataset and add them to pdata and plabel respectively
            pdata.append(batch["image"])
            plabel.append(batch["label"])
            step_num = step_num + 1
        pdata = Tensor(np.array(pdata), mstype.float32)
        plabel = Tensor(np.array(plabel), mstype.int32)
        self._data = Parameter(pdata, name="pdata", requires_grad=False)
        self._label = Parameter(plabel, name="plabel", requires_grad=False)
        self._step_num = Tensor(step_num, mstype.int32)

    def construct(self, index):
        """
        Build correct number.

        Args:
            index (int): Specified index value.

        Returns:
            Float, the total number of correct predictions.
        """
        self._correct_num = 0
        for data, label in zip(self._data, self._label):
            # Network output results
            outputs = self._network(data)

            # Predicted results y_pred
            y_pred = self.argmax(outputs)
            y_pred = self.cast(y_pred, mstype.int32)

            # Determine if the prediction is correct
            y_correct = self.equal(y_pred, label)
            y_correct = self.cast(y_correct, mstype.float32)
            y_correct_sum = self.reduce_sum(y_correct)
            self._correct_num += y_correct_sum
            index = index + 1

        # Statute operation on Tensor data of all devices in the communication group
        # to get the total number of correct predictions
        total_correct = self.allreduce(self._correct_num)
        return total_correct


class ClassifyCorrectCell(nn.Cell):
    """
    Calculate the correct classification rate on the validation set (without using caching).

    Args:
        network (net): Using network.
    """

    def __init__(self, network):
        super(ClassifyCorrectCell, self).__init__(auto_prefix=False)
        self._network = network
        self.argmax = P.Argmax()
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()

    def construct(self, data, label):
        """
        Build correct number.

        Args:
            data (dataset): Data used to calculate classification accuracy.
            label (dict): True label value of data.

        Returns:
            Float, the total number of correct predictions.
        """
        outputs = self._network(data)
        y_pred = self.argmax(outputs)
        y_pred = self.cast(y_pred, mstype.int32)
        y_correct = self.equal(y_pred, label)
        y_correct = self.cast(y_correct, mstype.float32)
        y_correct = self.reduce_sum(y_correct)
        total_correct = y_correct
        return total_correct


class DistAccuracy(nn.Metric):
    """
    Calculate the dist correct classification rate.

    Args:
        batch_size (int): Number of samples selected for a training.
        device_num (int): Number of device.
    """

    def __init__(self, batch_size, device_num):
        super(DistAccuracy, self).__init__()
        self.clear()
        self.batch_size = batch_size
        self.device_num = device_num
        if self.batch_size == 0 or self.device_num == 0:
            raise RuntimeError('Accuracy can not be calculated, because the number of batch_size or device_num is 0.')

    def clear(self):
        # Initialize the number of correct predictions, and the total number of correct predictions
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        """
        Update correct num and total num.

        Args:
            inputs (int): Distributed accuracy will be converted from Tensor to numpy.
        """
        if len(inputs) != 1:
            raise ValueError('Distribute accuracy needs 1 input (y_correct), but got {}'.format(len(inputs)))
        y_correct = self._convert_data(inputs[0])
        self._correct_num += y_correct
        self._total_num += self.batch_size * self.device_num

    def eval(self):
        """
        Calculation accuracy.

        Returns:
            Float, the total number of correct predictions.
        """
        return self._correct_num / self._total_num
