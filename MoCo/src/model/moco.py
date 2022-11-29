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
"""MoCo model."""

import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops

from model.resnet_util import ResNet18, BasicBlock


class ModelMoCo(nn.Cell):
    """MoCo model based on ResNet18."""
    def __init__(self, i=4096, m=0.01, t=0.1, symmetric=True):
        super(ModelMoCo, self).__init__()

        self.i = i
        self.m = m
        self.t = t
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = ResNet18(BasicBlock, [2, 2, 2, 2], 128)
        self.encoder_k = ResNet18(BasicBlock, [2, 2, 2, 2], 128)

        for param_q, param_k in zip(self.encoder_q.trainable_params(), self.encoder_k.trainable_params()):
            param_k = param_q.clone()
            param_k.requires_grad = False

        self.queue = mindspore.Parameter(ops.Zeros()((128, 4096), mindspore.float32), name="queue", requires_grad=False)
        self.queue = ops.L2Normalize(axis=0)(self.queue)
        self.queue_ptr = mindspore.Parameter(ops.Zeros()(1, mindspore.float32), name="queue_ptr", requires_grad=False)

    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.trainable_params(),
                                    self.encoder_k.trainable_params()):
            param_k.set_data(param_k.data * (1 - self.m) + param_q.data * self.m)

    def _dequeue_and_enqueue(self, keys):
        """encoding and decoding function."""
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.i

        self.queue_ptr[0] = ptr

    @staticmethod
    def _batch_shuffle_single_gpu(x):
        """batch shuffle is used for multi gpu simulation."""

        # random shuffle index
        n_x = Tensor([x.shape[0]], dtype=mindspore.int32)
        randperm = ops.Randperm(max_length=x.shape[0], pad=-1)
        idx_shuffle = randperm(n_x)
        n_2 = ops.Cast()(idx_shuffle, mindspore.float32)

        # index for restoring
        idx_unshuffle_2 = ops.Sort()(n_2)
        idx_unshuffle = idx_unshuffle_2[1]

        return x[idx_shuffle], idx_unshuffle

    @staticmethod
    def _batch_unshuffle_single_gpu(x, idx_unshuffle):
        """Undo batch shuffle is used for multi gpu simulation."""

        return x[idx_unshuffle]

    def infonce_loss(self, im_q, im_k):
        """InfoNCE loss function."""
        # compute query features
        q = self.encoder_q(im_q)
        q = ops.L2Normalize(axis=1)(q)

        # compute key features
        im_k_, idx_unshuffle = ModelMoCo._batch_shuffle_single_gpu(im_k)

        k = self.encoder_k(im_k_)
        k = ops.L2Normalize(axis=1)(k)

        # undo shuffle
        k = ModelMoCo._batch_unshuffle_single_gpu(k, idx_unshuffle)
        k = ops.stop_gradient(k)

        einsum0 = ops.ReduceSum()(q * k, -1)
        l_pos = ops.ExpandDims()(einsum0, -1)

        # negative logits: NxK
        l_neg = ops.MatMul()(q, self.queue)

        # logits: Nx(1+K)
        logits = ops.Concat(axis=1)((l_pos, l_neg))
        logits_n = ops.Cast()(logits, mindspore.float32)

        # apply temperature
        logits_x = logits_n / self.t

        # labels: positive key indicators
        labels_n = ops.Zeros()((logits.shape[0]), mindspore.int32)
        labels = ops.Cast()(labels_n, mindspore.int32)

        # Calculate the infonce loss
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')(logits_x, labels)
        k = ops.stop_gradient(k)
        loss = ops.stop_gradient(loss)

        return loss, k

    def construct(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:
            loss_12, k2 = self.infonce_loss(im1, im2)
            loss_21, k1 = self.infonce_loss(im2, im1)
            loss = loss_12 + loss_21
            k = ops.Concat(axis=0)(k1, k2)
        else:
            loss, k = self.infonce_loss(im1, im2)
        self._dequeue_and_enqueue(k)

        return loss
