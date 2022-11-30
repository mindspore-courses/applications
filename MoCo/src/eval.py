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
""" MoCo eval script."""

import numpy as np

import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.dataset


def test(net, memory_data_loader, test_data_loader, epoch, args):
    """
    test for MoCo

    Args:
        net : MoCo.encoder_q model
        memory_data_loader : memory_data
        test_data_loader: test_data
        epoch: epoch
        args: test args
    """
    net.set_train(False)
    classes = 10
    total_top1, total_num, step, feature_bank = 0.0, 0, 0, []
    x1 = np.random.normal(1, 1, (0))
    steps = test_data_loader.get_dataset_size()

    # generate feature bank
    for data1 in memory_data_loader.create_dict_iterator():
        feature = net(data1["image"])
        feature = ops.L2Normalize(axis=1)(feature)
        feature_bank.append(feature)
        x2 = data1["label"].asnumpy()
        x1 = np.concatenate([x1, x2], axis=0)

    feature_bank1 = ops.Concat(axis=0)(feature_bank)
    feature_bank2 = feature_bank1.T
    feature_labels = mindspore.Tensor(x1, mindspore.int32)

    # loop test data to predict the label by weighted knn search
    for data2 in test_data_loader.create_dict_iterator():
        feature = net(data2["image"])
        feature = ops.L2Normalize(axis=1)(feature)
        pred_labels = knn_predict(feature, feature_bank2, feature_labels, classes, args.knn_k, args.knn_t)
        cast = ops.Cast()
        total_num += data2["image"].shape[0]
        number = cast((pred_labels[:, 0] == data2["label"]), mindspore.float32)
        total_top1 += number.sum()
        if step % 20 == 0:

            print(f"Epoch: [{epoch} / {args.epochs}], "
                  f"step: [{step} / {steps}], "
                  f"Acc@1:{total_top1 / total_num * 100}%")
        step += 1

    return total_top1 / total_num * 100


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """
    knn_predict:compute cos similarity between each feature vector and feature bank ---> [B, N]
    knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
    implementation follows https://github.com/leftthomas/SimCLR
    """
    sim_matrix = ops.MatMul()(feature, feature_bank)

    topk = ops.TopK()
    sim_weight, sim_indices = topk(sim_matrix, knn_k)

    sim_labels = ops.GatherD()(ops.BroadcastTo((feature.shape[0], -1))(feature_labels), -1, sim_indices)
    sim_weight = ops.Exp()(sim_weight / knn_t)

    on_value, off_value = Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
    one_hot_label = ops.OneHot(axis=-1)(sim_labels.view(-1), 10, on_value, off_value)

    pred_scores = ops.ReduceSum()(one_hot_label.view(feature.shape[0], -1,
                                                     classes) * ops.ExpandDims()(sim_weight, -1), 1)
    sort = ops.Sort(axis=-1, descending=True)
    pred_labels = sort(pred_scores)[1]

    return pred_labels
