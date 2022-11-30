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
"""SplitBatchNorm is used to simulate multi gpu behavior."""

import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops


class SplitBatchNorm(nn.BatchNorm2d):
    """SplitBatchNorm:Simulate the behavior of BatchNorm's multiple gpus."""
    def __init__(self, num_features, num_splits=8, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def construct(self, inputs):
        """ build SplitBatchNorm network."""
        n, c, h, w = inputs.shape

        if self.training or not self.use_batch_statistics:
            moving_mean_split = np.tile(self.moving_mean, self.num_splits)
            moving_var_split = np.tile(self.moving_variance, self.num_splits)
            outcome = ops.BatchNorm(is_training=True, epsilon=1e-5, momentum=0.9,
                                    data_format="NCHW")(inputs.view(-1, c * self.num_splits, h, w),
                                                        np.tile(self.gamma, self.num_splits),
                                                        np.tile(self.beta, self.num_splits), moving_mean_split,
                                                        moving_var_split)[0]
            outcome = outcome.view(n, c, h, w)
            self.moving_mean.set_data(moving_mean_split.view(self.num_splits, c).mean(axis=0))
            self.moving_variance.set_data(moving_var_split.view(self.num_splits, c).mean(axis=0))
            return outcome

        return ops.BatchNorm(is_training=False, epsilon=1e-5, momentum=0.9)(
            inputs, self.moving_mean, self.moving_variance, self.gamma, self.beta)
