import mindspore.ops as P
import mindspore.common.dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore import Tensor

class Noam(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps, learning_rate=1.0):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.pow = P.Pow()
        self.min = P.Minimum()
        self.cast = P.Cast()
        self.const0 = Tensor(-0.5, mstype.float32)
        self.const1 = Tensor(-1.5, mstype.float32)

    def construct(self, global_step):
        p = self.cast(self.min(
            self.pow(global_step, self.const0),
            self.pow(self.warmup_steps, self.const1) * global_step),
            mstype.float32)
        return self.learning_rate * self.pow(self.d_model, self.const0) * p
