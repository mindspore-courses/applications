import mindspore.nn as nn
from mindspore.nn.loss.loss import LossBase
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.ops import operations as P

class PyramidEPE(LossBase):
    def __init__(self):
        super(PyramidEPE, self).__init__()
        self.scale_weights = [0.32, 0.08, 0.02, 0.01, 0.005]
        self.shape = P.Shape()

    @staticmethod
    def downsample2d_as(input, target_shape_tensor):
        _, _, h1, _ = P.Shape()(target_shape_tensor)
        _, _, h2, _ = P.Shape()(input)
        resize = h2 // h1
        return nn.AvgPool2d(1, stride=(resize, resize))(input) * (1.0 / resize)

    @staticmethod
    def elementwise_epe(input1, input2):
        return nn.Norm(axis=1, keep_dims=True)(input1 - input2)

    def construct(self, prediction, target, training=True):
        # if self.training:
        # target = target * 0.05
        # total_loss = 0
        # for i, pred in enumerate(prediction):
        #     _target = self.downsample2d_as(target, pred)
        #     total_loss += self.elementwise_epe(_target, pred).sum() * self.scale_weights[i]
        # return total_loss / P.Shape()(target)[0]
        # else:
            # loss = self.elementwise_epe(target, prediction)
            # total_loss = loss.mean()
            # return total_loss.sum()
        N, _, _, _ = self.shape(target)
        # div_flow trick
        if training:
            target = 0.05 * target
            # print(target.max(), target.min(), prediction[0].max(), prediction[0].min())
            total_loss = 0
            loss_ii = self.elementwise_epe(prediction[0], self.downsample2d_as(target, prediction[0])).sum()
            total_loss = total_loss + self.scale_weights[0] * loss_ii
            loss_ii = self.elementwise_epe(prediction[1], self.downsample2d_as(target, prediction[1])).sum()
            total_loss = total_loss + self.scale_weights[1] * loss_ii
            loss_ii = self.elementwise_epe(prediction[2], self.downsample2d_as(target, prediction[2])).sum()
            total_loss = total_loss + self.scale_weights[2] * loss_ii
            loss_ii = self.elementwise_epe(prediction[3], self.downsample2d_as(target, prediction[3])).sum()
            total_loss = total_loss + self.scale_weights[3] * loss_ii
            loss_ii = self.elementwise_epe(prediction[4], self.downsample2d_as(target, prediction[4])).sum()
            total_loss = total_loss + self.scale_weights[4] * loss_ii
            total_loss = total_loss / N
        
        else:
            epe = self.elementwise_epe(prediction, target)
            total_loss = epe.mean()
        
        return total_loss


class MultiStepLR(LearningRateSchedule):
    def __init__(self, lr, milestones, gamma):
        super().__init__()
        self.lr = lr
        self.milestones = milestones
        self.gamma = gamma
    
    def construct(self, global_step):
        lr = self.lr
        for milestone in self.milestones:
            if global_step >= milestone:
                lr *= self.gamma
        return lr