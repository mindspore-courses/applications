import math
from collections import Counter
import numpy as np

class _LinearWarmUp():
    def __init__(self, lr, warmup_epochs, steps_per_epoch, warmup_init_lr=0):
        self.base_lr = lr
        self.warmup_init_lr = warmup_init_lr
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)

    def get_warmup_steps(self):
        return self.warmup_steps

    def get_lr(self, current_step):
        lr_inc = (float(self.base_lr) - float(self.warmup_init_lr)) / float(self.warmup_steps)
        lr = float(self.warmup_init_lr) + lr_inc * current_step
        return lr
    
class CosineAnnealingLR():
    def __init__(self, lr, T_max, steps_per_epoch, max_epoch, warmup_epochs=0, eta_min=0):
        self.base_lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = int(max_epoch * steps_per_epoch)
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        current_lr = self.base_lr
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                cur_ep = i // self.steps_per_epoch
                if i % self.steps_per_epoch == 0 and i > 0:
                    current_lr = self.eta_min + \
                                 (self.base_lr - self.eta_min) * (1. + math.cos(math.pi*cur_ep / self.T_max)) / 2

                lr = current_lr
            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)
