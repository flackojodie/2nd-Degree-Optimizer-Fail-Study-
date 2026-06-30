import autograd.numpy as np


class Constant:
    def __init__(self, base_learning_rate):
        self.base_lr = base_learning_rate

    def step(self):
        pass

    def __call__(self):
        return self.base_lr


class StepDecay:
    def __init__(self, base_learning_rate, step_down_frequency=50, gamma=0.1):
        self.epoch = 0
        self.base_lr = base_learning_rate
        self.step_freq = step_down_frequency
        self.gamma = gamma

    def step(self):
        self.epoch += 1

    def __call__(self):
        return self.base_lr * (self.gamma ** (self.epoch // self.step_freq))


class ExponentialDecay:
    def __init__(self, base_learning_rate, decay_rate=0.96):
        self.epoch = 0
        self.base_lr = base_learning_rate
        self.decay_rate = decay_rate

    def step(self):
        self.epoch += 1

    def __call__(self):
        return self.base_lr * (self.decay_rate ** self.epoch)


class CosineAnnealing:
    def __init__(self, base_learning_rate, minimum_learning_rate, total_steps):
        self.epoch = 0
        self.base_lr = base_learning_rate
        self.min_lr = minimum_learning_rate
        self.total_epochs = total_steps

    def step(self):
        self.epoch += 1

    def __call__(self):
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * self.epoch / self.total_epochs))


class CosineAnnealingWithWarmRestarts:
    def __init__(self, base_learning_rate, minimum_learning_rate, restart_after_n_steps=50,
                 reset_factor=2):
        self.epoch = 0
        self.t0 = restart_after_n_steps
        self.reset_multiple = reset_factor
        self.base_lr = base_learning_rate
        self.min_lr = minimum_learning_rate

    def step(self):
        self.epoch += 1

    def __call__(self):
        current_step = self.epoch
        reset_step = self.t0
        while current_step >= reset_step:
            current_step -= reset_step
            reset_step *= self.reset_multiple
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * current_step / reset_step))


class OneCycleCosine:
    def __init__(self, total_steps, minimum_learning_rate, maximum_learning_rate, final_learning_rate,
                fraction_spent_to_reach_peak=0.3):
        self.epoch = 0
        self.total_epochs = total_steps
        self.min_lr = minimum_learning_rate
        self.max_lr = maximum_learning_rate
        self.pct_start = fraction_spent_to_reach_peak
        self.final_lr = final_learning_rate

    def step(self):
        self.epoch += 1

    def __call__(self):
        peak_epoch = int(self.pct_start * self.total_epochs)
        if self.epoch <= peak_epoch:
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 - np.cos(np.pi * self.epoch / peak_epoch))
        else:
            lr = self.final_lr + 0.5 * (self.max_lr - self.min_lr) * (
                        1 + np.cos(np.pi * (self.epoch - peak_epoch) / (self.total_epochs - peak_epoch)))
        return lr
