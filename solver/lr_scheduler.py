# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,  # steps
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _get_lr(self, epoch):
        """兼容 CosineLRScheduler 的接口，用于 processor.py 日志输出"""
        # 计算指定 epoch 的学习率
        warmup_factor = 1
        if epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, epoch)
            for base_lr in self.base_lrs
        ]


class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear decay learning rate scheduler with optional warmup.

    Decreases learning rate linearly from base_lr to min_lr over max_iters epochs.
    Supports warmup phase at the beginning of training.

    Args:
        optimizer: Wrapped optimizer
        max_iters: Total number of epochs for training
        warmup_factor: Initial warmup factor (default: 1/3)
        warmup_iters: Number of warmup epochs (default: 0)
        warmup_method: Warmup method, 'constant' or 'linear' (default: 'linear')
        min_lr: Minimum learning rate after decay (default: 0.0)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(
            self,
            optimizer,
            max_iters,
            warmup_factor=1.0 / 3,
            warmup_iters=0,
            warmup_method="linear",
            min_lr=0.0,
            last_epoch=-1,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, "
                "got {}".format(warmup_method)
            )
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.min_lr = min_lr
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    def _warmup_factor_at(self, epoch):
        """Calculate warmup factor at given epoch"""
        if epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                return self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = epoch / float(self.warmup_iters)
                return self.warmup_factor * (1 - alpha) + alpha
        return 1.0

    def _decay_factor_at(self, epoch):
        """Calculate linear decay factor at given epoch"""
        if epoch <= self.warmup_iters:
            return 1.0
        # Linear decay from 1.0 to 0.0 over (max_iters - warmup_iters) epochs
        total_decay_iters = max(1, self.max_iters - self.warmup_iters)
        progress = (epoch - self.warmup_iters) / float(total_decay_iters)
        return max(1.0 - progress, 0.0)

    def get_lr(self):
        """Calculate current learning rate"""
        warmup = self._warmup_factor_at(self.last_epoch)
        decay = self._decay_factor_at(self.last_epoch)
        return [max(self.min_lr, base_lr * warmup * decay) for base_lr in self.base_lrs]

    def _get_lr(self, epoch):
        """Compatible interface for processor.py logging"""
        warmup = self._warmup_factor_at(epoch)
        decay = self._decay_factor_at(epoch)
        return [max(self.min_lr, base_lr * warmup * decay) for base_lr in self.base_lrs]
