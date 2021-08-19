import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=2)
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot.float(), reduce=None)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=None)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

from torch.optim import lr_scheduler
import math

class SineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super(SineAnnealingLR, self).__init__(optimizer=optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch /self.T_max + math.pi)) / 2
            for base_lr in self.base_lrs]


"""
Try this if not working well : https://github.com/Tony-Y/pytorch_warmup#radam-warmup
it says : The user warning about calling lr_scheduler.step() before optimizer.step() may be ignored.
also recommand 
For PyTorch 1.4 or above, use an LR scheduler as the following:

        lr_scheduler.step(lr_scheduler.last_epoch+1)
"""