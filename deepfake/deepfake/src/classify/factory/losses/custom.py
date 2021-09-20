import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class MixupBCELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        if type(y_true) == dict:
            # Training
            y_true1 = y_true['y_true1']
            y_true2 = y_true['y_true2']
            lam = y_true['lam']
            mix_loss1 = F.binary_cross_entropy_with_logits(y_pred, y_true1, reduction='none')
            mix_loss2 = F.binary_cross_entropy_with_logits(y_pred, y_true2, reduction='none')
            return (lam * mix_loss1 + (1. - lam) * mix_loss2).mean()
        else:
            # Validation
            return F.binary_cross_entropy_with_logits(y_pred, y_true)

