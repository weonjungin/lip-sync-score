# lipsyncscore/loss/contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SyncNetContrastiveLoss(nn.Module):
    """
    L = 0.5 * (lambda_pos * d_pos^2 + lambda_neg * max(0, margin - d_neg)^2)
    """
    def __init__(self, margin=1.0, lambda_pos=1.0, lambda_neg=1.0):
        super().__init__()
        self.margin = margin
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg

    def forward(self, v, a_pos, a_neg):
        d_pos = F.pairwise_distance(v, a_pos)
        d_neg = F.pairwise_distance(v, a_neg)

        loss = 0.5 * (
            self.lambda_pos * d_pos.pow(2) +
            self.lambda_neg * F.relu(self.margin - d_neg).pow(2)
        )
        return loss.mean(), {
            "d_pos": d_pos.mean().item(),
            "d_neg": d_neg.mean().item(),
        }