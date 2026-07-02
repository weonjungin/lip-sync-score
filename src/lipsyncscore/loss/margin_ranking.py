import torch
import torch.nn as nn
import torch.nn.functional as F


class SyncNetMarginRankingLoss(nn.Module):
    """
    Margin Ranking Loss using cosine similarity

    목표:
    s(v, a_pos) > s(v, a_neg) + margin

    L = max(0, margin - s_pos + s_neg)
    """

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, v, a_pos, a_neg):

        # cosine similarity
        s_pos = F.cosine_similarity(v, a_pos)
        s_neg = F.cosine_similarity(v, a_neg)

        loss = F.relu(self.margin - s_pos + s_neg)

        return loss.mean(), {
            "s_pos": s_pos.mean().item(),
            "s_neg": s_neg.mean().item(),
        }