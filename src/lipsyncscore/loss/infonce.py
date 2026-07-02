import torch
import torch.nn as nn
import torch.nn.functional as F


class SyncNetInfoNCELoss(nn.Module):
    """
    InfoNCE loss with in-batch negatives

    v_i 는 a_pos_i 와 매칭되어야 한다.
    batch 내 다른 audio는 negative가 된다.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, a_pos, a_neg=None):

        # normalize embeddings
        v = F.normalize(v, dim=1)
        a_pos = F.normalize(a_pos, dim=1)

        # similarity matrix
        logits = torch.matmul(v, a_pos.T) / self.temperature

        labels = torch.arange(v.size(0), device=v.device)

        loss = F.cross_entropy(logits, labels)

        # statistics
        s_pos = logits.diag().mean()
        s_neg = (logits.sum() - logits.diag().sum()) / (logits.numel() - logits.size(0))

        return loss, {
            "s_pos": s_pos.item(),
            "s_neg": s_neg.item(),
        }