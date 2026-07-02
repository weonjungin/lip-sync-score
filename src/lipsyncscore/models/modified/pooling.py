# src/lipsyncscore/models/modified/pooling.py
import torch
import torch.nn as nn

class AttnPool1D(nn.Module):
    """
    Attention pooling over token sequence.

    Input:
      tokens: (B, T, C)
    Output:
      pooled: (B, C)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # (B,T,1)
        w = self.score(tokens)
        # (B,T)
        w = torch.softmax(w.squeeze(-1), dim=1)
        # (B,C)
        pooled = torch.sum(tokens * w.unsqueeze(-1), dim=1)
        return pooled
