import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoderTokens(nn.Module):
    """
    SyncNet-style audio encoder that can return time tokens.

    Input:
      x: (B, 80, T) mel

    Output:
      - forward_tokens(x): (B, T', C=256)
      - forward(x):        (B, emb_dim) normalized
    """
    def __init__(self, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
        )

        # (B,256) -> (B,emb_dim)
        self.proj = nn.Linear(256, emb_dim)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
          tokens: (B, T', 256)
        """
        if x.dim() != 3:
            raise ValueError(f"Audio input must be (B,80,T), got {tuple(x.shape)}")

        x = x.unsqueeze(1)      # (B,1,80,T)
        h = self.conv(x)        # (B,256,F',T') where F'=80/4=20, T'=T/4

        # pool ONLY frequency axis (F'), keep time axis (T')
        # (B,256,F',T') -> (B,256,T')
        h = h.mean(dim=2)

        # (B,256,T') -> (B,T',256)
        tokens = h.transpose(1, 2).contiguous()
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default embedding: mean over time tokens -> proj -> normalize.
        """
        tokens = self.forward_tokens(x)     # (B,T',256)
        pooled = tokens.mean(dim=1)         # (B,256)
        out = self.proj(pooled)             # (B,emb_dim)
        out = F.normalize(out, dim=1)
        return out