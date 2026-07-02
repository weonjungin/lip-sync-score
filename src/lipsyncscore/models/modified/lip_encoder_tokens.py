import torch
import torch.nn as nn
import torch.nn.functional as F


class LipEncoderTokens(nn.Module):
    """
    Lip encoder that returns per-frame tokens.

    Input:
      lips: (B, N, C, H, W) VAE latent (C=4, H=12, W=12)

    Output:
      - forward_tokens(lips): (B, N, 256)
      - forward(lips):        (B, emb_dim) normalized
    """
    def __init__(self, in_channels=4, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim

        self.frame_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 12 → 6

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # MaxPool 제거 — 6→3 너무 작아짐

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, emb_dim)

    def forward_tokens(self, lips: torch.Tensor) -> torch.Tensor:
        """
        Returns:
          tokens: (B, N, 256)
        """
        if lips.dim() != 5:
            raise ValueError(f"Lip input must be (B,N,C,H,W), got {tuple(lips.shape)}")

        B, N, C, H, W = lips.shape
        x = lips.reshape(B * N, C, H, W)     # (B*N, 4, 12, 12)

        h = self.frame_cnn(x)                # (B*N, 256, h', w')
        h = self.spatial_pool(h).flatten(1)  # (B*N, 256)

        tokens = h.view(B, N, 256).contiguous()  # (B, N, 256)
        return tokens

    def forward(self, lips: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(lips)  # (B, N, 256)
        pooled = tokens.mean(dim=1)         # (B, 256)
        out = self.proj(pooled)             # (B, emb_dim)
        out = F.normalize(out, dim=1)
        return out