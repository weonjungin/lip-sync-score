# /home/jiweon/projects/lip-sync-score/src/lipsyncscore/models/modified/syncnet_temporal.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from .audio_encoder_tokens import AudioEncoderTokens
from .lip_encoder_tokens import LipEncoderTokens
from .temporal import build_temporal
from .pooling import AttnPool1D


class SyncNetTemporal(nn.Module):
    """
    Dual encoder + (optional) temporal module per modality.

    Pooling options:
      - "mean"   : average over time tokens
      - "center" : take the center time token
      - "attn"   : attention pooling over time tokens
    """
    def __init__(
        self,
        in_frames: int = 5,
        emb_dim: int = 256,
        temporal_cfg: Optional[dict] = None,
        pooling: str = "mean",
        lip_in_channels: int = 4,
    ):
        super().__init__()
        self.in_frames = int(in_frames)
        self.emb_dim = int(emb_dim)

        # token encoders
        self.lip_enc = LipEncoderTokens(in_channels=lip_in_channels, emb_dim=emb_dim)
        self.aud_enc = AudioEncoderTokens(emb_dim=emb_dim)

        # we apply temporal on raw token dim (256)
        self.token_dim = 256

        # Separate projections (token_dim -> emb_dim)
        self.lip_proj = nn.Linear(self.token_dim, self.emb_dim)
        self.aud_proj = nn.Linear(self.token_dim, self.emb_dim)

        temporal_cfg = temporal_cfg or {}
        self.temporal_lip = build_temporal(self.token_dim, temporal_cfg.get("lip", temporal_cfg))
        self.temporal_aud = build_temporal(self.token_dim, temporal_cfg.get("audio", temporal_cfg))

        pooling = (pooling or "mean").lower()
        assert pooling in ["mean", "center", "attn"], f"pooling must be mean|center|attn, got {pooling}"
        self.pooling = pooling

        if self.pooling == "attn":
            self.lip_pool = AttnPool1D(self.token_dim)
            self.aud_pool = AttnPool1D(self.token_dim)
        else:
            self.lip_pool = None
            self.aud_pool = None

    def forward_lip_tokens(self, lips: torch.Tensor) -> torch.Tensor:
        tokens = self.lip_enc.forward_tokens(lips)  # (B,N,256)
        if self.temporal_lip is not None:
            tokens = self.temporal_lip(tokens)      # (B,N,256)
        return tokens

    def forward_audio_tokens(self, mel: torch.Tensor) -> torch.Tensor:
        tokens = self.aud_enc.forward_tokens(mel)   # (B,T',256)
        if self.temporal_aud is not None:
            tokens = self.temporal_aud(tokens)      # (B,T',256)
        return tokens

    def _pool(self, tokens: torch.Tensor, which: str) -> torch.Tensor:
        """
        tokens: (B,T,C) -> (B,C)
        """
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be (B,T,C), got {tuple(tokens.shape)}")

        if self.pooling == "mean":
            return tokens.mean(dim=1)

        if self.pooling == "center":
            # pick center time index
            t = tokens.shape[1]
            if t <= 0:
                raise ValueError("tokens has zero-length time dimension")
            center_idx = t // 2
            return tokens[:, center_idx, :]

        # attn pooling
        if self.pooling == "attn":
            if which == "lip":
                if self.lip_pool is None:
                    raise RuntimeError("lip_pool is None but pooling='attn'")
                return self.lip_pool(tokens)
            else:
                if self.aud_pool is None:
                    raise RuntimeError("aud_pool is None but pooling='attn'")
                return self.aud_pool(tokens)

        raise RuntimeError(f"Unknown pooling: {self.pooling}")

    def forward_lip(self, lips: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_lip_tokens(lips)      # (B,N,256)
        pooled = self._pool(tokens, "lip")          # (B,256)
        out = self.lip_proj(pooled)                 # (B,D)
        return F.normalize(out, dim=1)

    def forward_audio(self, mel: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_audio_tokens(mel)     # (B,T',256)
        pooled = self._pool(tokens, "audio")        # (B,256)
        out = self.aud_proj(pooled)                 # (B,D)
        return F.normalize(out, dim=1)

    def forward(self, lips: torch.Tensor, mel: torch.Tensor):
        v = self.forward_lip(lips)
        a = self.forward_audio(mel)
        return v, a
