# src/lipsyncscore/models/modified/fusion_crossattn.py
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttnFusion(nn.Module):
    """
    Minimal cross-attention fusion module.
    - Input: q tokens (B, Tq, D), kv tokens (B, Tkv, D)
    - Output: fused q tokens (B, Tq, D)

    Implemented using TransformerDecoderLayer(s) to get:
      self-attn on q + cross-attn(q <- kv) + FFN
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
        num_layers: int = 1,
        dim_feedforward: int | None = None,
        norm_first: bool = True,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=norm_first,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_key_padding_mask: torch.Tensor | None = None,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q:  (B, Tq, D)
            kv: (B, Tkv, D)
            q_key_padding_mask:  (B, Tq) bool, True for PAD
            kv_key_padding_mask: (B, Tkv) bool, True for PAD
        """
        x = q
        for layer in self.layers:
            x = layer(
                tgt=x,
                memory=kv,
                tgt_key_padding_mask=q_key_padding_mask,
                memory_key_padding_mask=kv_key_padding_mask,
            )
        return self.out_norm(x)
