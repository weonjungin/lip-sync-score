"""
syncnet_progressive.py

Progressive Local-to-Global SyncNet with Cross-Attention.

구조:
    lips  → lip_enc  → F_lip
    jaw   → jaw_enc  → F_jaw  → CrossAttn(Q=F_jaw,  KV=F_lip)      → F_jaw_out
    face  → face_enc → F_face → CrossAttn(Q=F_face, KV=F_jaw_out)  → F_face_out

fusion_mode:
    "last"   : F_face_out만 사용          → pool → proj → normalize
    "concat" : F_lip + F_jaw_out + F_face_out concat → pool → proj → normalize
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .lip_encoder_tokens import LipEncoderTokens
from .audio_encoder_tokens import AudioEncoderTokens
from .temporal import build_temporal
from .pooling import AttnPool1D


class HierarchicalCrossAttn(nn.Module):
    """
    Single-layer cross-attention.
    Query: local feature (jaw or face)
    Key/Value: context from previous level (lip or jaw_out)

    Input:
        query:   (B, N, C)
        context: (B, N, C)
    Output:
        out: (B, N, C)
    """
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query:   (B, N, C)
        context: (B, N, C)
        returns: (B, N, C)
        """
        B, N, C = query.shape

        # pre-norm
        q = self.norm_q(query)
        c = self.norm_c(context)

        Q = self.q_proj(q)   # (B, N, C)
        K = self.k_proj(c)
        V = self.v_proj(c)

        # reshape to multi-head
        def split_heads(x):
            # (B, N, C) -> (B, n_heads, N, head_dim)
            return x.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, n_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, n_heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        out = self.out_proj(out)

        # residual
        query = query + out
        query = self.norm_out(query)

        # FFN + residual
        query = query + self.ffn(self.norm_ffn(query))

        return query  # (B, N, C)


class SyncNetProgressive(nn.Module):
    """
    Progressive Local-to-Global SyncNet.

    Args:
        in_frames      : 프레임 수 (N)
        emb_dim        : 최종 임베딩 차원
        token_dim      : 인코더 내부 토큰 차원 (기본 256)
        n_heads        : cross-attention 헤드 수
        dropout        : dropout
        temporal_cfg   : temporal module 설정 (build_temporal 참고)
        pooling        : "mean" | "center" | "attn"
        fusion_mode    : "last" | "concat"
            "last"   → F_face_out만 pool
            "concat" → [F_lip, F_jaw_out, F_face_out] concat 후 pool
        lip_in_channels: lip latent 채널 수 (VAE latent = 4)
    """
    def __init__(
        self,
        in_frames: int = 16,
        emb_dim: int = 256,
        token_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.0,
        temporal_cfg: Optional[dict] = None,
        pooling: str = "mean",
        fusion_mode: str = "last",
        lip_in_channels: int = 4,
    ):
        super().__init__()
        self.in_frames = in_frames
        self.emb_dim = emb_dim
        self.token_dim = token_dim
        self.fusion_mode = fusion_mode.lower()
        assert self.fusion_mode in ("last", "concat"), \
            f"fusion_mode must be 'last' or 'concat', got {fusion_mode}"

        # ── 세 브랜치 인코더 (구조 동일, 가중치 독립) ──
        self.lip_enc  = LipEncoderTokens(in_channels=lip_in_channels, emb_dim=token_dim)
        self.jaw_enc  = LipEncoderTokens(in_channels=lip_in_channels, emb_dim=token_dim)
        self.face_enc = LipEncoderTokens(in_channels=lip_in_channels, emb_dim=token_dim)

        # ── 오디오 인코더 ──
        self.aud_enc = AudioEncoderTokens(emb_dim=token_dim)

        # ── Temporal modules (브랜치별 독립) ──
        temporal_cfg = temporal_cfg or {}
        self.temporal_lip  = build_temporal(token_dim, temporal_cfg.get("lip",   temporal_cfg))
        self.temporal_jaw  = build_temporal(token_dim, temporal_cfg.get("jaw",   temporal_cfg))
        self.temporal_face = build_temporal(token_dim, temporal_cfg.get("face",  temporal_cfg))
        self.temporal_aud  = build_temporal(token_dim, temporal_cfg.get("audio", temporal_cfg))

        # ── Cross-Attention 계층 ──
        # Level 1: jaw  참고 lip
        self.cross_attn_jaw  = HierarchicalCrossAttn(token_dim, n_heads=n_heads, dropout=dropout)
        # Level 2: face 참고 jaw_out
        self.cross_attn_face = HierarchicalCrossAttn(token_dim, n_heads=n_heads, dropout=dropout)

        # ── Pooling ──
        pooling = (pooling or "mean").lower()
        assert pooling in ("mean", "center", "attn"), f"pooling must be mean|center|attn, got {pooling}"
        self.pooling = pooling

        if self.pooling == "attn":
            self.lip_pool  = AttnPool1D(token_dim)
            self.aud_pool  = AttnPool1D(token_dim)
        else:
            self.lip_pool = None
            self.aud_pool = None

        # ── Projection: token_dim -> emb_dim ──
        # fusion_mode에 따라 proj 입력 차원이 달라짐
        if self.fusion_mode == "concat":
            self.lip_proj = nn.Linear(token_dim * 3, emb_dim)  # [F_lip, F_jaw_out, F_face_out]
        else:  # "last"
            self.lip_proj = nn.Linear(token_dim, emb_dim)      # F_face_out만

        self.aud_proj = nn.Linear(token_dim, emb_dim)

    # ─────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────

    def _pool_tokens(self, tokens: torch.Tensor, pool_module) -> torch.Tensor:
        """(B, T, C) -> (B, C)"""
        if self.pooling == "mean":
            return tokens.mean(dim=1)
        if self.pooling == "center":
            t = tokens.shape[1]
            return tokens[:, t // 2, :]
        if self.pooling == "attn":
            return pool_module(tokens)
        raise RuntimeError(f"Unknown pooling: {self.pooling}")

    def _encode_branch(self, enc, temporal, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C, H, W)
        returns: (B, N, token_dim)
        """
        tokens = enc.forward_tokens(x)       # (B, N, token_dim)
        if temporal is not None:
            tokens = temporal(tokens)         # (B, N, token_dim)
        return tokens

    # ─────────────────────────────────────────────────
    # Forward: lip embedding (progressive local-to-global)
    # ─────────────────────────────────────────────────

    def forward_lip(
        self,
        lips:  torch.Tensor,   # (B, N, 4, 12, 12)
        jaw:   torch.Tensor,   # (B, N, 4, 12, 12)
        face:  torch.Tensor,   # (B, N, 4, 12, 12)
    ) -> torch.Tensor:
        """
        Returns:
            emb: (B, emb_dim) L2-normalized
        """
        # ── Stage 1: lip encoding ──
        F_lip = self._encode_branch(self.lip_enc, self.temporal_lip, lips)
        # F_lip: (B, N, token_dim)

        # ── Stage 2: jaw encoding + cross-attn with lip ──
        F_jaw = self._encode_branch(self.jaw_enc, self.temporal_jaw, jaw)
        F_jaw_out = self.cross_attn_jaw(query=F_jaw, context=F_lip)
        # F_jaw_out: (B, N, token_dim)

        # ── Stage 3: face encoding + cross-attn with jaw_out ──
        F_face = self._encode_branch(self.face_enc, self.temporal_face, face)
        F_face_out = self.cross_attn_face(query=F_face, context=F_jaw_out)
        # F_face_out: (B, N, token_dim)

        # ── Fusion ──
        if self.fusion_mode == "last":
            # F_face_out만 사용 (lip 정보가 이미 cross-attn으로 전파됨)
            fused = F_face_out                                    # (B, N, token_dim)
            pooled = self._pool_tokens(fused, self.lip_pool)      # (B, token_dim)

        else:  # "concat"
            # 세 브랜치 모두 pool 후 concat
            p_lip  = self._pool_tokens(F_lip,      self.lip_pool)  # (B, token_dim)
            p_jaw  = self._pool_tokens(F_jaw_out,  self.lip_pool)  # (B, token_dim)
            p_face = self._pool_tokens(F_face_out, self.lip_pool)  # (B, token_dim)
            pooled = torch.cat([p_lip, p_jaw, p_face], dim=-1)     # (B, token_dim*3)

        out = self.lip_proj(pooled)          # (B, emb_dim)
        return F.normalize(out, dim=1)

    # ─────────────────────────────────────────────────
    # Forward: audio embedding
    # ─────────────────────────────────────────────────

    def forward_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, 80, T)
        Returns: (B, emb_dim) L2-normalized
        """
        tokens = self.aud_enc.forward_tokens(mel)   # (B, T', token_dim)
        if self.temporal_aud is not None:
            tokens = self.temporal_aud(tokens)
        pooled = self._pool_tokens(tokens, self.aud_pool)  # (B, token_dim)
        out = self.aud_proj(pooled)                        # (B, emb_dim)
        return F.normalize(out, dim=1)

    # ─────────────────────────────────────────────────
    # Combined forward
    # ─────────────────────────────────────────────────

    def forward(
        self,
        lips:    torch.Tensor,
        jaw:     torch.Tensor,
        face:    torch.Tensor,
        mel:     torch.Tensor,
    ):
        v = self.forward_lip(lips, jaw, face)
        a = self.forward_audio(mel)
        return v, a