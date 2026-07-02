import torch
import torch.nn as nn


class TemporalTCN(nn.Module):
    """
    Lightweight temporal module (TCN) over tokens.

    Input/Output:
      tokens: (B, T, C) -> (B, T, C)
    """
    def __init__(self, dim: int, num_layers: int = 2, kernel_size: int = 3, dilation_base: int = 2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            pad = ((kernel_size - 1) * dilation) // 2
            layers += [
                nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad, dilation=dilation),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(dim),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B,C,T)
        x = tokens.transpose(1, 2)
        x = self.net(x)
        # (B,C,T) -> (B,T,C)
        x = x.transpose(1, 2).contiguous()
        return x


class TemporalGRU(nn.Module):
    """
    GRU temporal module.

    Input/Output:
      tokens: (B,T,C) -> (B,T,C)
    """
    def __init__(self, dim: int, hidden_dim: int = None, num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(tokens)   # (B,T,out_dim)
        y = self.proj(y)          # (B,T,dim)
        return y


def build_temporal(dim: int, cfg: dict):
    """
    cfg example:
      {"type":"tcn", "num_layers":2, "kernel_size":3, "dilation_base":2}
      {"type":"gru", "num_layers":1, "bidirectional":True}
      {"type":"none"}
    """
    if cfg is None:
        return None
    t = (cfg.get("type", "none") or "none").lower()
    if t == "none":
        return None
    if t == "tcn":
        return TemporalTCN(
            dim=dim,
            num_layers=int(cfg.get("num_layers", 2)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            dilation_base=int(cfg.get("dilation_base", 2)),
        )
    if t == "gru":
        return TemporalGRU(
            dim=dim,
            hidden_dim=cfg.get("hidden_dim", None),
            num_layers=int(cfg.get("num_layers", 1)),
            bidirectional=bool(cfg.get("bidirectional", True)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    raise ValueError(f"Unknown temporal type: {t}")