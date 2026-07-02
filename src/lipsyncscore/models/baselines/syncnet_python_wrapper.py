# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SYNCNET_ROOT = "/home/jiweon/projects/syncnet_python"
if SYNCNET_ROOT not in sys.path:
    sys.path.append(SYNCNET_ROOT)

from SyncNetModel import S  # noqa: E402


class SyncNetPythonWrapper(nn.Module):
    def __init__(self, ckpt_path: Optional[str] = None, load_pretrained: bool = True):
        super().__init__()
        self.model = S()

        if load_pretrained:
            if ckpt_path is None:
                raise ValueError("ckpt_path must be provided when load_pretrained=True")

            ckpt_path = str(Path(ckpt_path).expanduser())
            state = torch.load(ckpt_path, map_location="cpu")

            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            self.model.load_state_dict(state, strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        super().eval()
        self.model.eval()
        return self

    def _prepare_video(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert to [B, C, T, H, W] for syncnet_python.
        Accepts:
          - [B, T, H, W]
          - [B, T, C, H, W]
          - [B, C, T, H, W]
        """

        if x.ndim == 4:
            # [B, T, H, W] -> [B, T, 1, H, W]
            x = x.unsqueeze(2)

        if x.ndim != 5:
            raise ValueError(f"Video tensor must be 4D or 5D, got shape={tuple(x.shape)}")

        # [B, T, C, H, W] -> [B, C, T, H, W]
        if x.shape[1] > 1 and x.shape[2] in (1, 3):
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        # already [B, C, T, H, W]
        elif x.shape[1] in (1, 3):
            pass

        else:
            raise ValueError(
                "Could not infer video layout. "
                f"Expected [B,T,C,H,W] or [B,C,T,H,W], got {tuple(x.shape)}"
            )

        x = x.float()

        # grayscale -> fake RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)

        # resize each frame to 224x224 for original syncnet_python
        B, C, T, H, W = x.shape
        if (H, W) != (224, 224):
            x2 = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
            x2 = F.interpolate(x2, size=(224, 224), mode="bilinear", align_corners=False)
            x = x2.view(B, T, C, 224, 224).permute(0, 2, 1, 3, 4).contiguous()

        return x

    def _prepare_audio(self, a: torch.Tensor) -> torch.Tensor:
        if a.ndim == 3:
            a = a.unsqueeze(1)  # [B,F,T] -> [B,1,F,T]
        if a.ndim != 4:
            raise ValueError(f"Audio tensor must be 3D or 4D, got shape={tuple(a.shape)}")
        return a.float()

    def forward_video(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_video(x)
        return self.model.forward_lip(x)

    def forward_audio(self, a: torch.Tensor) -> torch.Tensor:
        a = self._prepare_audio(a)
        return self.model.forward_aud(a)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.forward_video(video)
        a = self.forward_audio(audio)
        return v, a
    
    def forward_lip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_video(x)

    def forward_aud(self, a: torch.Tensor) -> torch.Tensor:
        return self.forward_audio(a)