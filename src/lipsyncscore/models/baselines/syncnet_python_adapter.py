# src/lipsyncscore/models/baselines/syncnet_python_adapter.py
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

class SyncNetPythonAdapter(nn.Module):
    """
    Adapter to use syncnet_python's network inside lip-sync-score eval loop.
    It exposes:
      - forward_lip(lips): (B, N, H, W) float in [0,1] -> (B, D)
      - forward_audio(mel): (B, 80, T) float -> (B, D)
    """

    def __init__(self, syncnet_python_root: str, model_path: str, device: str = "cpu"):
        super().__init__()
        self.root = Path(syncnet_python_root).resolve()
        self.model_path = str(Path(model_path).resolve())
        self.device = device

        # allow importing syncnet_python modules
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        # ---- IMPORTANT ----
        # 아래 import 경로/클래스명은 syncnet_python fork마다 다를 수 있음.
        # 너 repo에서 "loadParameters"가 로드하는 네트워크 클래스가 어디인지 보고 맞춰야 함.
        #
        # 보통은 SyncNetModel.py 안에 SyncNetModel / S / SyncNet 같은 클래스가 있고,
        # audio/lip forward가 각각 있는 경우가 많아.
        #
        # 가장 쉬운 방법:
        #   syncnet_python에서 SyncNetInstance.py 열고,
        #   self.__S__ = ??? (모델 생성) 부분의 클래스명을 그대로 import 해서 여기에 넣기.

        # 예시 1) SyncNetModel.py에 SyncNetModel 클래스가 있는 경우:
        # from SyncNetModel import SyncNetModel
        # self.net = SyncNetModel().to(device)

        # 예시 2) SyncNetModel.py에 S 라는 클래스가 있는 경우:
        # from SyncNetModel import S
        # self.net = S().to(device)

        # ---- 너의 repo에 맞게 아래 2줄을 수정해야 함 ----
        from SyncNetModel import S  # <-- 여기 (클래스명/모듈명) 맞추기
        self.net = S().to(device)

        self._load_weights()

        self.net.eval()

    def _load_weights(self):
        # syncnet_python의 .model 포맷이 torch.load state_dict인 경우가 많음
        ckpt = torch.load(self.model_path, map_location="cpu")

        # fork에 따라 ckpt 구조 다름: state_dict 바로일 수도 dict일 수도 있음
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        # 키 prefix 정리 필요할 수 있음 (module. 제거 등)
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v

        self.net.load_state_dict(new_state, strict=False)

    @torch.no_grad()
    def forward_lip(self, lips: torch.Tensor) -> torch.Tensor:
        """
        lips: (B, N, H, W) float32 [0,1], grayscale
        syncnet_python이 RGB/채널 입력을 기대하면 채널 복제.
        """
        if lips.ndim != 4:
            raise ValueError(f"lips must be (B,N,H,W), got {tuple(lips.shape)}")

        # (B, N, H, W) -> (B, 1, N, H, W)
        x = lips.unsqueeze(1)

        # if model expects 3 channels: (B,3,N,H,W)
        # (많은 fork가 3채널로 짬)
        if getattr(self.net, "expects_rgb", False):
            x = x.repeat(1, 3, 1, 1, 1)
        else:
            # 안전하게 3채널도 같이 시도 가능:
            # x = x.repeat(1, 3, 1, 1, 1)
            pass

        x = x.to(self.device)
        # ---- 너 fork의 lip forward 함수명에 맞게 수정 ----
        # 보통 self.net.forward_lip(x) 또는 self.net.forward_vid(x)
        if hasattr(self.net, "forward_lip"):
            v = self.net.forward_lip(x)
        elif hasattr(self.net, "forward_vid"):
            v = self.net.forward_vid(x)
        else:
            # 마지막 수단: net(x, None) 같은 형태면 여기도 맞춰야 함
            v = self.net(x)

        return self._l2norm(v)

    @torch.no_grad()
    def forward_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, 80, T) float32
        syncnet_python이 MFCC를 기대하면, 여기서 mel->mfcc로 바꾸는 대신
        'processed meta에 wav_path가 있다면' wav segment를 읽어서 mfcc로 만드는 쪽이 정확함.
        (일단은 mel 그대로 넣는 버전부터 맞추고, 필요하면 MFCC로 교체)
        """
        if mel.ndim != 3:
            raise ValueError(f"mel must be (B,80,T), got {tuple(mel.shape)}")
        x = mel.to(self.device)

        # ---- 너 fork의 audio forward 함수명에 맞게 수정 ----
        if hasattr(self.net, "forward_audio"):
            a = self.net.forward_audio(x)
        elif hasattr(self.net, "forward_aud"):
            a = self.net.forward_aud(x)
        else:
            a = self.net(x)

        return self._l2norm(a)

    def _l2norm(self, z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return z / (z.norm(p=2, dim=1, keepdim=True) + eps)