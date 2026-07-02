"""
dataset_hdtf.py

SyncNetTemporal 재학습용 HDTF 데이터셋

반환:
    lips    : [N, 4, 12, 12]  VAE latent (pos 시점)
    pos_mel : [80, mel_len]   pos 시점 mel
    neg_mel : [80, mel_len]   neg 시점 mel (다른 시점)

구조:
    latent_root/{video}/latent.npy       [T, 4, 12, 12]
    data_root/{video}/mel_latentsync.npy [80, T_mel]
    data_root/{video}/frame_indices.npy  [T]
"""

import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

MEL_STEP = 52   # mel_latentsync.npy 기준 window 길이 (ADLip2 dataset과 동일)
FPS      = 25   # HDTF 기준


class DatasetHDTF(Dataset):
    """
    Args:
        latent_root : VAE latent 저장 루트 (/media/HDD/jiweon/hdtf_latents)
        data_root   : HDTF processed 루트 (/home/ihjung/HDTF_ssd/processed)
        N           : lip 시퀀스 길이 (프레임 수)
        min_gap     : pos/neg 최소 프레임 간격
        max_gap     : pos/neg 최대 프레임 간격 (None이면 제한 없음)
        split_file  : train.txt / val.txt 경로 (None이면 전체 사용)
        samples_per_video : 비디오당 샘플 수
    """
    def __init__(
        self,
        latent_root: str,
        data_root: str,
        N: int = 16,
        min_gap: int = 5,
        max_gap: int = 100,
        split_file: str = None,
        samples_per_video: int = 200,
        min_frames: int = 100,
    ):
        self.latent_root = Path(latent_root)
        self.data_root   = Path(data_root)
        self.N           = N
        self.min_gap     = min_gap
        self.max_gap     = max_gap
        self.samples_per_video = samples_per_video

        # 비디오 목록
        if split_file is not None:
            with open(split_file) as f:
                video_names = [l.strip() for l in f if l.strip()]
        else:
            video_names = [
                d.name for d in sorted(self.latent_root.iterdir())
                if d.is_dir()
            ]

        # 유효 비디오 필터링 + 샘플 생성
        self.samples = []
        n_videos = 0

        for name in video_names:
            latent_path = self.latent_root / name / "latent.npy"
            mel_path    = self.data_root   / name / "mel_latentsync.npy"
            fidx_path   = self.data_root   / name / "frame_indices.npy"

            if not latent_path.exists():
                continue
            if not mel_path.exists():
                continue
            if not fidx_path.exists():
                continue

            # 프레임 수 확인 (latent shape)
            try:
                z = np.load(str(latent_path), mmap_mode='r')
                T = z.shape[0]
            except Exception:
                continue

            if T < min_frames:
                continue

            # 유효한 pos 시작점: [0, T-N)
            valid_starts = list(range(0, T - N))
            if len(valid_starts) == 0:
                continue

            # samples_per_video개 샘플링
            chosen = random.choices(valid_starts, k=min(samples_per_video, len(valid_starts)))

            for t0 in chosen:
                self.samples.append({
                    "name":         name,
                    "latent_path":  str(latent_path),
                    "mel_path":     str(mel_path),
                    "fidx_path":    str(fidx_path),
                    "T":            T,
                    "t0":           t0,
                })
            n_videos += 1

        print(f"DatasetHDTF: {n_videos} videos, {len(self.samples)} samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def _sample_neg_t0(self, t0: int, T: int) -> int:
        """
        t0와 min_gap 이상, max_gap 이하 떨어진 neg 시작점 샘플링
        """
        candidates = []
        lo = max(0, t0 - self.max_gap) if self.max_gap else 0
        hi = min(T - self.N, t0 + self.max_gap) if self.max_gap else T - self.N

        for t in range(lo, hi + 1):
            if abs(t - t0) >= self.min_gap:
                candidates.append(t)

        if len(candidates) == 0:
            # fallback: min_gap만 만족하면 됨
            candidates = [
                t for t in range(0, T - self.N)
                if abs(t - t0) >= self.min_gap
            ]

        if len(candidates) == 0:
            return (t0 + self.min_gap) % max(1, T - self.N)

        return random.choice(candidates)

    def _get_mel(self, mel: np.ndarray, frame_indices: np.ndarray, t0: int) -> torch.Tensor:
        """
        t0 프레임 기준 mel window 추출 → [80, MEL_STEP]
        """
        orig_frame_idx = int(frame_indices[t0])
        # mel_latentsync는 25fps 기준
        s = int(80. * (orig_frame_idx / FPS))
        chunk = mel[:, s:s + MEL_STEP]
        if chunk.shape[1] < MEL_STEP:
            chunk = np.pad(chunk, ((0, 0), (0, MEL_STEP - chunk.shape[1])))
        return torch.tensor(np.array(chunk), dtype=torch.float32)  # [80, MEL_STEP]

    def __getitem__(self, idx):
        s = self.samples[idx]
        T  = s["T"]
        t0 = s["t0"]

        # latent 로드
        latent = np.load(s["latent_path"], mmap_mode='r')  # [T, 4, 12, 12]
        mel    = np.load(s["mel_path"],    mmap_mode='r')  # [80, T_mel]
        fidx   = np.load(s["fidx_path"],   mmap_mode='r')  # [T]

        # neg 시점 샘플링
        t0_neg = self._sample_neg_t0(t0, T)

        # lip latent 슬라이싱
        lips = torch.tensor(
            np.array(latent[t0:t0 + self.N]),
            dtype=torch.float32,
        )  # [N, 4, 12, 12]

        # mel 추출
        pos_mel = self._get_mel(mel, fidx, t0)       # [80, MEL_STEP]
        neg_mel = self._get_mel(mel, fidx, t0_neg)   # [80, MEL_STEP]

        return {
            "lips":    lips,     # [N, 4, 12, 12]
            "pos_mel": pos_mel,  # [80, MEL_STEP]
            "neg_mel": neg_mel,  # [80, MEL_STEP]
        }