"""
dataset_hdtf_progressive_merged.py

merge_latents.py로 생성한 merged latent 사용.
파일 하나만 열어서 IO 병목 해소.

반환:
    lips    : [N, 4, 12, 12]  latent[:, 0]  lip only
    jaw     : [N, 4, 12, 12]  latent[:, 1]  하관
    face    : [N, 4, 12, 12]  latent[:, 2]  full face
    pos_mel : [80, mel_len]
    neg_mel : [80, mel_len]

디렉토리 구조:
    latent_root/{video}/latent.npy  (T, 3, 4, 12, 12)
    data_root/{video}/mel_latentsync.npy
    data_root/{video}/frame_indices.npy
"""

import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

MEL_STEP = 52
FPS      = 25


class DatasetHDTFProgressiveMerged(Dataset):
    """
    Args:
        latent_root : merged latent 루트 (/media/HDD/jiweon/latents/hdtf_merged)
        data_root   : HDTF processed 루트
        N           : 시퀀스 길이
        min_gap     : pos/neg 최소 프레임 간격
        max_gap     : pos/neg 최대 프레임 간격
        split_file  : train.txt / val.txt 경로
        samples_per_video : 비디오당 샘플 수
        min_frames  : 유효 비디오 최소 프레임 수
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

        if split_file is not None:
            with open(split_file) as f:
                video_names = [l.strip() for l in f if l.strip()]
        else:
            video_names = [
                d.name for d in sorted(self.latent_root.iterdir())
                if d.is_dir()
            ]

        self.samples = []
        n_videos = 0

        for name in video_names:
            lat_path  = self.latent_root / name / "latent.npy"
            mel_path  = self.data_root   / name / "mel_latentsync.npy"
            fidx_path = self.data_root   / name / "frame_indices.npy"

            if not all(p.exists() for p in [lat_path, mel_path, fidx_path]):
                continue

            try:
                z = np.load(str(lat_path), mmap_mode='r')
                T = z.shape[0]
                # shape 검증: (T, 3, 4, 12, 12)
                assert z.ndim == 5 and z.shape[1] == 3, \
                    f"Expected (T,3,4,12,12), got {z.shape}"
            except Exception:
                continue

            if T < min_frames:
                continue

            valid_starts = list(range(0, T - N))
            if not valid_starts:
                continue

            chosen = random.choices(valid_starts, k=min(samples_per_video, len(valid_starts)))

            for t0 in chosen:
                self.samples.append({
                    "name":      name,
                    "lat_path":  str(lat_path),
                    "mel_path":  str(mel_path),
                    "fidx_path": str(fidx_path),
                    "T":         T,
                    "t0":        t0,
                })
            n_videos += 1

        print(f"DatasetHDTFProgressiveMerged: {n_videos} videos, {len(self.samples)} samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def _sample_neg_t0(self, t0: int, T: int) -> int:
        lo = max(0, t0 - self.max_gap) if self.max_gap else 0
        hi = min(T - self.N, t0 + self.max_gap) if self.max_gap else T - self.N

        candidates = [t for t in range(lo, hi + 1) if abs(t - t0) >= self.min_gap]
        if not candidates:
            candidates = [t for t in range(0, T - self.N) if abs(t - t0) >= self.min_gap]
        if not candidates:
            return (t0 + self.min_gap) % max(1, T - self.N)
        return random.choice(candidates)

    def _get_mel(self, mel: np.ndarray, frame_indices: np.ndarray, t0: int) -> torch.Tensor:
        orig_frame_idx = int(frame_indices[t0])
        s = int(80. * (orig_frame_idx / FPS))
        chunk = mel[:, s:s + MEL_STEP]
        if chunk.shape[1] < MEL_STEP:
            chunk = np.pad(chunk, ((0, 0), (0, MEL_STEP - chunk.shape[1])))
        return torch.tensor(np.array(chunk), dtype=torch.float32)

    def __getitem__(self, idx):
        s  = self.samples[idx]
        T  = s["T"]
        t0 = s["t0"]

        # 파일 하나만 열기 (T, 3, 4, 12, 12)
        lat  = np.load(s["lat_path"],  mmap_mode='r')
        mel  = np.load(s["mel_path"],  mmap_mode='r')
        fidx = np.load(s["fidx_path"], mmap_mode='r')

        t0_neg = self._sample_neg_t0(t0, T)

        chunk = np.array(lat[t0:t0 + self.N])  # (N, 3, 4, 12, 12)

        lips = torch.tensor(chunk[:, 0], dtype=torch.float32)  # (N, 4, 12, 12)
        jaw  = torch.tensor(chunk[:, 1], dtype=torch.float32)
        face = torch.tensor(chunk[:, 2], dtype=torch.float32)

        pos_mel = self._get_mel(mel, fidx, t0)
        neg_mel = self._get_mel(mel, fidx, t0_neg)

        return {
            "lips":    lips,
            "jaw":     jaw,
            "face":    face,
            "pos_mel": pos_mel,
            "neg_mel": neg_mel,
        }