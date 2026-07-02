"""
dataset_hdtf_progressive.py

SyncNetProgressive 학습용 HDTF 데이터셋.
기존 DatasetHDTF와 동일한 구조지만 lips + jaw + face latent를 동시에 반환.

반환:
    lips    : [N, 4, 12, 12]  VAE latent - lip only (stage1)
    jaw     : [N, 4, 12, 12]  VAE latent - 하관 (stage2)
    face    : [N, 4, 12, 12]  VAE latent - full face (stage3)
    pos_mel : [80, mel_len]   pos 시점 mel
    neg_mel : [80, mel_len]   neg 시점 mel

디렉토리 구조:
    latent_root_s1/{video}/latent.npy   [T, 4, 12, 12]  lip only
    latent_root_s2/{video}/latent.npy   [T, 4, 12, 12]  하관
    latent_root_s3/{video}/latent.npy   [T, 4, 12, 12]  full face
    data_root/{video}/mel_latentsync.npy [80, T_mel]
    data_root/{video}/frame_indices.npy  [T]
"""

import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

MEL_STEP = 52
FPS      = 25


class DatasetHDTFProgressive(Dataset):
    """
    Args:
        latent_root_s1 : stage1 lip latent 루트    (/media/HDD/jiweon/hdtf_latents)
        latent_root_s2 : stage2 하관 latent 루트   (/media/HDD/jiweon/hdtf_latents_s2)
        latent_root_s3 : stage3 full face latent 루트 (/media/HDD/jiweon/hdtf_latents_s3)
        data_root      : HDTF processed 루트
        N              : 시퀀스 길이
        min_gap        : pos/neg 최소 프레임 간격
        max_gap        : pos/neg 최대 프레임 간격
        split_file     : train.txt / val.txt 경로
        samples_per_video : 비디오당 샘플 수
        min_frames     : 유효 비디오 최소 프레임 수
    """
    def __init__(
        self,
        latent_root_s1: str,
        latent_root_s2: str,
        latent_root_s3: str,
        data_root: str,
        N: int = 16,
        min_gap: int = 5,
        max_gap: int = 100,
        split_file: str = None,
        samples_per_video: int = 200,
        min_frames: int = 100,
    ):
        self.latent_root_s1 = Path(latent_root_s1)
        self.latent_root_s2 = Path(latent_root_s2)
        self.latent_root_s3 = Path(latent_root_s3)
        self.data_root      = Path(data_root)
        self.N              = N
        self.min_gap        = min_gap
        self.max_gap        = max_gap
        self.samples_per_video = samples_per_video

        # 비디오 목록
        if split_file is not None:
            with open(split_file) as f:
                video_names = [l.strip() for l in f if l.strip()]
        else:
            video_names = [
                d.name for d in sorted(self.latent_root_s1.iterdir())
                if d.is_dir()
            ]

        self.samples = []
        n_videos = 0

        for name in video_names:
            # 세 stage latent 모두 존재해야 유효
            p_s1 = self.latent_root_s1 / name / "latent.npy"
            p_s2 = self.latent_root_s2 / name / "latent.npy"
            p_s3 = self.latent_root_s3 / name / "latent.npy"
            mel_path  = self.data_root / name / "mel_latentsync.npy"
            fidx_path = self.data_root / name / "frame_indices.npy"

            if not all(p.exists() for p in [p_s1, p_s2, p_s3, mel_path, fidx_path]):
                continue

            try:
                z = np.load(str(p_s1), mmap_mode='r')
                T = z.shape[0]
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
                    "name":     name,
                    "p_s1":     str(p_s1),
                    "p_s2":     str(p_s2),
                    "p_s3":     str(p_s3),
                    "mel_path": str(mel_path),
                    "fidx_path":str(fidx_path),
                    "T":        T,
                    "t0":       t0,
                })
            n_videos += 1

        print(f"DatasetHDTFProgressive: {n_videos} videos, {len(self.samples)} samples", flush=True)

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

        # 세 stage latent 로드
        lat_s1 = np.load(s["p_s1"], mmap_mode='r')  # [T, 4, 12, 12]
        lat_s2 = np.load(s["p_s2"], mmap_mode='r')
        lat_s3 = np.load(s["p_s3"], mmap_mode='r')
        mel    = np.load(s["mel_path"],  mmap_mode='r')
        fidx   = np.load(s["fidx_path"], mmap_mode='r')

        t0_neg = self._sample_neg_t0(t0, T)

        lips = torch.tensor(np.array(lat_s1[t0:t0 + self.N]), dtype=torch.float32)
        jaw  = torch.tensor(np.array(lat_s2[t0:t0 + self.N]), dtype=torch.float32)
        face = torch.tensor(np.array(lat_s3[t0:t0 + self.N]), dtype=torch.float32)

        pos_mel = self._get_mel(mel, fidx, t0)
        neg_mel = self._get_mel(mel, fidx, t0_neg)

        return {
            "lips":    lips,     # [N, 4, 12, 12]
            "jaw":     jaw,      # [N, 4, 12, 12]
            "face":    face,     # [N, 4, 12, 12]
            "pos_mel": pos_mel,  # [80, MEL_STEP]
            "neg_mel": neg_mel,  # [80, MEL_STEP]
        }