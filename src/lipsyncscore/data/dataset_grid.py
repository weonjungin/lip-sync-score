# /home/jiweon/projects/lip-sync-score/src/lipsyncscore/data/dataset_grid.py

import json
import math
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def sample_t0_biased(T_v: int, N: int = 5, margin: float = 0.15, p_back: float = 0.65):
    if T_v is None or T_v <= 0:
        return None
    if N <= 0:
        raise ValueError("N must be positive")
    if not (0.0 <= margin < 0.5):
        raise ValueError("margin must be in [0, 0.5)")
    if not (0.0 <= p_back <= 1.0):
        raise ValueError("p_back must be in [0, 1]")

    keep_start = math.ceil(margin * T_v)
    keep_end = math.floor((1 - margin) * T_v) - 1
    valid_start = keep_start
    valid_end = keep_end - (N - 1)

    if valid_end < valid_start:
        return None

    L = valid_end - valid_start + 1
    if L == 1:
        return valid_start

    mid = valid_start + L // 2
    if random.random() < p_back:
        return random.randint(mid, valid_end)
    else:
        return random.randint(valid_start, mid - 1)


def _get_first(meta: dict, keys: list):
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    raise KeyError(f"Missing keys {keys}. Available keys: {list(meta.keys())[:40]}...")


def _list_roi_files(roi_dir: Path):
    npy = sorted(roi_dir.glob("*.npy"))
    if npy:
        return npy
    png = sorted(roi_dir.glob("*.png"))
    if png:
        return png
    jpg = sorted(roi_dir.glob("*.jpg")) + sorted(roi_dir.glob("*.jpeg"))
    return jpg


def _load_roi_frame(path: Path) -> torch.Tensor:
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"ROI frame must be grayscale 2D. Got {arr.shape} at {path}")
        x = torch.from_numpy(arr).float()
        if x.numel() > 0 and x.max() > 1.5:
            x = x / 255.0
        return x

    try:
        from PIL import Image
    except Exception as e:
        raise ImportError("Pillow is required to load ROI images (.png/.jpg).") from e

    img = Image.open(path).convert("L")
    x = torch.from_numpy(np.array(img)).float()
    if x.numel() > 0 and x.max() > 1.5:
        x = x / 255.0
    return x


def _pad_feat_if_needed(feat: torch.Tensor, end_idx_exclusive: int) -> torch.Tensor:
    T = feat.shape[1]
    if end_idx_exclusive <= T:
        return feat
    pad = end_idx_exclusive - T
    return torch.cat([feat, torch.zeros(feat.shape[0], pad, dtype=feat.dtype)], dim=1)


def _sample_neg_t0_from_candidates(
    *,
    t0: int,
    T_v: int,
    N: int,
    fps: float,
    max_shift_sec: float,
    min_sep_frames: int,
    hard_prob: float = 0.0,
    hard_range: Optional[Tuple[int, int]] = None,
    avoid_zero: bool = True,
):
    """
    Returns (t0_neg, off_frames) satisfying:
      - t0_neg in [0, T_v-N]
      - |off| >= min_sep_frames
      - |off| <= round(max_shift_sec*fps)  (via window [t0-max_shift, t0+max_shift])

    hard sampling:
      - if hard_range=(lo,hi), then hard candidates are those with lo<=|off|<=hi
      - with probability hard_prob, sample from hard candidates (if exists), else sample from easy candidates.
    """
    max_shift_frames = int(round(max_shift_sec * fps))
    if max_shift_frames <= 0:
        raise ValueError("max_shift_sec too small for given fps")

    lo = max(0, t0 - max_shift_frames)
    hi = min(T_v - N, t0 + max_shift_frames)

    hard_cands = []
    easy_cands = []

    hard_lo, hard_hi = None, None
    if hard_range is not None:
        hard_lo, hard_hi = int(hard_range[0]), int(hard_range[1])
        if hard_lo < 0 or hard_hi < 0 or hard_hi < hard_lo:
            raise ValueError(f"Invalid hard_range={hard_range}. Expect (lo<=hi), non-negative.")

    for t in range(lo, hi + 1):
        off = t - t0
        if avoid_zero and off == 0:
            continue
        if abs(off) < min_sep_frames:
            continue

        if hard_lo is not None and hard_hi is not None and (hard_lo <= abs(off) <= hard_hi):
            hard_cands.append((t, off))
        else:
            easy_cands.append((t, off))

    if not hard_cands and not easy_cands:
        return None, None

    use_hard = (random.random() < float(hard_prob)) and (len(hard_cands) > 0)
    pool = hard_cands if use_hard else (easy_cands if len(easy_cands) > 0 else hard_cands)

    t0_neg, off = random.choice(pool)
    return int(t0_neg), int(off)


class DatasetGRID(Dataset):
    """
    Returns:
      lips    : (N,H,W)
      pos_mel : (F, audio_len)   # name kept for compatibility; may actually hold MFCC if audio_type='mfcc'
      neg_mel : (F, audio_len)

    Debug keys (optional):
      neg_off_frames, t0, t0_neg, ...
    """

    def __init__(
        self,
        utt_dirs,
        N=5,
        margin=0.15,
        p_back=0.65,
        pad_mel=True,
        return_debug=True,
        max_shift_sec=2.0,
        min_sep_frames=None,
        max_resample_utt=20,
        neg_hard_prob: float = 0.0,
        neg_hard_range: Optional[Tuple[Optional[int], Optional[int]]] = (None, None),
        neg_avoid_zero: bool = True,
        audio_type="mel",
    ):
        self.utt_dirs = [Path(u) for u in utt_dirs]
        self.N = int(N)
        self.margin = float(margin)
        self.p_back = float(p_back)
        self.pad_mel = bool(pad_mel)
        self.return_debug = bool(return_debug)

        self.max_shift_sec = float(max_shift_sec)
        self.min_sep_frames = max(self.N, 5) if min_sep_frames is None else int(min_sep_frames)
        self.max_resample_utt = int(max_resample_utt)

        self.neg_hard_prob = float(neg_hard_prob)
        self.neg_avoid_zero = bool(neg_avoid_zero)

        hr0, hr1 = neg_hard_range if neg_hard_range is not None else (None, None)
        if hr0 is None or hr1 is None:
            self.neg_hard_range = None
        else:
            self.neg_hard_range = (int(hr0), int(hr1))

        if not (0.0 <= self.neg_hard_prob <= 1.0):
            raise ValueError("neg_hard_prob must be in [0,1]")

        self.audio_type = str(audio_type).lower()
        if self.audio_type not in ("mel", "mfcc"):
            raise ValueError(f"audio_type must be 'mel' or 'mfcc', got {self.audio_type}")

    def __len__(self):
        return len(self.utt_dirs)

    def __getitem__(self, idx):
        for _try in range(self.max_resample_utt):
            utt_dir = self.utt_dirs[idx]

            meta_path = utt_dir / "meta.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            fps = float(_get_first(meta, ["video_fps", "fps", "video_frame_rate"]))
            sr = int(_get_first(meta, ["sr", "sample_rate"]))
            hop = int(_get_first(meta, ["hop_length_audio", "hop_length", "hop", "hop_length_mel"]))

            if self.audio_type == "mel":
                audio_meta_keys = ["mel_path", "mel_npy", "mel_file"]
                audio_fallbacks = ["mel_80_100hz.npy", "mel.npy", "mel_80.npy"]
                expected_feat_dim = 80
            else:
                audio_meta_keys = ["mfcc_path", "mfcc_npy", "mfcc_file"]
                audio_fallbacks = ["mfcc_13_100hz.npy", "mfcc.npy", "mfcc_13.npy"]
                expected_feat_dim = 13

            audio_path = None
            for k in audio_meta_keys:
                if k in meta and meta[k] is not None:
                    cand = utt_dir / str(meta[k])
                    if cand.exists():
                        audio_path = cand
                        break

            if audio_path is None:
                for cand_name in audio_fallbacks:
                    cand_path = utt_dir / cand_name
                    if cand_path.exists():
                        audio_path = cand_path
                        break

            if audio_path is None:
                idx = random.randint(0, len(self.utt_dirs) - 1)
                continue

            roi_dir = utt_dir / "roi"
            roi_files = _list_roi_files(roi_dir)
            T_v = len(roi_files)
            if T_v < self.N:
                idx = random.randint(0, len(self.utt_dirs) - 1)
                continue

            t0 = sample_t0_biased(T_v, N=self.N, margin=self.margin, p_back=self.p_back)
            if t0 is None:
                t0 = random.randint(0, T_v - self.N)

            t0_neg, off_frames = _sample_neg_t0_from_candidates(
                t0=t0,
                T_v=T_v,
                N=self.N,
                fps=fps,
                max_shift_sec=self.max_shift_sec,
                min_sep_frames=self.min_sep_frames,
                hard_prob=self.neg_hard_prob,
                hard_range=self.neg_hard_range,
                avoid_zero=self.neg_avoid_zero,
            )
            if t0_neg is None:
                idx = random.randint(0, len(self.utt_dirs) - 1)
                continue

            lips = torch.stack([_load_roi_frame(p) for p in roi_files[t0:t0 + self.N]], dim=0)

            feat = torch.from_numpy(np.load(audio_path)).float()
            if feat.ndim != 2 or feat.shape[0] != expected_feat_dim:
                idx = random.randint(0, len(self.utt_dirs) - 1)
                continue

            m_per_v = sr / hop / fps
            feat_len = max(1, int(round(self.N * m_per_v)))

            pos_mel_start = int(round(t0 * m_per_v))
            pos_mel_end = pos_mel_start + feat_len
            if self.pad_mel:
                feat = _pad_feat_if_needed(feat, pos_mel_end)
            else:
                pos_mel_start = max(0, min(pos_mel_start, feat.shape[1] - feat_len))
            pos_mel = feat[:, pos_mel_start:pos_mel_start + feat_len]

            neg_mel_start = int(round(t0_neg * m_per_v))
            neg_mel_end = neg_mel_start + feat_len
            if self.pad_mel:
                feat = _pad_feat_if_needed(feat, neg_mel_end)
            else:
                neg_mel_start = max(0, min(neg_mel_start, feat.shape[1] - feat_len))
            neg_mel = feat[:, neg_mel_start:neg_mel_start + feat_len]

            out = {
                "lips": lips,
                "pos_mel": pos_mel,
                "neg_mel": neg_mel,
                "t0": int(t0),
                "t0_neg": int(t0_neg),
                "neg_off_frames": int(off_frames),
            }

            if self.return_debug:
                out.update({
                    "utt_dir": str(utt_dir),
                    "T_v": int(T_v),
                    "fps": float(fps),
                    "sr": int(sr),
                    "hop": int(hop),
                    "audio_type": self.audio_type,
                    "audio_path": str(audio_path),
                    "m_per_v": float(m_per_v),
                    "mel_len": int(feat_len),
                    "pos_mel_start": int(pos_mel_start),
                    "neg_mel_start": int(neg_mel_start),
                    "max_shift_sec": float(self.max_shift_sec),
                    "min_sep_frames": int(self.min_sep_frames),
                    "neg_hard_prob": float(self.neg_hard_prob),
                    "neg_hard_range": str(self.neg_hard_range),
                    "neg_avoid_zero": bool(self.neg_avoid_zero),
                })

            return out

        raise RuntimeError(
            f"Failed to sample a valid (pos,neg) within {self.max_resample_utt} tries. "
            "Your processed data may contain many very short/invalid utterances."
        )