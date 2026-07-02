"""
eval_hdtf_progressive.py

SyncNetProgressive (Local-to-Global Hierarchical Sync) 평가 스크립트.
merged latent (T, 3, 4, 12, 12) 기반 offset curve 측정.

사용법:
    # expP1 (fusion=last)
    PYTHONPATH=src python scripts/eval_hdtf_progressive.py \
        --ckpt logs/expP1/checkpoints/best.pth \
        --latent_root /media/HDD/jiweon/latents/hdtf_merged \
        --data_root /home/ihjung/HDTF_ssd/processed \
        --val_split /home/jiweon/projects/ADLip2/data/splits/val.txt \
        --fusion_mode last \
        --device cuda:0 \
        --out logs/expP1/eval_offset.png

    # expP2 (fusion=concat)
    PYTHONPATH=src python scripts/eval_hdtf_progressive.py \
        --ckpt logs/expP2/checkpoints/best.pth \
        --latent_root /media/HDD/jiweon/latents/hdtf_merged \
        --data_root /home/ihjung/HDTF_ssd/processed \
        --val_split /home/jiweon/projects/ADLip2/data/splits/val.txt \
        --fusion_mode concat \
        --device cuda:0 \
        --out logs/expP2/eval_offset.png
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/home/jiweon/projects/lip-sync-score/src")
from lipsyncscore.models.modified.syncnet_progressive import SyncNetProgressive

MEL_STEP = 52
FPS      = 25
N        = 16
MIN_GAP  = 5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path, fusion_mode, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # config에서 모델 파라미터 읽기 (저장된 config 우선)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    model = SyncNetProgressive(
        in_frames       = int(model_cfg.get("in_frames", N)),
        emb_dim         = int(model_cfg.get("emb_dim", 256)),
        token_dim       = int(model_cfg.get("token_dim", 256)),
        n_heads         = int(model_cfg.get("n_heads", 4)),
        dropout         = float(model_cfg.get("dropout", 0.0)),
        temporal_cfg    = model_cfg.get("temporal", None),
        pooling         = str(model_cfg.get("pooling", "mean")),
        fusion_mode     = fusion_mode,
        lip_in_channels = int(model_cfg.get("lip_in_channels", 4)),
    ).to(device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"Loaded: {ckpt_path}  fusion_mode={fusion_mode}", flush=True)
    return model


def get_mel(mel, fidx, t0):
    orig_frame_idx = int(fidx[t0])
    s = int(80. * (orig_frame_idx / FPS))
    chunk = mel[:, s:s + MEL_STEP]
    if chunk.shape[1] < MEL_STEP:
        chunk = np.pad(chunk, ((0, 0), (0, MEL_STEP - chunk.shape[1])))
    return torch.tensor(np.array(chunk), dtype=torch.float32)


def eval_offset(
    model, latent_root, data_root, val_split, device,
    offsets=range(-20, 21), num_utts=200, samples_per_utt=5,
):
    with open(val_split) as f:
        video_names = [l.strip() for l in f if l.strip()]

    # 유효 비디오 필터링
    valid = []
    for name in video_names:
        lp = Path(latent_root) / name / "latent.npy"
        mp = Path(data_root)   / name / "mel_latentsync.npy"
        fp = Path(data_root)   / name / "frame_indices.npy"
        if lp.exists() and mp.exists() and fp.exists():
            valid.append(name)

    random.shuffle(valid)
    valid = valid[:num_utts]
    print(f"Eval videos: {len(valid)}", flush=True)

    offset_distances = {off: [] for off in offsets}

    with torch.no_grad():
        for name in tqdm(valid, desc="Eval"):
            # merged latent (T, 3, 4, 12, 12)
            latent = np.load(str(Path(latent_root) / name / "latent.npy"), mmap_mode='r')
            mel    = np.load(str(Path(data_root)   / name / "mel_latentsync.npy"), mmap_mode='r')
            fidx   = np.load(str(Path(data_root)   / name / "frame_indices.npy"), mmap_mode='r')
            T      = latent.shape[0]

            if T < N + max(abs(min(offsets)), abs(max(offsets))) + MIN_GAP:
                continue

            for _ in range(samples_per_utt):
                t0 = random.randint(0, T - N - 1)

                chunk = np.array(latent[t0:t0 + N])  # (N, 3, 4, 12, 12)

                lips = torch.tensor(chunk[:, 0], dtype=torch.float32).unsqueeze(0).to(device)
                jaw  = torch.tensor(chunk[:, 1], dtype=torch.float32).unsqueeze(0).to(device)
                face = torch.tensor(chunk[:, 2], dtype=torch.float32).unsqueeze(0).to(device)

                mel_pos = get_mel(mel, fidx, t0).unsqueeze(0).to(device)

                v_embed = model.forward_lip(lips, jaw, face)  # (1, emb_dim)
                a_pos   = model.forward_audio(mel_pos)        # (1, emb_dim)

                for off in offsets:
                    t_neg = t0 + off
                    if t_neg < 0 or t_neg + N > T:
                        continue

                    if off == 0:
                        a_neg = a_pos
                    else:
                        mel_neg = get_mel(mel, fidx, t_neg).unsqueeze(0).to(device)
                        a_neg   = model.forward_audio(mel_neg)

                    dist = F.pairwise_distance(v_embed, a_neg).item()
                    offset_distances[off].append(dist)

    results = {}
    for off in offsets:
        dists = offset_distances[off]
        results[off] = float(np.mean(dists)) if dists else None

    return results


def plot_offset_curve(results, out_path, title="SyncNetProgressive — Offset Curve"):
    try:
        import matplotlib.pyplot as plt

        offsets = sorted([k for k, v in results.items() if v is not None])
        dists   = [results[k] for k in offsets]

        plt.figure(figsize=(10, 5))
        plt.plot(offsets, dists, 'b-o', markersize=4, label="distance")
        plt.axvline(x=0, color='r', linestyle='--', label="offset=0 (pos)")
        plt.xlabel("Audio offset (frames)")
        plt.ylabel("Mean pairwise distance")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot saved: {out_path}", flush=True)
    except ImportError:
        print("matplotlib 없음 — 숫자만 출력", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",            default="logs/expP1/checkpoints/best.pth")
    parser.add_argument("--latent_root",     default="/media/HDD/jiweon/latents/hdtf_merged")
    parser.add_argument("--data_root",       default="/home/ihjung/HDTF_ssd/processed")
    parser.add_argument("--val_split",       default="/home/jiweon/projects/ADLip2/data/splits/val.txt")
    parser.add_argument("--fusion_mode",     default="last", choices=["last", "concat"])
    parser.add_argument("--device",          default="cuda:0")
    parser.add_argument("--num_utts",        type=int, default=200)
    parser.add_argument("--samples_per_utt", type=int, default=5)
    parser.add_argument("--offset_min",      type=int, default=-20)
    parser.add_argument("--offset_max",      type=int, default=20)
    parser.add_argument("--out",             default="logs/expP1/eval_offset.png")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)

    model   = load_model(args.ckpt, args.fusion_mode, device)
    offsets = range(args.offset_min, args.offset_max + 1)

    results = eval_offset(
        model,
        latent_root     = args.latent_root,
        data_root       = args.data_root,
        val_split       = args.val_split,
        device          = device,
        offsets         = offsets,
        num_utts        = args.num_utts,
        samples_per_utt = args.samples_per_utt,
    )

    # 결과 출력
    print("\n=== Offset Curve ===")
    for off in sorted(results.keys()):
        v = results[off]
        marker = " ← pos" if off == 0 else ""
        if v is not None:
            print(f"  offset={off:+3d}: dist={v:.4f}{marker}")

    d0   = results.get(0)
    dmax = max(v for v in results.values() if v is not None)
    if d0 is not None:
        print(f"\nd_pos(offset=0) = {d0:.4f}")
        print(f"d_neg(max)      = {dmax:.4f}")
        print(f"gap             = {dmax - d0:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_offset_curve(
        results,
        out_path = args.out,
        title    = f"SyncNetProgressive fusion={args.fusion_mode} — Offset Curve",
    )


if __name__ == "__main__":
    main()