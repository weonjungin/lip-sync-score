"""
eval_hdtf_latent.py

HDTF latent + mel 기반 SyncNetTemporal 평가
offset curve: offset별 평균 distance 측정

사용법:
    PYTHONPATH=src python scripts/eval_hdtf_latent.py \
        --ckpt logs/syncnet_hdtf/checkpoints/best.pth \
        --latent_root /media/HDD/jiweon/hdtf_latents \
        --data_root /home/ihjung/HDTF_ssd/processed \
        --val_split /home/jiweon/projects/ADLip2/data/splits/val.txt \
        --device cuda:0 \
        --out logs/syncnet_hdtf/eval_offset.png
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
from lipsyncscore.models.modified.syncnet_temporal import SyncNetTemporal

MEL_STEP = 52
FPS      = 25
N        = 16
MIN_GAP  = 5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path, device):
    model = SyncNetTemporal(
        in_frames=16,
        emb_dim=256,
        pooling="attn",
        lip_in_channels=4,
        temporal_cfg={"type": "gru", "num_layers": 2,
                      "bidirectional": True, "hidden_dim": 256},
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded: {ckpt_path}", flush=True)
    return model


def get_mel(mel, fidx, t0):
    orig_frame_idx = int(fidx[t0])
    s = int(80. * (orig_frame_idx / FPS))
    chunk = mel[:, s:s + MEL_STEP]
    if chunk.shape[1] < MEL_STEP:
        chunk = np.pad(chunk, ((0, 0), (0, MEL_STEP - chunk.shape[1])))
    return torch.tensor(np.array(chunk), dtype=torch.float32)  # [80, MEL_STEP]


def eval_offset(model, latent_root, data_root, val_split, device,
                offsets=range(-20, 21), num_utts=200, samples_per_utt=5):
    """
    offset별 평균 distance 측정
    offset=0: pos (같은 시점)
    offset≠0: neg (다른 시점)
    """
    # 비디오 목록
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
            latent = np.load(str(Path(latent_root) / name / "latent.npy"), mmap_mode='r')
            mel    = np.load(str(Path(data_root)   / name / "mel_latentsync.npy"), mmap_mode='r')
            fidx   = np.load(str(Path(data_root)   / name / "frame_indices.npy"), mmap_mode='r')
            T      = latent.shape[0]

            if T < N + max(abs(min(offsets)), abs(max(offsets))) + MIN_GAP:
                continue

            for _ in range(samples_per_utt):
                # pos 시점 샘플링
                t0 = random.randint(0, T - N - 1)

                lips = torch.tensor(
                    np.array(latent[t0:t0 + N]),
                    dtype=torch.float32,
                ).unsqueeze(0).to(device)  # [1, 16, 4, 12, 12]

                mel_pos_np = get_mel(mel, fidx, t0)
                mel_pos = mel_pos_np.unsqueeze(0).to(device)  # [1, 80, 52]

                v_embed = model.forward_lip(lips)      # [1, 256]
                a_pos   = model.forward_audio(mel_pos) # [1, 256]

                for off in offsets:
                    t_neg = t0 + off
                    if t_neg < 0 or t_neg + N > T:
                        continue

                    mel_neg_np = get_mel(mel, fidx, t_neg)
                    mel_neg = mel_neg_np.unsqueeze(0).to(device)

                    if off == 0:
                        a_neg = a_pos
                    else:
                        a_neg = model.forward_audio(mel_neg)

                    dist = F.pairwise_distance(v_embed, a_neg).item()
                    offset_distances[off].append(dist)

    # 평균 계산
    results = {}
    for off in offsets:
        dists = offset_distances[off]
        if len(dists) > 0:
            results[off] = float(np.mean(dists))
        else:
            results[off] = None

    return results


def plot_offset_curve(results, out_path):
    try:
        import matplotlib.pyplot as plt

        offsets = sorted([k for k, v in results.items() if v is not None])
        dists   = [results[k] for k in offsets]

        plt.figure(figsize=(10, 5))
        plt.plot(offsets, dists, 'b-o', markersize=4, label="distance")
        plt.axvline(x=0, color='r', linestyle='--', label="offset=0 (pos)")
        plt.xlabel("Audio offset (frames)")
        plt.ylabel("Mean pairwise distance")
        plt.title("SyncNet HDTF Eval — Offset Curve")
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
    parser.add_argument("--ckpt",        default="logs/syncnet_hdtf/checkpoints/best.pth")
    parser.add_argument("--latent_root", default="/media/HDD/jiweon/hdtf_latents")
    parser.add_argument("--data_root",   default="/home/ihjung/HDTF_ssd/processed")
    parser.add_argument("--val_split",   default="/home/jiweon/projects/ADLip2/data/splits/val.txt")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--num_utts",    type=int, default=200)
    parser.add_argument("--samples_per_utt", type=int, default=5)
    parser.add_argument("--offset_min",  type=int, default=-20)
    parser.add_argument("--offset_max",  type=int, default=20)
    parser.add_argument("--out",         default="logs/syncnet_hdtf/eval_offset.png")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)

    model   = load_model(args.ckpt, device)
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

    # 그래프 저장
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_offset_curve(results, args.out)


if __name__ == "__main__":
    main()