"""
preprocess_latent.py

HDTF processed 디렉토리의 각 비디오 → VAE encode → latent.npy 저장

Curriculum Stage:
    stage1: frames/           lip only crop    → /media/HDD/jiweon/hdtf_latents/
    stage2: lower_frames/     입+턱 crop       → /media/HDD/jiweon/hdtf_latents_s2/
    stage3: lower_frames_s3/  하관 전체 crop   → /media/HDD/jiweon/hdtf_latents_s3/

사용법:
    # Stage 1 (lip only - 기존)
    PYTHONPATH=src python scripts/preprocess_latent.py \
        --stage stage1 \
        --data_root /home/ihjung/HDTF_ssd/processed \
        --out_root  /media/HDD/jiweon/hdtf_latents \
        --device cuda:0

    # Stage 2 (입+턱)
    PYTHONPATH=src python scripts/preprocess_latent.py \
        --stage stage2 \
        --data_root /home/ihjung/HDTF_ssd/processed \
        --out_root  /media/HDD/jiweon/hdtf_latents_s2 \
        --device cuda:0

    # Stage 3 (하관 전체)
    PYTHONPATH=src python scripts/preprocess_latent.py \
        --stage stage3 \
        --data_root /home/ihjung/HDTF_ssd/processed \
        --out_root  /media/HDD/jiweon/hdtf_latents_s3 \
        --device cuda:0

출력:
    모든 stage: {out_root}/{video_name}/latent.npy  shape: (T, 4, 12, 12)
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/home/jiweon/projects/ADLip2")
from models.vae_wrapper import VAEWrapper

IMG_SIZE = 96  # VAE 입력 크기 (모든 stage 동일)


def load_frames(frame_dir: Path, img_size: int = IMG_SIZE):
    """
    frame_dir의 이미지를 읽어서 텐서로 반환.
    Stage 1: frames/
    Stage 2: lower_frames/
    Stage 3: lower_frames_s3/
    모두 이미 96x96으로 저장돼 있음.
    """
    frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        return None

    imgs = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        t = torch.tensor(img).permute(2, 0, 1).float() / 127.5 - 1.0
        imgs.append(t)

    return torch.stack(imgs, dim=0) if imgs else None


def encode_frames(vae, frames: torch.Tensor, device, batch_size: int = 16) -> np.ndarray:
    all_latents = []
    with torch.no_grad():
        for i in range(0, frames.shape[0], batch_size):
            chunk = frames[i:i + batch_size].to(device)
            z = vae.encode(chunk)
            all_latents.append(z.cpu().numpy())
    return np.concatenate(all_latents, axis=0)


# stage별 frame 디렉토리 이름
STAGE_DIR = {
    "stage1": "frames",
    "stage2": "lower_frames",
    "stage3": "lower_frames_s3",
}


def get_video_dirs(data_root: Path, stage: str):
    frame_dir_name = STAGE_DIR[stage]
    dirs = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / frame_dir_name).exists():
            continue
        if not (d / "mel_latentsync.npy").exists():
            continue
        dirs.append(d)
    return dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",      required=True, choices=["stage1", "stage2", "stage3"],
                        help="stage1=lip, stage2=입+턱, stage3=하관전체")
    parser.add_argument("--data_root",  default="/home/ihjung/HDTF_ssd/processed")
    parser.add_argument("--out_root",   required=True)
    parser.add_argument("--model_name", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="처리할 최대 비디오 수 (테스트용)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    device    = torch.device(args.device)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[Stage]  {args.stage}", flush=True)
    print(f"[Input]  {data_root}", flush=True)
    print(f"[Output] {out_root}", flush=True)

    print(f"[VAE] Loading: {args.model_name}", flush=True)
    vae = VAEWrapper(args.model_name).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    with torch.no_grad():
        z_dummy = vae.encode(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device))
        print(f"[VAE] Latent shape per frame: {z_dummy.shape}", flush=True)

    video_dirs = get_video_dirs(data_root, args.stage)
    if args.max_videos is not None:
        video_dirs = video_dirs[:args.max_videos]
    print(f"[Data] Total videos: {len(video_dirs)}", flush=True)

    frame_dir_name = STAGE_DIR[args.stage]
    skip = done = error = 0

    for video_dir in tqdm(video_dirs, desc=f"Encoding [{args.stage}]"):
        video_name = video_dir.name
        save_dir   = out_root / video_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path   = save_dir / "latent.npy"

        if out_path.exists() and not args.overwrite:
            skip += 1
            continue

        try:
            frames = load_frames(video_dir / frame_dir_name)

            if frames is None:
                print(f"  SKIP (no frames): {video_name}", flush=True)
                error += 1
                continue

            latents = encode_frames(vae, frames, device, batch_size=args.batch_size)
            np.save(str(out_path), latents)
            done += 1

        except Exception as e:
            print(f"  ERROR {video_name}: {e}", flush=True)
            error += 1

    print(f"\n[Done] done={done}  skip={skip}  error={error}", flush=True)


if __name__ == "__main__":
    main()