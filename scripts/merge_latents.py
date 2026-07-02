"""
merge_latents.py

s1, s2, s3 latent를 하나의 파일로 합치기.
기존 파일 건드리지 않고 새로 생성.

입력:
    /media/HDD/jiweon/latents/hdtf_s1/{video}/latent.npy  (T, 4, 12, 12)
    /media/HDD/jiweon/latents/hdtf_s2/{video}/latent.npy  (T, 4, 12, 12)
    /media/HDD/jiweon/latents/hdtf_s3/{video}/latent.npy  (T, 4, 12, 12)

출력:
    /media/HDD/jiweon/latents/hdtf_merged/{video}/latent.npy  (T, 3, 4, 12, 12)
    axis=1: [0]=s1(lip), [1]=s2(하관), [2]=s3(full face)

실행:
    python scripts/merge_latents.py
    python scripts/merge_latents.py --max_videos 5  # 테스트
    python scripts/merge_latents.py --num_workers 4  # 병렬
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def merge_one(video_name, root_s1, root_s2, root_s3, out_root, overwrite):
    p_s1 = Path(root_s1) / video_name / "latent.npy"
    p_s2 = Path(root_s2) / video_name / "latent.npy"
    p_s3 = Path(root_s3) / video_name / "latent.npy"

    out_dir  = Path(out_root) / video_name
    out_path = out_dir / "latent.npy"

    # skip
    if out_path.exists() and not overwrite:
        return video_name, "skip"

    if not all(p.exists() for p in [p_s1, p_s2, p_s3]):
        missing = [str(p) for p in [p_s1, p_s2, p_s3] if not p.exists()]
        return video_name, f"missing: {missing}"

    try:
        s1 = np.load(str(p_s1), mmap_mode='r')  # (T, 4, 12, 12)
        s2 = np.load(str(p_s2), mmap_mode='r')
        s3 = np.load(str(p_s3), mmap_mode='r')

        # shape 체크
        if not (s1.shape == s2.shape == s3.shape):
            return video_name, f"shape mismatch: {s1.shape} {s2.shape} {s3.shape}"

        # stack → (T, 3, 4, 12, 12)
        merged = np.stack([
            np.array(s1),
            np.array(s2),
            np.array(s3),
        ], axis=1)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), merged)

        return video_name, "done"

    except Exception as e:
        return video_name, f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_s1",     default="/media/HDD/jiweon/latents/hdtf_s1")
    ap.add_argument("--root_s2",     default="/media/HDD/jiweon/latents/hdtf_s2")
    ap.add_argument("--root_s3",     default="/media/HDD/jiweon/latents/hdtf_s3")
    ap.add_argument("--out_root",    default="/media/HDD/jiweon/latents/hdtf_merged")
    ap.add_argument("--overwrite",   action="store_true")
    ap.add_argument("--max_videos",  type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=1)
    args = ap.parse_args()

    root_s1 = Path(args.root_s1)

    video_names = sorted([d.name for d in root_s1.iterdir() if d.is_dir()])
    if args.max_videos:
        video_names = video_names[:args.max_videos]

    print(f"[Merge] 총 {len(video_names)}개 비디오")
    print(f"[Merge] 출력: {args.out_root}")

    worker_fn = partial(
        merge_one,
        root_s1=args.root_s1,
        root_s2=args.root_s2,
        root_s3=args.root_s3,
        out_root=args.out_root,
        overwrite=args.overwrite,
    )

    done = skip = error = 0

    if args.num_workers <= 1:
        results = [worker_fn(name) for name in tqdm(video_names)]
    else:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_fn, video_names),
                total=len(video_names),
            ))

    for name, status in results:
        if status == "skip":
            skip += 1
        elif status == "done":
            done += 1
        else:
            error += 1
            print(f"  [{name}] {status}")

    print(f"\n[Done] done={done}  skip={skip}  error={error}")


if __name__ == "__main__":
    main()