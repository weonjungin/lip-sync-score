import os
import argparse
import subprocess
import json
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
import librosa
import shutil
from pathlib import Path
import yaml
import glob


# ---------------------------------------
# Run command
# ---------------------------------------
def run_cmd(cmd: str):
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)


# ---------------------------------------
# YAML loader (project-relative 지원)
# ---------------------------------------
def load_yaml(path: str):
    p = Path(path).expanduser()
    if not p.is_absolute():
        proj_root = Path(__file__).resolve().parents[1]
        p = (proj_root / p).resolve()
    with open(p, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------
# Write meta.json
# ---------------------------------------
def write_meta(sample_dir: str, meta: dict):
    meta_path = os.path.join(sample_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------
# Extract audio wav (16k mono) from video
# ---------------------------------------
def extract_audio_wav(video_path: str, wav_path: str, sr: int = 16000):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    run_cmd(
        f'ffmpeg -y -i "{video_path}" -vn -ac 1 -ar {sr} -c:a pcm_s16le "{wav_path}" -loglevel error'
    )


# ---------------------------------------
# Compute mel (from wav array)  [single audio load]
# - returns mel_db float32 (80, T)
# ---------------------------------------
def compute_mel_from_wav(
    wav: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 80,
    to_db: bool = True,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    )
    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype("float32")


# ---------------------------------------
# Compute MFCC from mel_db (no redundant mel compute)
# - returns mfcc float32 (13, T)
# ---------------------------------------
def compute_mfcc_from_mel_db(mel_db: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)
    return mfcc.astype("float32")


# ---------------------------------------
# Extract lips ROI using insightface
# - square crop
# - temporal smoothing (EMA)
# - grayscale 저장 + size x size
# ---------------------------------------
def extract_lips(
    detector,
    video_path,
    roi_out_dir,
    size=96,              
    smooth_alpha=0.4,
    min_side_px=20,
):
    os.makedirs(roi_out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    idx = 0
    n_frames = 0
    n_no_face = 0
    n_multi_face = 0
    n_saved = 0

    prev_gray = None
    motion_vals = []
    smooth_state = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        n_frames += 1
        faces = detector.get(frame)

        if len(faces) == 0:
            n_no_face += 1
            idx += 1
            continue

        if len(faces) >= 2:
            n_multi_face += 1

        face = faces[0]
        lm = face.landmark_3d_68
        if lm is None or lm.shape[0] != 68:
            idx += 1
            continue

        # mouth landmarks (48~67)
        mouth = lm[48:68, :2]
        x1 = float(np.min(mouth[:, 0]))
        y1 = float(np.min(mouth[:, 1]))
        x2 = float(np.max(mouth[:, 0]))
        y2 = float(np.max(mouth[:, 1]))

        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        # ratio-based margins (tunable later)
        mx = 0.20 * w
        top = 0.08 * h
        bot = 0.50 * h

        x1m = x1 - mx
        y1m = y1 - top
        x2m = x2 + mx
        y2m = y2 + bot

        # square crop around center
        cx = (x1m + x2m) / 2.0
        cy = (y1m + y2m) / 2.0 + 0.02 * h

        side = max(x2m - x1m, y2m - y1m) * 0.90
        side = max(float(min_side_px), side)

        # temporal smoothing (EMA on cx,cy,side)
        if smooth_state is None:
            cx_s, cy_s, side_s = cx, cy, side
        else:
            pcx, pcy, pside = smooth_state
            a = float(smooth_alpha)
            cx_s = a * cx + (1 - a) * pcx
            cy_s = a * cy + (1 - a) * pcy
            side_s = a * side + (1 - a) * pside

        smooth_state = (cx_s, cy_s, side_s)

        # bbox coords
        half = side_s / 2.0
        x1c = int(round(cx_s - half))
        y1c = int(round(cy_s - half))
        x2c = int(round(cx_s + half))
        y2c = int(round(cy_s + half))

        # clamp
        H, W = frame.shape[:2]
        x1c = max(0, x1c)
        y1c = max(0, y1c)
        x2c = min(W, x2c)
        y2c = min(H, y2c)

        roi = frame[y1c:y2c, x1c:x2c]
        if roi.size > 0 and (x2c - x1c) > 1 and (y2c - y1c) > 1:
            roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                m = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
                motion_vals.append(m)
            prev_gray = gray

            cv2.imwrite(os.path.join(roi_out_dir, f"{idx:05d}.png"), gray)
            n_saved += 1

        idx += 1

    cap.release()

    stats = {
        "n_frames": n_frames,
        "n_no_face": n_no_face,
        "n_multi_face": n_multi_face,
        "n_saved": n_saved,
        "no_face_ratio": (n_no_face / n_frames) if n_frames > 0 else 1.0,
        "multi_face_ratio": (n_multi_face / n_frames) if n_frames > 0 else 0.0,
        "motion_energy": float(np.mean(motion_vals)) if len(motion_vals) > 0 else 0.0,
        "params": {
            "smooth_alpha": float(smooth_alpha),
            "min_side_px": int(min_side_px),
            "out_size": int(size),
            "gray": True
        }
    }
    return stats


# ---------------------------------------
# Success criteria
# - ROI png >= min_roi_frames
# - audio.wav exists
# - mel_80_100hz.npy exists with (80, >=min_audio_frames)
# - mfcc_13_100hz.npy exists with (13, >=min_audio_frames) and same T
# ---------------------------------------
def is_success(sample_dir, min_roi_frames=5, min_audio_frames=20):
    roi_dir = os.path.join(sample_dir, "roi")
    wav_path = os.path.join(sample_dir, "audio.wav")
    mel_path = os.path.join(sample_dir, "mel_80_100hz.npy")
    mfcc_path = os.path.join(sample_dir, "mfcc_13_100hz.npy")

    if not os.path.isdir(roi_dir):
        return False
    frames = [f for f in os.listdir(roi_dir) if f.lower().endswith(".png")]
    if len(frames) < min_roi_frames:
        return False

    if not os.path.isfile(wav_path):
        return False
    if not os.path.isfile(mel_path):
        return False
    if not os.path.isfile(mfcc_path):
        return False

    try:
        mel = np.load(mel_path)
        mfcc = np.load(mfcc_path)

        if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] < min_audio_frames:
            return False
        if mfcc.ndim != 2 or mfcc.shape[0] != 13 or mfcc.shape[1] < min_audio_frames:
            return False
        if mel.shape[1] != mfcc.shape[1]:
            return False
    except Exception:
        return False

    return True


# ---------------------------------------
# Process one GRID mpg
# ---------------------------------------
def process_one_mpg(
    mpg_path,
    detector,
    out_root,
    roi_size=96,
    smooth_alpha=0.4,
    min_side_px=20,
    sr=16000,
    n_fft_audio=512,
    hop_length_audio=160,   # 10ms => 100Hz at 16k
    win_length_audio=400,   # 25ms
    n_mels=80,
    n_mfcc=13,
    min_roi_frames=5,
):
    mpg_path = str(mpg_path)

    parent = os.path.basename(os.path.dirname(mpg_path))          # s1_processed
    speaker = parent.replace("_processed", "")
    clip_id = os.path.splitext(os.path.basename(mpg_path))[0]     # bbaf2n

    sample_dir = os.path.join(out_root, speaker, clip_id)
    roi_dir = os.path.join(sample_dir, "roi")

    wav_path = os.path.join(sample_dir, "audio.wav")
    mel_path = os.path.join(sample_dir, "mel_80_100hz.npy")
    mfcc_path = os.path.join(sample_dir, "mfcc_13_100hz.npy")

    meta_path = os.path.join(sample_dir, "meta.json")
    if os.path.isfile(meta_path) and is_success(sample_dir, min_roi_frames=min_roi_frames, min_audio_frames=20):
        return True

    try:
        os.makedirs(sample_dir, exist_ok=True)

        cap = cv2.VideoCapture(mpg_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        meta = {
            "dataset": "GRID",
            "orig_path": mpg_path,
            "speaker": speaker,
            "clip_id": clip_id,

            # video
            "video_fps": float(fps) if fps and fps > 0 else 25.0,
            "frame_count_reported": int(frame_count),
            "roi_dir": "roi",
            "roi_size": int(roi_size),
            "roi_gray": True,

            # audio raw
            "sr": int(sr),
            "wav_path": "audio.wav",

            # audio feature params (100Hz)
            "audio_feature_fps": 100,
            "n_fft_audio": int(n_fft_audio),
            "hop_length_audio": int(hop_length_audio),
            "win_length_audio": int(win_length_audio),
            "n_mels": int(n_mels),
            "n_mfcc": int(n_mfcc),

            # feature paths
            "mel_path": "mel_80_100hz.npy",
            "mfcc_path": "mfcc_13_100hz.npy",
        }
        write_meta(sample_dir, meta)

        # 1) lips ROI
        lip_stats = extract_lips(
            detector,
            mpg_path,
            roi_dir,
            size=roi_size,
            smooth_alpha=smooth_alpha,
            min_side_px=min_side_px,
        )
        meta["lip_stats"] = lip_stats
        meta["num_frames_saved"] = int(lip_stats.get("n_saved", 0))
        write_meta(sample_dir, meta)

        # 2) audio.wav (ground truth)
        extract_audio_wav(mpg_path, wav_path, sr=sr)

        # 3) single audio load from audio.wav
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        wav = wav.astype("float32")
        meta["num_audio_samples"] = int(wav.shape[0])

        # 4) mel (db) + save
        mel_db = compute_mel_from_wav(
            wav,
            sr=sr,
            n_fft=n_fft_audio,
            hop_length=hop_length_audio,
            win_length=win_length_audio,
            n_mels=n_mels,
            to_db=True,
        )
        np.save(mel_path, mel_db)

        # 5) mfcc from mel_db + save
        mfcc = compute_mfcc_from_mel_db(mel_db, n_mfcc=n_mfcc)
        np.save(mfcc_path, mfcc)

        # meta shapes
        meta["mel_shape"] = [int(mel_db.shape[0]), int(mel_db.shape[1])]
        meta["mfcc_shape"] = [int(mfcc.shape[0]), int(mfcc.shape[1])]
        meta["audio_frames"] = int(mel_db.shape[1])
        write_meta(sample_dir, meta)

        if not is_success(sample_dir, min_roi_frames=min_roi_frames, min_audio_frames=20):
            raise RuntimeError("NOT success criteria")

        return True

    except Exception as e:
        shutil.rmtree(sample_dir, ignore_errors=True)
        print(f"✘ FAILED {mpg_path}: {e}")
        return False


# ---------------------------------------
# Main
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    prep = cfg.get("prepare", {})

    mode = prep.get("mode", "grid")
    out_root = prep.get("out", None)
    grid_root = prep.get("grid_root", None)

    if mode != "grid":
        raise ValueError(f"This script version supports only prepare.mode=grid (got {mode})")
    if out_root is None:
        raise ValueError("YAML prepare.out 이 필요합니다.")
    if grid_root is None:
        raise ValueError("YAML prepare.grid_root 이 필요합니다. (GRID raw data root)")

    limit = prep.get("limit", None)
    min_roi_frames = int(prep.get("min_lip_frames", 5))

    # ROI
    roi_size = int(prep.get("roi_size", 96))  
    smooth_alpha = float(prep.get("smooth_alpha", 0.4))
    min_side_px = int(prep.get("min_side_px", 20))

    # Audio raw
    sr = int(prep.get("sr", 16000))

    # Audio feature (100Hz)
    n_fft_audio = int(prep.get("n_fft_audio", 512))
    hop_length_audio = int(prep.get("hop_length_audio", 160))
    win_length_audio = int(prep.get("win_length_audio", 400))
    n_mels = int(prep.get("n_mels", 80))
    n_mfcc = int(prep.get("n_mfcc", 13))

    os.makedirs(out_root, exist_ok=True)

    mpg_files = sorted(glob.glob(os.path.join(grid_root, "**", "*.mpg"), recursive=True))
    if limit:
        mpg_files = mpg_files[: int(limit)]

    print(f"[GRID] Found mpg files: {len(mpg_files)}")
    print(f"[GRID] out_root: {out_root}")
    print(f"[GRID] roi_size={roi_size}, smooth_alpha={smooth_alpha}, sr={sr}")
    print(f"[GRID] audio_feat: n_fft={n_fft_audio}, hop={hop_length_audio}, win={win_length_audio} (100Hz)")

    detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=0)

    ok = 0
    for mpg_path in tqdm(mpg_files):
        if process_one_mpg(
            mpg_path,
            detector,
            out_root=out_root,
            roi_size=roi_size,
            smooth_alpha=smooth_alpha,
            min_side_px=min_side_px,
            sr=sr,
            n_fft_audio=n_fft_audio,
            hop_length_audio=hop_length_audio,
            win_length_audio=win_length_audio,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            min_roi_frames=min_roi_frames,
        ):
            ok += 1

    print(f"Done. SUCCESS={ok}/{len(mpg_files)}")


if __name__ == "__main__":
    main()
