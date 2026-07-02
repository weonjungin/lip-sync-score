# /home/jiweon/projects/lip-sync-score/scripts/eval_offset_curve.py

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
import cv2

from lipsyncscore.models.syncnet_like import SyncNetLike
from lipsyncscore.models.modified.syncnet_temporal import SyncNetTemporal
try:
    from lipsyncscore.models.modified.syncnet_crossattn import SyncNetCrossAttn
except ImportError:
    SyncNetCrossAttn = None


# -------------------------------------------------
# Model builder (copied from train.py)
# -------------------------------------------------
def build_model(model_cfg: dict, N: int, device: str):
    model_name = str(model_cfg.get("name", "syncnet_like"))

    if model_name == "syncnet_like":
        model = SyncNetLike(
            in_frames=int(model_cfg.get("in_frames", N)),
            emb_dim=int(model_cfg.get("emb_dim", 256)),
        ).to(device)
        return model_name, model

    elif model_name == "syncnet_temporal":
        temporal_cfg = model_cfg.get("temporal", {}) or {}
        pooling = str(model_cfg.get("pooling", "mean"))  

        model = SyncNetTemporal(
            in_frames=int(model_cfg.get("in_frames", N)),
            emb_dim=int(model_cfg.get("emb_dim", 256)),
            temporal_cfg=temporal_cfg if temporal_cfg else None,
            pooling=pooling,  
        ).to(device)
        return model_name, model

    elif model_name == "syncnet_crossattn":
        cfg = dict(model_cfg)
        cfg.pop("name", None)
        model = SyncNetCrossAttn(**cfg).to(device)
        return model_name, model

    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported: syncnet_like, syncnet_temporal"
        )


# -------------------------------------------------
# Utils
# -------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scan_utt_dirs(root: Path):
    utts = []
    for spk in sorted(root.glob("s*")):
        for utt in sorted(spk.glob("*")):
            if (utt / "meta.json").exists() and (utt / "roi").exists():
                utts.append(utt)
    return utts


def speaker_id_from_utt(utt_dir: Path) -> str:
    return utt_dir.parent.name


def _get_first(meta: dict, keys, meta_path: Path):
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    raise KeyError(
        f"Missing keys {keys} in meta: {meta_path}\n"
        f"Available keys: {list(meta.keys())}"
    )


# -------------------------------------------------
# Sampling
# -------------------------------------------------
def sample_t0_biased(T_v, N, margin, p_back, rng):
    keep_start = math.ceil(margin * T_v)
    keep_end = math.floor((1 - margin) * T_v) - 1
    valid_end = keep_end - (N - 1)
    if valid_end < keep_start:
        return None

    L = valid_end - keep_start + 1
    mid = keep_start + L // 2

    if rng.random() < p_back:
        return rng.randint(mid, valid_end)
    else:
        if mid - 1 < keep_start:
            return rng.randint(keep_start, valid_end)
        return rng.randint(keep_start, mid - 1)


def _list_roi_files(utt_dir: Path):
    roi_dir = utt_dir / "roi"
    pngs = sorted(roi_dir.glob("*.png"))
    if len(pngs) > 0:
        return pngs
    jpgs = sorted(roi_dir.glob("*.jpg"))
    if len(jpgs) > 0:
        return jpgs
    jpegs = sorted(roi_dir.glob("*.jpeg"))
    if len(jpegs) > 0:
        return jpegs
    return sorted(roi_dir.glob("*"))


def load_roi_window_png(roi_files, t0: int, N: int):
    """
    Returns:
      lips: (N, H, W) float32 in [0,1]
    """
    frames = []
    for p in roi_files[t0:t0 + N]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read ROI image: {p}")
        frames.append(img.astype(np.float32) / 255.0)

    lips = np.stack(frames, axis=0)  # (N,H,W)
    return lips.astype(np.float32)


def load_mel(utt_dir: Path, meta: dict, meta_path: Path):
    mel_rel = _get_first(meta, ["mel_path", "mel"], meta_path)
    mel_path = utt_dir / mel_rel
    return np.load(mel_path, mmap_mode="r")


def mel_window(mel, t0, off, N, m_per_v):
    """
    mel: (80, Tmel)
    t0: video frame index
    off: video-frame offset (can be negative)
    """
    mel_len = max(1, int(round(N * m_per_v)))
    start = int(round((t0 + off) * m_per_v))
    end = start + mel_len

    out = np.zeros((80, mel_len), dtype=np.float32)
    s0 = max(start, 0)
    s1 = min(end, mel.shape[1])
    if s1 > s0:
        out[:, (s0 - start):(s1 - start)] = mel[:, s0:s1]
    return out


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip_bad_meta", action="store_true")

    # global delay simulation
    ap.add_argument(
        "--global_delay_frames",
        type=int,
        default=0,
        help="Simulate global audio delay in *video frames*. (e.g., 12 ≈ 0.5s at 25fps)",
    )
    ap.add_argument(
        "--global_delay_ms",
        type=float,
        default=None,
        help="Simulate global audio delay in ms. Converted to frames using fps from meta.",
    )

    args = ap.parse_args()

    cfg = load_yaml(args.config)
    eval_cfg = cfg["eval"]
    data_cfg = cfg["data"]
    samp = data_cfg["sampling"]

    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    # -----------------------------
    # paths (ADD: print + exists check)
    # -----------------------------
    out_dir = Path(cfg["train"]["out_dir"])
    ckpt_path = out_dir / "checkpoints" / "best.pth"
    split_json = out_dir / "split.json"

    print(f"[INFO] out_dir    = {out_dir}")
    print(f"[INFO] ckpt_path  = {ckpt_path}")
    print(f"[INFO] split_json = {split_json}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not split_json.exists():
        raise FileNotFoundError(f"Missing split.json: {split_json}")

    split_info = json.load(open(split_json))
    split_name = eval_cfg["split"]
    spk_key = f"{split_name}_speakers"

    if spk_key in split_info:
        target_speakers = set(split_info[spk_key])
        print(f"[INFO] using speaker filter: {spk_key} ({len(target_speakers)} speakers)")
    elif split_name in ["train", "val", "test"]:
        raise KeyError(
            f"Missing required key '{spk_key}' in split.json. "
            f"Available keys: {list(split_info.keys())}"
        )
    else:
        target_speakers = None
        print(f"[WARN] {spk_key} not found in split.json; assuming cross-dataset eval")
        
    # -----------------------------
    # model (ADD: N mismatch guard + unify N)
    # -----------------------------
    model_cfg = cfg["model"]
    N_model = int(model_cfg.get("in_frames", samp["N"]))
    N_samp = int(samp["N"])
    if N_model != N_samp:
        raise ValueError(
            f"N mismatch: model.in_frames={N_model} vs data.sampling.N={N_samp}. "
            f"Make them equal for eval."
        )
    N = N_model  # unify

    model_name, model = build_model(model_cfg, N=N, device="cpu")

    print(f"[INFO] model_name={model_cfg.get('name')}")
    if model_cfg.get("name") == "syncnet_temporal":
        print(f"[INFO] model_cfg.pooling={model_cfg.get('pooling', 'mean')}")
        print(f"[INFO] model_cfg.temporal={model_cfg.get('temporal', {})}")
        
    # optional: quick info for temporal on/off
    if model_name == "syncnet_temporal":
        print("[INFO] temporal_lip:", getattr(model, "temporal_lip", None))
        print("[INFO] temporal_aud:", getattr(model, "temporal_aud", None))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # utts
    processed_root = Path(data_cfg["processed_root"])
    all_utts = scan_utt_dirs(processed_root)
    utts = [u for u in all_utts if speaker_id_from_utt(u) in target_speakers]
    random.shuffle(utts)
    utts = utts[: eval_cfg["num_utts"]]

    offsets = list(range(
        eval_cfg["offset"]["min"],
        eval_cfg["offset"]["max"] + 1,
        eval_cfg["offset"]["step"]
    ))

    score_sum = np.zeros(len(offsets), dtype=np.float64)
    score_sq = np.zeros(len(offsets), dtype=np.float64)

    peak0 = 0
    peak_pm1 = 0

    used_utts = 0
    skipped_utts = 0

    # ---- output filenames (avoid overwrite across delays)
    if args.global_delay_ms is None:
        delay_tag_frames = int(args.global_delay_frames)
        tag = f"{split_name}_d{delay_tag_frames:+d}f"
    else:
        tag = f"{split_name}_dms{args.global_delay_ms:.1f}"

    out_csv = out_dir / f"eval_offset_curve_{tag}.csv"
    bad_csv = out_dir / f"eval_offset_curve_{tag}_bad_utts.csv"
    summary_csv = out_dir / f"eval_summary_{tag}.csv"

    need_reopen_for_ms = args.global_delay_ms is not None

    def _open_writers(_out_csv: Path, _bad_csv: Path):
        _f_out = open(_out_csv, "w", newline="")
        _f_bad = open(_bad_csv, "w", newline="")
        _w = csv.writer(_f_out)
        _wb = csv.writer(_f_bad)
        _w.writerow(["utt", "offset", "score"])
        _wb.writerow(["utt", "meta_path", "reason"])
        return _f_out, _f_bad, _w, _wb

    f_out, f_bad, w, wb = _open_writers(out_csv, bad_csv)

    global_delay_frames = int(args.global_delay_frames)

    printed_mpv = False  # ADD: print m_per_v once

    for ui, utt in enumerate(utts):
        rng = random.Random(hash(str(utt)) & 0xFFFFFFFF)
        meta_path = utt / "meta.json"

        # Load meta
        try:
            meta = json.load(open(meta_path))
        except Exception as e:
            skipped_utts += 1
            wb.writerow([str(utt), str(meta_path), f"meta_load_error: {repr(e)}"])
            if not args.skip_bad_meta:
                raise
            continue

        # Load mel + compute fps/sr/hop + m_per_v
        try:
            mel = load_mel(utt, meta, meta_path)

            fps = float(_get_first(meta, ["video_fps", "fps"], meta_path))
            sr = float(_get_first(meta, ["sr", "audio_sr", "sample_rate"], meta_path))
            hop = float(_get_first(meta, ["hop_length_audio", "hop_length", "hop"], meta_path))

            if fps <= 0 or sr <= 0 or hop <= 0:
                raise ValueError(f"Non-positive fps/sr/hop: fps={fps}, sr={sr}, hop={hop}")

            m_per_v = (sr / hop) / fps

            if not printed_mpv:
                print(f"[INFO] meta fps={fps} sr={sr} hop={hop} => m_per_v={m_per_v:.4f} "
                      f"(mel_T={mel.shape[1]})")
                printed_mpv = True

        except Exception as e:
            skipped_utts += 1
            wb.writerow([str(utt), str(meta_path), f"meta_or_feature_error: {repr(e)}"])
            if not args.skip_bad_meta:
                raise
            continue

        # Resolve ms -> frames once (and reopen files with final tag)
        if need_reopen_for_ms:
            global_delay_frames = int(round((args.global_delay_ms / 1000.0) * fps))
            tag2 = f"{split_name}_d{global_delay_frames:+d}f"
            out_csv2 = out_dir / f"eval_offset_curve_{tag2}.csv"
            bad_csv2 = out_dir / f"eval_offset_curve_{tag2}_bad_utts.csv"
            summary_csv2 = out_dir / f"eval_summary_{tag2}.csv"

            f_out.close()
            f_bad.close()
            out_csv = out_csv2
            bad_csv = bad_csv2
            summary_csv = summary_csv2
            f_out, f_bad, w, wb = _open_writers(out_csv, bad_csv)

            need_reopen_for_ms = False

        # ROI files (png/jpg)
        roi_files = _list_roi_files(utt)
        if len(roi_files) == 0:
            skipped_utts += 1
            wb.writerow([str(utt), str(meta_path), "no_roi_files"])
            if not args.skip_bad_meta:
                raise RuntimeError(f"No ROI files: {utt}")
            continue

        T_v = len(roi_files)

        curves = []
        for _ in range(eval_cfg["samples_per_utt"]):
            t0 = sample_t0_biased(
                T_v=T_v,
                N=N,  # CHANGED: unify N
                margin=samp["margin_trim"],
                p_back=samp["p_back"],
                rng=rng,
            )
            if t0 is None:
                continue
            if t0 + N > T_v:  # CHANGED
                continue

            try:
                lips = load_roi_window_png(roi_files, t0, N)  # CHANGED
            except Exception as e:
                skipped_utts += 1
                wb.writerow([str(utt), str(meta_path), f"roi_read_error: {repr(e)}"])
                lips = None

            if lips is None:
                continue

            lips_t = torch.from_numpy(lips).unsqueeze(0).to(device)

            with torch.no_grad():
                v = model.forward_lip(lips_t)

            scores = []
            for off in offsets:
                off_total = off + global_delay_frames

                mel_win = mel_window(mel, t0, off_total, N, m_per_v)  # CHANGED
                mel_t = torch.from_numpy(mel_win).unsqueeze(0).to(device)
                with torch.no_grad():
                    a = model.forward_audio(mel_t)

                s = float((v * a).sum().item())
                scores.append(s)
                w.writerow([str(utt), off, s])  # save sweep offset (off)

            curves.append(scores)

        if len(curves) == 0:
            skipped_utts += 1
            wb.writerow([str(utt), str(meta_path), "no_valid_samples_in_utt"])
            if not args.skip_bad_meta:
                raise RuntimeError(f"No valid samples in utt: {utt}")
            continue

        mean_curve = np.mean(np.array(curves, dtype=np.float64), axis=0)
        score_sum += mean_curve
        score_sq += mean_curve ** 2
        used_utts += 1

        peak_off = offsets[int(np.argmax(mean_curve))]
        if peak_off == 0:
            peak0 += 1
        if abs(peak_off) <= 1:
            peak_pm1 += 1

        if ui % 200 == 0:
            print(f"[{ui}/{len(utts)}] used={used_utts} skipped={skipped_utts} (global_delay={global_delay_frames:+d}f)")

    f_out.close()
    f_bad.close()

    if used_utts == 0:
        raise RuntimeError(
            f"No utts evaluated successfully. "
            f"Check {bad_csv} for reasons. processed_root={processed_root}"
        )

    mean = score_sum / used_utts
    std = np.sqrt(score_sq / used_utts - mean ** 2)

    # ---- scalar metrics from the mean offset curve
    offsets_arr = np.asarray(offsets, dtype=int)
    if 0 not in offsets_arr:
        raise RuntimeError("offset=0 is required to compute score(0)/margin/sharpness")

    i0 = int(np.where(offsets_arr == 0)[0][0])
    score0 = float(mean[i0])

    mean_wo_0 = np.delete(mean, i0)
    offset_margin = float(score0 - float(np.mean(mean_wo_0)))
    peak_sharpness = float(score0 - float(np.max(mean_wo_0)))
    mean_peak_offset = int(offsets_arr[int(np.argmax(mean))])

    with open(summary_csv, "w", newline="") as f:
        wsum = csv.writer(f)
        wsum.writerow(["offset", "mean", "std"])
        for o, m, s in zip(offsets, mean, std):
            wsum.writerow([o, float(m), float(s)])

        wsum.writerow([])
        # curve-level scalars
        wsum.writerow(["score(0)", score0])
        wsum.writerow(["offset_margin", offset_margin])
        wsum.writerow(["peak_sharpness", peak_sharpness])
        wsum.writerow(["mean_peak_offset", mean_peak_offset])

        # utterance-level accuracies
        wsum.writerow(["peak@0_acc", peak0 / used_utts])
        wsum.writerow(["peak@±1_acc", peak_pm1 / used_utts])

        # bookkeeping
        wsum.writerow(["used_utts", used_utts])
        wsum.writerow(["skipped_utts", skipped_utts])
        wsum.writerow(["global_delay_frames", global_delay_frames])
        if args.global_delay_ms is not None:
            wsum.writerow(["global_delay_ms", float(args.global_delay_ms)])
        wsum.writerow(["bad_utts_csv", str(bad_csv)])

    print("=== EVAL DONE ===")
    print("saved:", out_csv)
    print("saved:", summary_csv)
    if skipped_utts > 0:
        print("bad utts:", bad_csv)


if __name__ == "__main__":
    main()
