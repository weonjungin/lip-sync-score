# /home/jiweon/projects/lip-sync-score/scripts/eval_offset_curve_hdtf.py

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml
import cv2

from lipsyncscore.models.syncnet_like import SyncNetLike
from lipsyncscore.models.modified.syncnet_temporal import SyncNetTemporal
from lipsyncscore.models.modified.syncnet_crossattn import SyncNetCrossAttn
from lipsyncscore.models.baselines.syncnet_python_wrapper import SyncNetPythonWrapper


# -------------------------------------------------
# Model builder (copied from train.py / eval_offset_curve.py)
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
    
    elif model_name == "syncnet_python":
        ckpt_path = model_cfg.get("ckpt", None)
        init_from_pretrained = bool(model_cfg.get("init_from_pretrained", True))

        if init_from_pretrained:
            if ckpt_path is None:
                raise ValueError(
                    "model.ckpt is required when init_from_pretrained=True "
                    "for model.name == 'syncnet_python'"
                )
            model = SyncNetPythonWrapper(
                ckpt_path=ckpt_path,
                load_pretrained=True,
            ).to(device)
        else:
            model = SyncNetPythonWrapper(
                ckpt_path=None,
                load_pretrained=False,
            ).to(device)

        return model_name, model

    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported: syncnet_like, syncnet_temporal, syncnet_crossattn"
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


def l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize embeddings to unit length along dim=1.
    This makes score = dot(l2norm(v), l2norm(a)) ~= cosine similarity.
    """
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)


def _get_first(meta: dict, keys, meta_path: Path):
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    raise KeyError(
        f"Missing keys {keys} in meta: {meta_path}\n"
        f"Available keys: {list(meta.keys())}"
    )


def scan_utt_dirs_flat(processed_root: Path):
    """
    HDTF processed format:
      processed_root/<sample_id>/meta.json
      processed_root/<sample_id>/roi/*.png
    """
    utts = []
    for d in sorted(processed_root.glob("*")):
        if not d.is_dir():
            continue
        if (d / "meta.json").exists() and (d / "roi").exists():
            utts.append(d)
    return utts


def speaker_id_from_name(name: str) -> str:
    m = re.search(r"Radio(\d+)", name)
    if m:
        return f"Radio{m.group(1)}"
    return "Unknown"


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
    for p in roi_files[t0 : t0 + N]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read ROI image: {p}")
        frames.append(img.astype(np.float32) / 255.0)

    lips = np.stack(frames, axis=0)  # (N,H,W)
    return lips.astype(np.float32)


def load_audio_feature(utt_dir: Path, meta: dict, meta_path: Path, audio_type: str):
    audio_type = str(audio_type).lower()

    if audio_type == "mel":
        rel = _get_first(meta, ["mel_path", "mel"], meta_path)
    elif audio_type == "mfcc":
        rel = _get_first(meta, ["mfcc_path", "mfcc"], meta_path)
    else:
        raise ValueError(f"Unsupported audio_type: {audio_type}")

    feat_path = utt_dir / rel
    return np.load(feat_path, mmap_mode="r")


def audio_window(feat, t0, off, N, a_per_v):
    """
    feat: (C, Taudio)  e.g. mel=(80, T), mfcc=(13, T) or similar
    """
    win_len = max(1, int(round(N * a_per_v)))
    start = int(round((t0 + off) * a_per_v))
    end = start + win_len

    C = feat.shape[0]
    out = np.zeros((C, win_len), dtype=np.float32)

    s0 = max(start, 0)
    s1 = min(end, feat.shape[1])
    if s1 > s0:
        out[:, (s0 - start):(s1 - start)] = feat[:, s0:s1]

    return out


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


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip_bad_meta", action="store_true")

    # Optional: evaluate only these utt dirs (one per line, relative or absolute)
    ap.add_argument(
        "--utts_list",
        type=str,
        default=None,
        help="Text file with one utt_dir per line (absolute or relative to processed_root).",
    )

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
    # model checkpoint path
    # -----------------------------
    out_dir = Path(cfg["eval"].get("save_dir", cfg["train"]["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_type = str(data_cfg.get("audio_type", "mel")).lower()

    model_cfg = cfg["model"]
    model_name_cfg = str(model_cfg.get("name", "syncnet_like"))

    if model_name_cfg == "syncnet_python" and bool(model_cfg.get("init_from_pretrained", True)):
        ckpt_path = Path(model_cfg["ckpt"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing pretrained checkpoint: {ckpt_path}")
        load_external_pretrained = True
    else:
        ckpt_path = out_dir / "checkpoints" / "best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        load_external_pretrained = False

    print(f"[INFO] out_dir   = {out_dir}")
    print(f"[INFO] ckpt_path = {ckpt_path}")
    print(f"[INFO] audio_type={audio_type}")

    # -----------------------------
    # N guard
    # -----------------------------
    model_cfg = cfg["model"]
    N_model = int(model_cfg.get("in_frames", samp["N"]))
    N_samp = int(samp["N"])
    if N_model != N_samp:
        raise ValueError(
            f"N mismatch: model.in_frames={N_model} vs data.sampling.N={N_samp}. "
            f"Make them equal for eval."
        )
    N = N_model

    # -----------------------------
    # build/load model
    # -----------------------------
    model_name, model = build_model(model_cfg, N=N, device="cpu")
    print(f"[INFO] model_name={model_cfg.get('name')}")
    if model_cfg.get("name") == "syncnet_temporal":
        print(f"[INFO] model_cfg.pooling={model_cfg.get('pooling', 'mean')}")
        print(f"[INFO] model_cfg.temporal={model_cfg.get('temporal', {})}")

    if load_external_pretrained:
        print("[INFO] using external pretrained SyncNet checkpoint directly from model.ckpt")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        if model_name == "syncnet_python":
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[INFO] syncnet_python load_state_dict strict=False")
            print(f"[INFO] missing keys: {len(missing)}")
            print(f"[INFO] unexpected keys: {len(unexpected)}")
        else:
            model.load_state_dict(state, strict=True)
            
    model.to(device)
    model.eval()

    # -----------------------------
    # utts
    # -----------------------------
    processed_root = Path(eval_cfg.get("processed_root", data_cfg["processed_root"]))
    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root not found: {processed_root}")

    all_utts = scan_utt_dirs_flat(processed_root)

    # optional filter by utts_list
    if args.utts_list is not None:
        keep = []
        p_list = Path(args.utts_list)
        lines = [ln.strip() for ln in p_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines:
            p = Path(ln)
            if not p.is_absolute():
                p = (processed_root / p).resolve()
            if p.exists():
                keep.append(p)
        all_utts = [u for u in all_utts if u.resolve() in set([k.resolve() for k in keep])]
        print(f"[INFO] utts_list filter: {len(all_utts)} utts kept")

    random.shuffle(all_utts)
    num_utts = int(eval_cfg["num_utts"])
    utts = all_utts[:num_utts]
    print(f"[INFO] utts scanned={len(all_utts)} eval_utts={len(utts)} root={processed_root}")

    offsets = list(
        range(
            eval_cfg["offset"]["min"],
            eval_cfg["offset"]["max"] + 1,
            eval_cfg["offset"]["step"],
        )
    )

    score_sum = np.zeros(len(offsets), dtype=np.float64)
    score_sq = np.zeros(len(offsets), dtype=np.float64)
    peak0 = 0
    peak_pm1 = 0

    used_utts = 0
    skipped_utts = 0

    # ---- output filenames (avoid overwrite across delays)
    split_tag = str(eval_cfg.get("split", "hdtf")).lower()
    if args.global_delay_ms is None:
        delay_tag_frames = int(args.global_delay_frames)
        tag = f"{split_tag}_d{delay_tag_frames:+d}f"
    else:
        tag = f"{split_tag}_dms{args.global_delay_ms:.1f}"

    out_csv = out_dir / f"eval_offset_curve_hdtf_{tag}.csv"
    bad_csv = out_dir / f"eval_offset_curve_hdtf_{tag}_bad_utts.csv"
    summary_csv = out_dir / f"eval_summary_hdtf_{tag}.csv"

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
    printed_mpv = False

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

        # ---- HDTF coverage filter ----
        nf = int(meta.get("num_frames_saved", 0) or 0)
        fc = int(meta.get("frame_count_reported", 0) or 0)

        if fc > 0:
            coverage = nf / fc
            if coverage < 0.9:  # ← 0.85~0.95 사이로 조절 가능
                skipped_utts += 1
                wb.writerow([str(utt), str(meta_path), f"low_coverage: {coverage:.3f}"])
                continue

        # Load mel + compute fps/sr/hop + m_per_v
        try:
            feat = load_audio_feature(utt, meta, meta_path, audio_type=audio_type)

            fps = float(_get_first(meta, ["video_fps", "fps"], meta_path))
            sr = float(_get_first(meta, ["sr", "audio_sr", "sample_rate"], meta_path))
            hop = float(_get_first(meta, ["hop_length_audio", "hop_length", "hop"], meta_path))

            if fps <= 0 or sr <= 0 or hop <= 0:
                raise ValueError(f"Non-positive fps/sr/hop: fps={fps}, sr={sr}, hop={hop}")

            a_per_v = (sr / hop) / fps

            if not printed_mpv:
                print(
                    f"[INFO] meta fps={fps} sr={sr} hop={hop} => a_per_v={a_per_v:.4f} "
                    f"(audio_C={feat.shape[0]} audio_T={feat.shape[1]})"
                )
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
            tag2 = f"{split_tag}_d{global_delay_frames:+d}f"
            out_csv2 = out_dir / f"eval_offset_curve_hdtf_{tag2}.csv"
            bad_csv2 = out_dir / f"eval_offset_curve_hdtf_{tag2}_bad_utts.csv"
            summary_csv2 = out_dir / f"eval_summary_hdtf_{tag2}.csv"

            f_out.close()
            f_bad.close()
            out_csv = out_csv2
            bad_csv = bad_csv2
            summary_csv = summary_csv2
            f_out, f_bad, w, wb = _open_writers(out_csv, bad_csv)

            need_reopen_for_ms = False

        roi_files = _list_roi_files(utt)
        if len(roi_files) == 0:
            skipped_utts += 1
            wb.writerow([str(utt), str(meta_path), "no_roi_files"])
            if not args.skip_bad_meta:
                raise RuntimeError(f"No ROI files: {utt}")
            continue

        T_v = len(roi_files)
        curves = []

        for _ in range(int(eval_cfg["samples_per_utt"])):
            t0 = sample_t0_biased(
                T_v=T_v,
                N=N,
                margin=float(samp["margin_trim"]),
                p_back=float(samp["p_back"]),
                rng=rng,
            )
            if t0 is None:
                continue
            if t0 + N > T_v:
                continue

            try:
                lips = load_roi_window_png(roi_files, t0, N)
            except Exception as e:
                wb.writerow([str(utt), str(meta_path), f"roi_read_error: {repr(e)}"])
                lips = None

            if lips is None:
                continue

            lips_t = torch.from_numpy(lips).unsqueeze(0).to(device)

            with torch.no_grad():
                if hasattr(model, "forward_lip"):
                    v = model.forward_lip(lips_t)
                else:
                    raise AttributeError(f"{type(model).__name__} has no forward_lip")
                v = l2norm(v)  # normalize (cosine metric)

            scores = []
            for off in offsets:
                off_total = off + global_delay_frames
                aud_win = audio_window(feat, t0, off_total, N, a_per_v)
                aud_t = torch.from_numpy(aud_win).unsqueeze(0).to(device)

                with torch.no_grad():
                    if hasattr(model, "forward_audio"):
                        a = model.forward_audio(aud_t)
                    elif hasattr(model, "forward_aud"):
                        a = model.forward_aud(aud_t)
                    else:
                        raise AttributeError(
                            f"{type(model).__name__} has neither forward_audio nor forward_aud"
                        )
                    a = l2norm(a)  # normalize (cosine metric)

                s = float((v * a).sum().item())
                scores.append(s)
                w.writerow([str(utt), off, s])

            curves.append(scores)

        if len(curves) == 0:
            skipped_utts += 1
            wb.writerow([str(utt), str(meta_path), "no_valid_samples_in_utt"])
            if not args.skip_bad_meta:
                raise RuntimeError(f"No valid samples in utt: {utt}")
            continue

        mean_curve = np.mean(np.array(curves, dtype=np.float64), axis=0)
        score_sum += mean_curve
        score_sq += mean_curve**2
        used_utts += 1

        peak_off = offsets[int(np.argmax(mean_curve))]
        if peak_off == 0:
            peak0 += 1
        if abs(peak_off) <= 1:
            peak_pm1 += 1

        if ui % 200 == 0:
            spk = speaker_id_from_name(utt.name)
            print(
                f"[{ui}/{len(utts)}] used={used_utts} skipped={skipped_utts} "
                f"(global_delay={global_delay_frames:+d}f) example={utt.name} spk={spk}"
            )

    f_out.close()
    f_bad.close()

    if used_utts == 0:
        raise RuntimeError(
            f"No utts evaluated successfully. "
            f"Check {bad_csv} for reasons. processed_root={processed_root}"
        )

    mean = score_sum / used_utts
    std = np.sqrt(score_sq / used_utts - mean**2)

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
        wsum.writerow(["score(0)", score0])
        wsum.writerow(["offset_margin", offset_margin])
        wsum.writerow(["peak_sharpness", peak_sharpness])
        wsum.writerow(["mean_peak_offset", mean_peak_offset])

        wsum.writerow(["peak@0_acc", peak0 / used_utts])
        wsum.writerow(["peak@±1_acc", peak_pm1 / used_utts])

        wsum.writerow(["used_utts", used_utts])
        wsum.writerow(["skipped_utts", skipped_utts])
        wsum.writerow(["global_delay_frames", global_delay_frames])
        if args.global_delay_ms is not None:
            wsum.writerow(["global_delay_ms", float(args.global_delay_ms)])
        wsum.writerow(["bad_utts_csv", str(bad_csv)])

    print("=== EVAL DONE (HDTF) ===")
    print("saved:", out_csv)
    print("saved:", summary_csv)
    if skipped_utts > 0:
        print("bad utts:", bad_csv)


if __name__ == "__main__":
    main()