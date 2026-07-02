# /home/jiweon/projects/lip-sync-score/scripts/train.py

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from lipsyncscore.data.dataset_grid import DatasetGRID
from lipsyncscore.models.syncnet_like import SyncNetLike
from lipsyncscore.models.modified.syncnet_temporal import SyncNetTemporal
from lipsyncscore.models.modified.syncnet_crossattn import SyncNetCrossAttn
from lipsyncscore.loss.contrastive import SyncNetContrastiveLoss
from lipsyncscore.loss.margin_ranking import SyncNetMarginRankingLoss
from lipsyncscore.loss.infonce import SyncNetInfoNCELoss
from lipsyncscore.models.baselines.syncnet_python_wrapper import SyncNetPythonWrapper


def load_yaml(path: str) -> dict:
    p = Path(path).expanduser()
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def scan_utt_dirs(processed_root: Path) -> List[Path]:
    utt_dirs = []

    # case 1: utt dirs are directly under processed_root  (e.g. HDTF)
    for d in sorted(processed_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "meta.json").exists() and (d / "roi").is_dir():
            utt_dirs.append(d)

    if utt_dirs:
        return utt_dirs

    # case 2: speaker/utt structure  (e.g. GRID)
    for spk in sorted(processed_root.iterdir()):
        if not spk.is_dir():
            continue
        for utt in sorted(spk.iterdir()):
            if not utt.is_dir():
                continue
            if (utt / "meta.json").exists() and (utt / "roi").is_dir():
                utt_dirs.append(utt)

    return utt_dirs

def get_speaker_id_from_utt_dir(utt_dir: Path) -> str:
    """
    Example:
      WDA_AndyKim_000_5670_5750 -> WDA_AndyKim
      RD_Radio1_000_0_80        -> RD_Radio1

    Rule:
      use everything except the last 3 underscore-separated fields
      (clip_idx, start, end)
    """
    name = utt_dir.name
    parts = name.split("_")
    if len(parts) >= 4:
        return "_".join(parts[:-3])
    return name

def speaker_id_from_utt(utt_dir: Path) -> str:
    return utt_dir.parent.name


def split_by_speaker(
    utt_dirs: List[Path],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[Path], List[Path], List[Path], Dict[str, List[str]]]:
    spk_to_utts: Dict[str, List[Path]] = {}
    for u in utt_dirs:
        spk = get_speaker_id_from_utt_dir(u)
        spk_to_utts.setdefault(spk, []).append(u)

    speakers = sorted(spk_to_utts.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))

    train_spk = speakers[:n_train]
    val_spk = speakers[n_train:n_train + n_val]
    test_spk = speakers[n_train + n_val:]

    train_utts = [u for spk in train_spk for u in spk_to_utts[spk]]
    val_utts = [u for spk in val_spk for u in spk_to_utts[spk]]
    test_utts = [u for spk in test_spk for u in spk_to_utts[spk]]

    split_info = {"train_speakers": train_spk, "val_speakers": val_spk, "test_speakers": test_spk}
    return train_utts, val_utts, test_utts, split_info


def batch_to_device(batch: dict, device: str):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# -----------------------
# Cross-attn compatibility helpers
# -----------------------
def _extract_lip_emb_from_forward(out) -> torch.Tensor:
    """
    Accept multiple possible forward outputs:
      - dict with "lip_emb"
      - tuple/list (lip_emb, audio_emb, score) or (lip_emb, audio_emb)
      - single tensor -> assume it's lip_emb
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        if "lip_emb" in out and torch.is_tensor(out["lip_emb"]):
            return out["lip_emb"]
        # fall back: first tensor value
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise ValueError("Forward returned dict but no tensor found for lip_emb.")
    if isinstance(out, (tuple, list)):
        if len(out) >= 1 and torch.is_tensor(out[0]):
            return out[0]
        raise ValueError("Forward returned tuple/list but first element is not a tensor.")
    raise TypeError(f"Unsupported forward output type: {type(out)}")


def _get_fused_lip_emb(model, lips: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
    """
    For syncnet_crossattn:
      priority:
        1) model.forward_fused_lip(lips, mel) if exists
        2) model(lips, mel)
        3) model.forward(lips, mel)
    """
    if hasattr(model, "forward_fused_lip") and callable(getattr(model, "forward_fused_lip")):
        out = model.forward_fused_lip(lips, mel)
        if out.dim() != 2:
            raise ValueError(f"forward_fused_lip must return (B,D). got {tuple(out.shape)}")
        return out

    # try __call__
    try:
        out = model(lips, mel)
        emb = _extract_lip_emb_from_forward(out)
        if emb.dim() != 2:
            raise ValueError(f"model(lips, mel) lip_emb must be (B,D). got {tuple(emb.shape)}")
        return emb
    except TypeError:
        pass

    if hasattr(model, "forward") and callable(getattr(model, "forward")):
        out = model.forward(lips, mel)
        emb = _extract_lip_emb_from_forward(out)
        if emb.dim() != 2:
            raise ValueError(f"model.forward(lips, mel) lip_emb must be (B,D). got {tuple(emb.shape)}")
        return emb

    raise AttributeError(
        "SyncNetCrossAttn must provide forward_fused_lip(lips, mel) "
        "or support calling model(lips, mel) / model.forward(lips, mel)."
    )


def _get_audio_emb(model, mel: torch.Tensor) -> torch.Tensor:
    """
    audio emb is still computed from audio-only encoder in most setups.
    Must return (B,D).
    """
    if not hasattr(model, "forward_audio"):
        raise AttributeError("Model must implement forward_audio(mel) -> (B,D).")
    out = model.forward_audio(mel)
    if not torch.is_tensor(out) or out.dim() != 2:
        raise ValueError(f"forward_audio must return (B,D). got {type(out)} shape={getattr(out, 'shape', None)}")
    return out


@torch.no_grad()
def evaluate(model, criterion, loader, device: str, use_amp: bool, model_name: str) -> dict:
    model.eval()
    loss_sum = 0.0
    dpos_sum = 0.0
    dneg_sum = 0.0
    n = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        lips = batch["lips"]
        pos_mel = batch["pos_mel"]
        neg_mel = batch["neg_mel"]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device == "cuda")):
            if model_name == "syncnet_crossattn":
                v_pos = _get_fused_lip_emb(model, lips, pos_mel)
                a_pos = _get_audio_emb(model, pos_mel)
                a_neg = _get_audio_emb(model, neg_mel)
                loss, stats = criterion(v_pos, a_pos, a_neg)
            else:
                v = model.forward_lip(lips)
                a_pos = model.forward_audio(pos_mel)
                a_neg = model.forward_audio(neg_mel)
                loss, stats = criterion(v, a_pos, a_neg)

        bs = lips.size(0)
        loss_sum += float(loss.item()) * bs
        if "d_pos" in stats:
            dpos_sum += float(stats["d_pos"]) * bs
            dneg_sum += float(stats["d_neg"]) * bs
        else:
            dpos_sum += float(stats["s_pos"]) * bs
            dneg_sum += float(stats["s_neg"]) * bs
        n += bs

    return {"loss": loss_sum / n, "d_pos": dpos_sum / n, "d_neg": dneg_sum / n}


def save_ckpt(path: Path, model, optimizer, scaler, epoch: int, step: int, best_val: float, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": cfg,
        },
        str(path),
    )


def build_model(model_cfg: dict, N: int, device: str):
    model_name = str(model_cfg.get("name", "syncnet_like"))

    if model_name == "syncnet_like":
        model = SyncNetLike(
            in_frames=int(model_cfg.get("in_frames", N)),
            emb_dim=int(model_cfg.get("emb_dim", 256)),
        ).to(device)
        return model_name, model

    if model_name == "syncnet_temporal":
        if SyncNetTemporal is None:
            raise ImportError(
                "SyncNetTemporal import failed. "
                "Check module path: lipsyncscore/models/modified/syncnet_temporal.py"
            )

        temporal_cfg = model_cfg.get("temporal", {}) or {}
        pooling = str(model_cfg.get("pooling", "mean"))

        # NOTE: SyncNetTemporal __init__이 pooling 인자를 받는다는 전제
        model = SyncNetTemporal(
            in_frames=int(model_cfg.get("in_frames", N)),
            emb_dim=int(model_cfg.get("emb_dim", 256)),
            temporal_cfg=temporal_cfg if temporal_cfg else None,
            pooling=pooling,
        ).to(device)

        return model_name, model

    if model_name == "syncnet_crossattn":
        ca = model_cfg.get("crossattn", {}) or {}
        model = SyncNetCrossAttn(
            in_frames=int(model_cfg.get("in_frames", N)),
            emb_dim=int(model_cfg.get("emb_dim", 256)),
            n_heads=int(ca.get("n_heads", 4)),
            dropout=float(ca.get("dropout", 0.0)),
            use_ffn=bool(ca.get("use_ffn", True)),
            ffn_mult=int(ca.get("ffn_mult", 4)),
            pool=str(ca.get("pool", "mean")),
            l2_norm=bool(ca.get("l2_norm", True)),
            num_layers=int(ca.get("num_layers", 1)),
        ).to(device)
        return model_name, model
    
    if model_name == "syncnet_python":
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

    raise ValueError(f"Unknown model.name={model_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device", "auto"))
    set_seed(seed)

    data_cfg = cfg["data"]
    split_cfg = cfg.get("split", {"type": "speaker_disjoint", "train_ratio": 0.85, "val_ratio": 0.10})
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    train_cfg = cfg["train"]

    out_dir = Path(train_cfg.get("out_dir", "logs/train_syncnetlike")).expanduser()
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # save config snapshot
    (out_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    processed_root = Path(data_cfg["processed_root"]).expanduser()
    utt_dirs = scan_utt_dirs(processed_root)

    print(f"[INFO] utts: {len(utt_dirs)}")


    def read_name_list(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())


    train_list_path = data_cfg.get("train_list", None)
    val_list_path = data_cfg.get("val_list", None)

    if train_list_path is not None and val_list_path is not None:
        print("[INFO] Using fixed train/val split from list files")

        train_names = read_name_list(train_list_path)
        val_names = read_name_list(val_list_path)

        train_utts = [p for p in utt_dirs if p.name in train_names]
        val_utts = [p for p in utt_dirs if p.name in val_names]
        test_utts = []

        train_name_set = set(p.name for p in train_utts)
        val_name_set = set(p.name for p in val_utts)

        missing_train = sorted(train_names - train_name_set)
        missing_val = sorted(val_names - val_name_set)
        overlap = sorted(train_name_set & val_name_set)

        if overlap:
            raise ValueError(f"Overlap between train/val lists: {len(overlap)} examples")

        print(f"[Split] utts: train={len(train_utts)} val={len(val_utts)} test={len(test_utts)}")
        print(f"[Split] missing from train_list: {len(missing_train)}")
        print(f"[Split] missing from val_list  : {len(missing_val)}")

        split_info = {
            "type": "fixed_list",
            "train_list": str(train_list_path),
            "val_list": str(val_list_path),
            "num_total_utts": len(utt_dirs),
            "num_train_utts": len(train_utts),
            "num_val_utts": len(val_utts),
            "num_test_utts": len(test_utts),
            "missing_train": missing_train,
            "missing_val": missing_val,
        }

    else:
        # 기존 speaker split 유지
        if split_cfg.get("type", "speaker_disjoint") != "speaker_disjoint":
            raise ValueError("Only speaker_disjoint split is supported.")

        train_utts, val_utts, test_utts, split_info = split_by_speaker(
            utt_dirs,
            seed=seed,
            train_ratio=float(split_cfg.get("train_ratio", 0.85)),
            val_ratio=float(split_cfg.get("val_ratio", 0.10)),
        )

        print(
            f"[Split] speakers: train={len(split_info['train_speakers'])} "
            f"val={len(split_info['val_speakers'])} test={len(split_info['test_speakers'])}"
        )
        print(f"[Split] utts: train={len(train_utts)} val={len(val_utts)} test={len(test_utts)}")

    (out_dir / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    # dataset sampling params
    samp = data_cfg["sampling"]
    N = int(samp.get("N", 5))

    # ✅ NEW: hard negative sampling config (optional)
    neg_cfg = samp.get("neg_sampling", {}) or {}
    neg_hard_prob = float(neg_cfg.get("hard_prob", 0.0))
    neg_hard_range = neg_cfg.get("hard_range", None)  # expected [lo,hi] or null
    neg_avoid_zero = bool(neg_cfg.get("avoid_zero", True))

    if neg_hard_range is None:
        neg_hard_range_tuple = (None, None)
    else:
        # allow yaml list/tuple
        if not isinstance(neg_hard_range, (list, tuple)) or len(neg_hard_range) != 2:
            raise ValueError(f"data.sampling.neg_sampling.hard_range must be [lo,hi] or null. got: {neg_hard_range}")
        neg_hard_range_tuple = (int(neg_hard_range[0]), int(neg_hard_range[1]))

    print(f"[INFO] sampling.N={N} margin_trim={samp.get('margin_trim', 0.15)} p_back={samp.get('p_back', 0.65)}")
    print(f"[INFO] neg_sampling.hard_prob={neg_hard_prob} hard_range={neg_hard_range_tuple} avoid_zero={neg_avoid_zero}")

    train_ds = DatasetGRID(
        utt_dirs=train_utts,
        N=N,
        margin=float(samp.get("margin_trim", 0.15)),
        p_back=float(samp.get("p_back", 0.65)),
        return_debug=False,
        max_shift_sec=float(samp.get("max_shift_sec", 2.0)),
        min_sep_frames=int(samp.get("min_sep_frames", max(N, 5))),
        audio_type=str(data_cfg.get("audio_type", "mel")),

        neg_hard_prob=neg_hard_prob,
        neg_hard_range=neg_hard_range_tuple,
        neg_avoid_zero=neg_avoid_zero,
    )
    val_ds = DatasetGRID(
        utt_dirs=val_utts,
        N=N,
        margin=float(samp.get("margin_trim", 0.15)),
        p_back=float(samp.get("p_back", 0.65)),
        return_debug=False,
        max_shift_sec=float(samp.get("max_shift_sec", 2.0)),
        min_sep_frames=int(samp.get("min_sep_frames", max(N, 5))),
        audio_type=str(data_cfg.get("audio_type", "mel")),

        neg_hard_prob=neg_hard_prob,
        neg_hard_range=neg_hard_range_tuple,
        neg_avoid_zero=neg_avoid_zero,
    )

    loader_cfg = data_cfg.get("loader", {})
    num_workers = int(loader_cfg.get("num_workers", 6))
    pin_memory = bool(loader_cfg.get("pin_memory", True)) and (device == "cuda")
    drop_last = bool(loader_cfg.get("drop_last", True))

    batch_size = int(train_cfg.get("batch_size", 64))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # model/loss
    model_name, model = build_model(model_cfg, N=N, device=device)
    loss_name = loss_cfg.get("name", "contrastive")

    if loss_name == "contrastive":
        criterion = SyncNetContrastiveLoss(
            margin=float(loss_cfg.get("margin", 1.0)),
            lambda_pos=float(loss_cfg.get("lambda_pos", 1.0)),
            lambda_neg=float(loss_cfg.get("lambda_neg", 1.0)),
        )

    elif loss_name == "margin_ranking":
        criterion = SyncNetMarginRankingLoss(
            margin=float(loss_cfg.get("margin", 0.2))
        )

    elif loss_name == "infonce":
        criterion = SyncNetInfoNCELoss(
            temperature=float(loss_cfg.get("temperature", 0.07))
        )

    else:
        raise ValueError(f"Unknown loss: {loss_name}")


    # optim
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    use_amp = bool(train_cfg.get("amp", False)) and (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # resume (optional)
    resume = train_cfg.get("resume_ckpt", "")
    start_epoch = 1
    global_step = 0
    best_val = float("inf")
    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val_loss", best_val))
        print(f"[RESUME] {resume} (start_epoch={start_epoch}, step={global_step}, best_val={best_val:.4f})")

    print(f"[INFO] device={device} amp={use_amp}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] model_name={model_name}")

    print(f"[INFO] model_cfg.name={model_cfg.get('name')}")
    if model_name == "syncnet_temporal":
        print(f"[INFO] model_cfg.pooling={model_cfg.get('pooling', 'mean')}")
        print(f"[INFO] model_cfg.temporal={model_cfg.get('temporal', {})}")

    # csv log
    csv_path = out_dir / "train_log.csv"
    new_csv = not csv_path.exists()
    fcsv = open(csv_path, "a", newline="")
    writer = csv.writer(fcsv)
    if new_csv:
        writer.writerow(["epoch", "step", "train_loss", "train_dpos", "train_dneg", "val_loss", "val_dpos", "val_dneg"])

    log_every = int(train_cfg.get("log_every", 200))
    epochs = int(train_cfg.get("epochs", 20))
    save_every_epoch = bool(train_cfg.get("save_every_epoch", True))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        dpos_sum = 0.0
        dneg_sum = 0.0
        seen = 0

        for batch in train_loader:
            global_step += 1
            batch = batch_to_device(batch, device)
            lips = batch["lips"]
            pos_mel = batch["pos_mel"]
            neg_mel = batch["neg_mel"]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                if model_name == "syncnet_crossattn":
                    v_pos = _get_fused_lip_emb(model, lips, pos_mel)
                    a_pos = _get_audio_emb(model, pos_mel)
                    a_neg = _get_audio_emb(model, neg_mel)
                    loss, stats = criterion(v_pos, a_pos, a_neg)
                else:
                    v = model.forward_lip(lips)
                    a_pos = model.forward_audio(pos_mel)
                    a_neg = model.forward_audio(neg_mel)
                    loss, stats = criterion(v, a_pos, a_neg)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = lips.size(0)
            loss_sum += float(loss.item()) * bs

            if "d_pos" in stats:
                dpos_sum += float(stats["d_pos"]) * bs
                dneg_sum += float(stats["d_neg"]) * bs
            else:
                dpos_sum += float(stats["s_pos"]) * bs
                dneg_sum += float(stats["s_neg"]) * bs
                
            seen += bs

            if log_every > 0 and global_step % log_every == 0:
                print(
                    f"[E{epoch:02d} step {global_step}] "
                    f"train_loss={loss_sum/seen:.4f} d_pos={dpos_sum/seen:.4f} d_neg={dneg_sum/seen:.4f}"
                )

        train_metrics = {"loss": loss_sum / seen, "d_pos": dpos_sum / seen, "d_neg": dneg_sum / seen}
        val_metrics = evaluate(model, criterion, val_loader, device, use_amp, model_name=model_name)

        print(
            f"[E{epoch:02d}] TRAIN loss={train_metrics['loss']:.4f} "
            f"d_pos={train_metrics['d_pos']:.4f} d_neg={train_metrics['d_neg']:.4f}"
        )
        print(
            f"[E{epoch:02d}]   VAL loss={val_metrics['loss']:.4f} "
            f"d_pos={val_metrics['d_pos']:.4f} d_neg={val_metrics['d_neg']:.4f}"
        )

        writer.writerow(
            [
                epoch,
                global_step,
                train_metrics["loss"],
                train_metrics["d_pos"],
                train_metrics["d_neg"],
                val_metrics["loss"],
                val_metrics["d_pos"],
                val_metrics["d_neg"],
            ]
        )
        fcsv.flush()

        if save_every_epoch:
            save_ckpt(out_dir / "checkpoints" / f"epoch_{epoch:03d}.pth", model, optimizer, scaler, epoch, global_step, best_val, cfg)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_ckpt(out_dir / "checkpoints" / "best.pth", model, optimizer, scaler, epoch, global_step, best_val, cfg)
            print(f"[SAVE] best.pth (best_val_loss={best_val:.4f})")

    fcsv.close()
    print("[DONE] training finished.")
    print(f"[INFO] best_val_loss: {best_val:.4f}")
    print(f"[INFO] logs: {out_dir}")


if __name__ == "__main__":
    main()
