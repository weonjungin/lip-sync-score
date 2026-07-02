"""
train_hdtf.py

SyncNetTemporal HDTF 학습 스크립트.
train_progressive.py와 독립적으로 동작.

실행:
    PYTHONPATH=src python scripts/train_hdtf.py --config configs/expP3.yaml
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from lipsyncscore.data.dataset_hdtf import DatasetHDTF
from lipsyncscore.models.modified.syncnet_temporal import SyncNetTemporal
from lipsyncscore.loss.contrastive import SyncNetContrastiveLoss
from lipsyncscore.loss.margin_ranking import SyncNetMarginRankingLoss
from lipsyncscore.loss.infonce import SyncNetInfoNCELoss


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


def batch_to_device(batch: dict, device: str) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def save_ckpt(
    path: Path,
    model, optimizer, scaler,
    epoch: int, step: int, best_val: float, cfg: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":         epoch,
            "step":          step,
            "best_val_loss": best_val,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scaler":        scaler.state_dict() if scaler is not None else None,
            "config":        cfg,
        },
        str(path),
    )


def build_model(model_cfg: dict, N: int, device: str) -> SyncNetTemporal:
    temporal_cfg    = model_cfg.get("temporal", {}) or {}
    pooling         = str(model_cfg.get("pooling", "mean"))
    lip_in_channels = int(model_cfg.get("lip_in_channels", 4))
    emb_dim         = int(model_cfg.get("emb_dim", 256))
    in_frames       = int(model_cfg.get("in_frames", N))

    model = SyncNetTemporal(
        in_frames       = in_frames,
        emb_dim         = emb_dim,
        temporal_cfg    = temporal_cfg if temporal_cfg else None,
        pooling         = pooling,
        lip_in_channels = lip_in_channels,
    ).to(device)

    return model


@torch.no_grad()
def evaluate(model, criterion, loader, device: str, use_amp: bool) -> dict:
    model.eval()
    loss_sum = dpos_sum = dneg_sum = 0.0
    n = 0

    for batch in loader:
        batch   = batch_to_device(batch, device)
        lips    = batch["lips"]
        pos_mel = batch["pos_mel"]
        neg_mel = batch["neg_mel"]

        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(use_amp and device == "cuda")):
            v     = model.forward_lip(lips)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    seed   = int(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device", "auto"))
    set_seed(seed)

    data_cfg   = cfg["data"]
    model_cfg  = cfg["model"]
    loss_cfg   = cfg["loss"]
    train_cfg  = cfg["train"]
    samp       = data_cfg.get("sampling", {})
    loader_cfg = data_cfg.get("loader", {})

    N           = int(samp.get("N", 16))
    batch_size  = int(train_cfg.get("batch_size", 64))
    num_workers = int(loader_cfg.get("num_workers", 4))
    pin_memory  = bool(loader_cfg.get("pin_memory", True)) and (device == "cuda")
    drop_last   = bool(loader_cfg.get("drop_last", True))

    out_dir = Path(train_cfg.get("out_dir", "logs/hdtf")).expanduser()
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    # ── Dataset ──────────────────────────────
    print("[INFO] DatasetHDTF 사용", flush=True)

    train_ds = DatasetHDTF(
        latent_root       = data_cfg["latent_root"],
        data_root         = data_cfg["data_root"],
        N                 = N,
        min_gap           = int(samp.get("min_gap", 5)),
        max_gap           = int(samp.get("max_gap", 100)),
        split_file        = data_cfg.get("train_split", None),
        samples_per_video = int(samp.get("samples_per_video", 200)),
    )
    val_ds = DatasetHDTF(
        latent_root       = data_cfg["latent_root"],
        data_root         = data_cfg["data_root"],
        N                 = N,
        min_gap           = int(samp.get("min_gap", 5)),
        max_gap           = int(samp.get("max_gap", 100)),
        split_file        = data_cfg.get("val_split", None),
        samples_per_video = int(samp.get("samples_per_video", 200)),
    )

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

    # ── Model ────────────────────────────────
    model = build_model(model_cfg, N=N, device=device)

    # ── Loss ─────────────────────────────────
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

    # ── Optimizer ────────────────────────────
    lr        = float(train_cfg.get("lr", 1e-4))
    wd        = float(train_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    use_amp = bool(train_cfg.get("amp", False)) and (device == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Resume ───────────────────────────────
    resume      = train_cfg.get("resume_ckpt", "")
    start_epoch = 1
    global_step = 0
    best_val    = float("inf")

    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))
        best_val    = float(ckpt.get("best_val_loss", best_val))
        print(f"[RESUME] {resume}  start_epoch={start_epoch}  step={global_step}  best_val={best_val:.4f}")

    # ── 로그 헤더 ────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[Data]   type=hdtf  train={len(train_ds)}  val={len(val_ds)}  "
          f"N={N}  min_gap={samp.get('min_gap',5)}  max_gap={samp.get('max_gap',100)}", flush=True)
    print(f"[Model]  name=syncnet_temporal  "
          f"total={total_params:,}  trainable={trainable_params:,}  "
          f"pooling={model_cfg.get('pooling','mean')}  "
          f"temporal={model_cfg.get('temporal',{}).get('type','none')}", flush=True)
    print(f"[Recipe] optimizer=AdamW  lr={lr}  wd={wd}  loss={loss_name}  "
          f"batch={batch_size}  epochs={train_cfg.get('epochs',30)}  "
          f"amp={use_amp}  out={out_dir}", flush=True)
    print(f"{'─'*80}", flush=True)

    # ── CSV log ──────────────────────────────
    csv_path = out_dir / "train_log.csv"
    new_csv  = not csv_path.exists()
    fcsv     = open(csv_path, "a", newline="")
    writer   = csv.writer(fcsv)
    if new_csv:
        writer.writerow([
            "epoch", "step",
            "train_loss", "train_dpos", "train_dneg",
            "val_loss",   "val_dpos",   "val_dneg",
        ])

    log_every        = int(train_cfg.get("log_every", 200))
    epochs           = int(train_cfg.get("epochs", 30))
    save_every_epoch = bool(train_cfg.get("save_every_epoch", True))

    # ── Train loop ───────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = dpos_sum = dneg_sum = 0.0
        seen = 0

        for batch in train_loader:
            global_step += 1
            batch   = batch_to_device(batch, device)
            lips    = batch["lips"]
            pos_mel = batch["pos_mel"]
            neg_mel = batch["neg_mel"]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                v     = model.forward_lip(lips)
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
                    f"[E{epoch:02d} step {global_step}]  "
                    f"train_loss={loss_sum/seen:.4f}  "
                    f"d_pos={dpos_sum/seen:.4f}  d_neg={dneg_sum/seen:.4f}",
                    flush=True,
                )

        train_metrics = {
            "loss":  loss_sum  / seen,
            "d_pos": dpos_sum  / seen,
            "d_neg": dneg_sum  / seen,
        }
        val_metrics = evaluate(model, criterion, val_loader, device, use_amp)

        print(
            f"[E{epoch:02d}] TRAIN  loss={train_metrics['loss']:.4f}  "
            f"d_pos={train_metrics['d_pos']:.4f}  d_neg={train_metrics['d_neg']:.4f}",
            flush=True,
        )
        print(
            f"[E{epoch:02d}]   VAL  loss={val_metrics['loss']:.4f}  "
            f"d_pos={val_metrics['d_pos']:.4f}  d_neg={val_metrics['d_neg']:.4f}",
            flush=True,
        )

        writer.writerow([
            epoch, global_step,
            train_metrics["loss"], train_metrics["d_pos"], train_metrics["d_neg"],
            val_metrics["loss"],   val_metrics["d_pos"],   val_metrics["d_neg"],
        ])
        fcsv.flush()

        if save_every_epoch:
            save_ckpt(
                out_dir / "checkpoints" / f"epoch_{epoch:03d}.pth",
                model, optimizer, scaler, epoch, global_step, best_val, cfg,
            )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_ckpt(
                out_dir / "checkpoints" / "best.pth",
                model, optimizer, scaler, epoch, global_step, best_val, cfg,
            )
            print(f"[SAVE] best.pth  best_val_loss={best_val:.4f}", flush=True)

    fcsv.close()
    print("[DONE] training finished.", flush=True)
    print(f"[INFO] best_val_loss: {best_val:.4f}", flush=True)
    print(f"[INFO] logs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()