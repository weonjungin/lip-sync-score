import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


META_KEYS = {
    "score(0)",
    "offset_margin",
    "peak_sharpness",
    "mean_peak_offset",
    "peak@0_acc",
    "peak@±1_acc",
    "used_utts",
    "skipped_utts",
    "global_delay_frames",
    "global_delay_ms",
    "bad_utts_csv",
}


def load_curve(csv_path: str):
    """
    Supports:
    1) raw eval curve csv: columns like [utt, offset, score]
    2) summary csv: rows like [offset, mean, std] followed by meta rows
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # 먼저 pandas로 raw curve 형식 시도
    try:
        df = pd.read_csv(path)
        cols = set(df.columns)

        # raw per-utt curve csv
        if {"offset", "score"}.issubset(cols):
            curve = (
                df.groupby("offset", as_index=False)["score"]
                .mean()
                .sort_values("offset")
            )
            return curve["offset"].to_numpy(), curve["score"].to_numpy()

        # clean summary dataframe
        if {"offset", "mean"}.issubset(cols):
            df = df.dropna(subset=["offset", "mean"]).copy()

            # meta rows 제거
            df = df[~df["offset"].astype(str).isin(META_KEYS)]

            df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
            df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
            df = df.dropna(subset=["offset", "mean"]).sort_values("offset")

            return df["offset"].to_numpy(), df["mean"].to_numpy()
    except Exception:
        pass

    # fallback: csv.reader로 직접 summary 파싱
    offsets, means = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            key = str(row[0]).strip()
            if key == "offset":
                continue
            if key in META_KEYS:
                continue

            if len(row) < 2:
                continue

            try:
                off = int(float(row[0]))
                mean = float(row[1])
            except Exception:
                continue

            offsets.append(off)
            means.append(mean)

    if len(offsets) == 0:
        raise ValueError(f"Could not parse curve from: {path}")

    order = np.argsort(offsets)
    offsets = np.asarray(offsets)[order]
    means = np.asarray(means)[order]
    return offsets, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help='label=csv_path 형식. 예: expN3=logs/expN3/eval_summary....csv',
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Offset Curve Comparison")
    args = ap.parse_args()

    plt.figure(figsize=(11, 6.5))

    for item in args.inputs:
        if "=" not in item:
            raise ValueError(f"Expected label=csv_path format, got: {item}")

        label, csv_path = item.split("=", 1)
        offsets, scores = load_curve(csv_path)
        plt.plot(offsets, scores, linewidth=2.2, label=label)

    plt.title(args.title)
    plt.xlabel("Offset (frames)")
    plt.ylabel("Score")

    plt.ylim(0.0, 1.0)
    plt.xlim(-35, 35)
    plt.xticks(np.arange(-35, 36, 5))
    plt.grid(True, alpha=0.3)

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print("saved:", out_path)


if __name__ == "__main__":
    main()