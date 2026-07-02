import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def read_offset_curve_from_summary(summary_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads eval_summary_*.csv and extracts the offset->mean curve.
    Expected that the first two columns of curve rows are numeric: offset, mean, ...
    """
    offsets = []
    means = []

    with open(summary_csv, newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue

            key = row[0].strip()

            # header
            if key == "offset":
                continue

            # curve rows: first two cols numeric
            if len(row) >= 2 and _is_number(row[0]) and _is_number(row[1]):
                offsets.append(int(float(row[0])))
                means.append(float(row[1]))
                continue

            # ignore footer/other rows

    if not offsets:
        raise ValueError(f"No offset-curve rows found in: {summary_csv}")

    offsets = np.asarray(offsets, dtype=int)
    means = np.asarray(means, dtype=float)

    # sort by offset
    order = np.argsort(offsets)
    return offsets[order], means[order]


def compute_curve_metrics_from_summary(summary_csv: Path) -> Dict[str, float]:
    """
    Your definitions (matches the snippet you provided):
      - score(0)       = mean score at offset 0
      - offset_margin  = score(0) - mean(score(offset != 0))
      - peak_sharpness = score(0) - max(score(offset != 0))
    """
    offsets, means = read_offset_curve_from_summary(summary_csv)

    if 0 not in offsets:
        raise ValueError(f"offset=0 not found in curve rows: {summary_csv}")

    i0 = int(np.where(offsets == 0)[0][0])
    score_0 = float(means[i0])

    means_wo_0 = np.delete(means, i0)
    offset_margin = float(score_0 - float(np.mean(means_wo_0)))
    peak_sharpness = float(score_0 - float(np.max(means_wo_0)))

    return {
        "score(0)": score_0,
        "offset_margin": offset_margin,
        "peak_sharpness": peak_sharpness,
    }


def compute_peak_acc_from_raw(raw_csv: Path) -> Dict[str, float]:
    """
    Computes:
      - peak@0_acc: fraction of utts whose argmax offset == 0
      - peak@±1_acc: fraction of utts whose argmax offset in {-1,0,+1}

    IMPORTANT:
      If samples_per_utt > 1, raw CSV usually contains multiple rows per (utt, offset).
      We first aggregate scores per (utt, offset) by mean to form an utterance-level curve,
      then take argmax over offsets.

    Tie-breaking:
      1) higher score wins
      2) smaller |offset| wins
      3) smaller offset wins
    """
    df = pd.read_csv(raw_csv)

    required = {"utt", "offset", "score"}
    if not required.issubset(df.columns):
        raise ValueError(f"RAW CSV must contain {required}. Found: {list(df.columns)}")

    # numeric types
    df["offset"] = df["offset"].astype(int)
    df["score"] = df["score"].astype(float)

    # 1) aggregate within each (utt, offset)  (handles samples_per_utt > 1)
    agg = (
        df.groupby(["utt", "offset"], as_index=False)["score"]
          .mean()
    )

    # 2) pick best offset per utt with deterministic tie-break
    agg["abs_off"] = agg["offset"].abs()

    agg_sorted = agg.sort_values(
        ["utt", "score", "abs_off", "offset"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )

    best = agg_sorted.groupby("utt", as_index=False).first()

    peak0 = float((best["offset"] == 0).mean())
    peak_pm1 = float(best["offset"].isin([-1, 0, 1]).mean())

    return {
        "peak@0_acc": peak0,
        "peak@±1_acc": peak_pm1,
    }


def save_metrics_csv(metrics: Dict[str, float], out_csv: Path, *, split: Optional[str] = None) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = ["split", "score(0)", "offset_margin", "peak_sharpness", "peak@0_acc", "peak@±1_acc"]
    row = {
        "split": split if split is not None else "",
        "score(0)": metrics.get("score(0)", ""),
        "offset_margin": metrics.get("offset_margin", ""),
        "peak_sharpness": metrics.get("peak_sharpness", ""),
        "peak@0_acc": metrics.get("peak@0_acc", ""),
        "peak@±1_acc": metrics.get("peak@±1_acc", ""),
    }

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="path to eval_summary_*.csv")
    ap.add_argument("--raw", default=None, help="path to eval_offset_curve_*.csv (utt, offset, score) for peak metrics")
    ap.add_argument("--out_csv", default=None, help="optional path to save metrics as csv")
    ap.add_argument("--split", default=None, help="optional split label (e.g., test/val)")
    ap.add_argument("--print_json", action="store_true", help="also print metrics as json")
    args = ap.parse_args()

    summary_csv = Path(args.summary)

    # 1) curve metrics (always from summary)
    metrics = compute_curve_metrics_from_summary(summary_csv)

    # 2) peak metrics (compute from raw so it ALWAYS exists)
    if args.raw is not None:
        raw_csv = Path(args.raw)
        metrics.update(compute_peak_acc_from_raw(raw_csv))
    else:
        # no raw -> cannot compute peak metrics reliably
        pass

    # Console print (your preferred style)
    print(f"score(0)            = {metrics['score(0)']:.6f}")
    print(f"offset_margin       = {metrics['offset_margin']:.6f}")
    print(f"peak_sharpness      = {metrics['peak_sharpness']:.6f}")

    if "peak@0_acc" in metrics:
        print(f"peak@0_acc          = {metrics['peak@0_acc']:.4f}")
    else:
        print("peak@0_acc          = (missing)  -> provide --raw to compute")

    if "peak@±1_acc" in metrics:
        print(f"peak@±1_acc         = {metrics['peak@±1_acc']:.4f}")
    else:
        print("peak@±1_acc         = (missing)  -> provide --raw to compute")

    if args.print_json:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.out_csv is not None:
        save_metrics_csv(metrics, Path(args.out_csv), split=args.split)


if __name__ == "__main__":
    main()