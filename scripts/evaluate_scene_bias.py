#!/usr/bin/env python3
"""Scene bias-AUC (pooled + per-video with bootstrap)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vad_eval.io import load_model_arrays, load_scene_pairs  # noqa: E402
from vad_eval.scene_bias import factor_bias  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--train-gt", default=None)
    ap.add_argument("--annotation", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--pairs", type=Path, default=ROOT / "annotations" / "scene_pairs.json")
    ap.add_argument("--lists", type=Path, default=ROOT / "annotations" / "lists")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("outputs/scene_bias"))
    args = ap.parse_args()

    flat_s, flat_g = load_model_arrays(
        args.pred, gt_path=args.gt, train_gt_path=args.train_gt, annotation_path=args.annotation
    )
    pairs = load_scene_pairs(args.pairs, args.lists)
    rows = []
    for sp in pairs:
        r = factor_bias(sp["ids_a"], sp["ids_b"], flat_s, flat_g, n_boot=args.n_boot, seed=args.seed)
        row = {
            "model": args.tag,
            "factor": sp["factor"],
            "pole_a": sp["pole_a"],
            "pole_b": sp["pole_b"],
            **r,
        }
        rows.append(row)
        b, pb = r["bias_auc_video"], r["p_bias"]
        print(
            f"{sp['factor']:32s}  pooled={r['bias_auc_pooled']:.3f}  "
            f"video={b:.3f} [{r['bias_auc_video_ci_lo']:.3f},{r['bias_auc_video_ci_hi']:.3f}]  "
            f"p_bias={pb:.3f}"
            if b is not None and pb is not None
            else f"{sp['factor']:32s}  n/a"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"scene_bias_{args.tag}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.out / f"scene_bias_{args.tag}.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
