#!/usr/bin/env python3
"""Scene-conditioned AUC and Delta_scene (Supp. S1.2)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vad_eval.io import (  # noqa: E402
    attach_scene_membership,
    collect_video_units,
    load_model_arrays,
    load_scene_pairs,
)
from vad_eval.scene_conditioned import pair_matrix, scene_conditioned  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--train-gt", default=None)
    ap.add_argument("--annotation", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--pairs", type=Path, default=ROOT / "annotations" / "scene_pairs.json")
    ap.add_argument("--lists", type=Path, default=ROOT / "annotations" / "lists")
    ap.add_argument("--out", type=Path, default=Path("outputs/scene_conditioned"))
    ap.add_argument("--write-pair-matrix", action="store_true")
    args = ap.parse_args()

    flat_s, flat_g = load_model_arrays(
        args.pred, gt_path=args.gt, train_gt_path=args.train_gt, annotation_path=args.annotation
    )
    pairs = load_scene_pairs(args.pairs, args.lists)
    units = collect_video_units(flat_s, flat_g)
    attach_scene_membership(units, pairs)

    rows, matrix_rows = [], []
    for sp in pairs:
        m = scene_conditioned(units, sp["factor"])
        row = {
            "model": args.tag,
            "factor": sp["factor"],
            "pole_a": sp["pole_a"],
            "pole_b": sp["pole_b"],
            "n_dj": m["n_dj"],
            "ReferenceAUC": m["ReferenceAUC"],
            "AUC_A": m["AUC_A"],
            "AUC_B": m["AUC_B"],
            "SceneConditionedAUC": m["SceneConditionedAUC"],
            "Delta_scene": m["Delta_scene"],
        }
        rows.append(row)
        d = m["Delta_scene"]
        print(
            f"{sp['factor']:32s}  ref={m['ReferenceAUC']:.3f}  cond={m['SceneConditionedAUC']:.3f}  d={d:+.4f}"
            if d is not None
            else f"{sp['factor']:32s}  incomplete"
        )
        if args.write_pair_matrix:
            pm = pair_matrix(units, sp["factor"])
            if pm:
                matrix_rows.append({"model": args.tag, **pm})

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"scene_conditioned_{args.tag}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
    if matrix_rows:
        mpath = args.out / f"pair_matrix_{args.tag}.csv"
        with open(mpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(matrix_rows[0].keys()))
            w.writeheader()
            w.writerows(matrix_rows)
        print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
