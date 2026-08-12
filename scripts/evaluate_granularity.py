#!/usr/bin/env python3
"""Compute global / per-category / within-video AUC with paired bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vad_eval.granularity import (  # noqa: E402
    align_units_by_ids,
    count_resolved,
    paired_bootstrap,
    point_metrics,
)
from vad_eval.io import (  # noqa: E402
    collect_video_units,
    load_model_arrays,
    official_test_ids,
)


def load_models(cfg_path, tags=None):
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    models = cfg.get("models", cfg)
    out = {}
    for tag, entry in models.items():
        if tags and tag not in tags:
            continue
        pred = entry.get("pred")
        if not pred or not Path(pred).is_file():
            print(f"[skip] {tag}: missing pred {pred}", file=sys.stderr)
            continue
        out[tag] = entry
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="JSON mapping tag -> {pred, gt, ...}")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--annotation", default=None, help="UCF_Annotation.txt for test split")
    ap.add_argument("--train-gt", default=None, help="optional train GT nested dict")
    ap.add_argument("--scope", choices=("test", "full"), default="test")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("outputs/granularity"))
    args = ap.parse_args()

    models = load_models(args.models, args.tags)
    if not models:
        sys.exit("no models loaded")

    test_ids = None
    if args.scope == "test":
        if not args.annotation:
            sys.exit("--annotation required for scope=test")
        test_ids = official_test_ids(args.annotation)

    units = {}
    for tag, entry in models.items():
        flat_s, flat_g = load_model_arrays(
            entry["pred"],
            gt_path=entry.get("gt"),
            train_gt_path=entry.get("train_gt", args.train_gt),
            annotation_path=args.annotation,
        )
        u = collect_video_units(flat_s, flat_g)
        if test_ids is not None:
            u = [x for x in u if x["id"] in test_ids]
        units[tag] = u
        print(f"[{tag}] {len(u)} videos")

    packs, shared = align_units_by_ids(units)
    print(f"shared support: {len(shared)} videos")

    rows_by = {}
    for tag, pack in packs.items():
        pt = point_metrics(pack)
        rows_by[tag] = {
            "model": tag,
            **{k: pt[k] for k in ("global", "per_category", "within_video", "delta_wv", "n_wv_videos", "n_categories")},
        }
        if pt["global"] is not None and pt["within_video"] is not None:
            print(
                f"  {tag}: G={pt['global']:.4f}  C={pt['per_category']:.4f}  "
                f"WV={pt['within_video']:.4f}  d={pt['delta_wv']:.4f}"
            )
        else:
            print(f"  {tag}: incomplete")

    paired = {}
    if args.n_boot > 0 and len(packs) >= 1:
        ci, paired = paired_bootstrap(packs, n_boot=args.n_boot, seed=args.seed)
        for tag in packs:
            for k in ("global", "per_category", "within_video", "delta_wv"):
                lo, hi = ci[tag][k]
                rows_by[tag][f"{k}_ci_lo"] = lo
                rows_by[tag][f"{k}_ci_hi"] = hi
        for metric in ("global", "per_category", "within_video"):
            n_res, n_tot = count_resolved(paired, metric)
            print(f"resolved {metric}: {n_res} / {n_tot}")

    rows = list(rows_by.values())

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"granularity_{args.scope}.csv"
    fields = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if paired:
        with open(args.out / f"paired_diffs_{args.scope}.json", "w") as fh:
            json.dump(paired, fh, indent=2)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
