#!/usr/bin/env python3
"""Pairwise model comparison at one granularity (resolution audit).

A pair is resolved when the 95% CI of the paired-bootstrap AUC difference
excludes zero. Select --granularity {global,per_category,within_video}.

Same --source scores|probe interface as evaluate_granularity.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vad_eval.features import load_flat_gt, load_probe_features  # noqa: E402
from vad_eval.granularity import align_packs, build_probe_records  # noqa: E402
from vad_eval.io import collect_video_units, load_model_arrays, official_test_ids  # noqa: E402
from vad_eval.resolution import GRANULARITIES, resolve_at_granularity  # noqa: E402


def _load_cfg(path, tags=None):
    with open(path) as fh:
        cfg = json.load(fh)
    models = cfg.get("models", cfg)
    if tags:
        models = {t: models[t] for t in tags if t in models}
    return models


def _packs_scores(models, annotation, train_gt, scope):
    test_ids = official_test_ids(annotation) if scope == "test" else None
    units = {}
    for tag, entry in models.items():
        flat_s, flat_g = load_model_arrays(
            entry["pred"],
            gt_path=entry.get("gt"),
            train_gt_path=entry.get("train_gt", train_gt),
            annotation_path=annotation,
        )
        u = collect_video_units(flat_s, flat_g)
        if test_ids is not None:
            u = [x for x in u if x["id"] in test_ids]
        units[tag] = u
    return align_packs(units, source="scores")[0]


def _packs_probe(models, annotation, gt):
    recs_by = {}
    for tag, entry in models.items():
        probe = entry.get("probe") or entry.get("features")
        gt_path = entry.get("gt") or gt
        ann = entry.get("annotation") or annotation
        gt_array = load_flat_gt(gt_path, ann)
        feats, gts = load_probe_features(
            probe, gt_array, ann,
            file_suffix=entry.get("file_suffix"),
            frame_level=bool(entry.get("frame_level", False)),
        )
        proto_feats = proto_gts = None
        if entry.get("train_probe") and entry.get("train_gt") and entry.get("train_ann"):
            train_gt = load_flat_gt(entry["train_gt"], entry["train_ann"])
            proto_feats, proto_gts = load_probe_features(
                entry["train_probe"], train_gt, entry["train_ann"],
                file_suffix=entry.get("file_suffix"),
                frame_level=bool(entry.get("frame_level", False)),
            )
        recs_by[tag] = build_probe_records(feats, gts, proto_feats, proto_gts)
    return align_packs(recs_by, source="probe")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("scores", "probe"), default="scores")
    ap.add_argument("--models", required=True)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--granularity", choices=GRANULARITIES, required=True)
    ap.add_argument("--annotation", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--train-gt", default=None)
    ap.add_argument("--scope", choices=("test", "full"), default="test")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("outputs/resolution"))
    args = ap.parse_args()

    models = _load_cfg(args.models, args.tags)
    if args.source == "scores":
        packs = _packs_scores(models, args.annotation, args.train_gt, args.scope)
    else:
        packs = _packs_probe(models, args.annotation, args.gt)

    result = resolve_at_granularity(
        packs, args.granularity, n_boot=args.n_boot, seed=args.seed
    )
    print(
        f"{args.granularity}: resolved {result['n_resolved']} / {result['n_pairs']} "
        f"(spread={100 * result['spread']:.2f} pts)"
        if result["spread"] is not None
        else f"{args.granularity}: resolved {result['n_resolved']} / {result['n_pairs']}"
    )
    for key, row in sorted(result["paired"].items(), key=lambda kv: -(abs(kv[1]["diff_mean"] or 0))):
        if not row["resolved"]:
            continue
        print(
            f"  {row['winner']} ≻ "
            f"{row['model_b'] if row['winner'] == row['model_a'] else row['model_a']}  "
            f"Δ={100 * row['diff_mean']:+.2f} "
            f"[{100 * row['ci_lo']:+.2f}, {100 * row['ci_hi']:+.2f}]"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"resolution_{args.granularity}.json", "w") as fh:
        json.dump(result, fh, indent=2)

    rows = []
    for row in result["paired"].values():
        rows.append(row)
    if rows:
        path = args.out / f"resolution_{args.granularity}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
