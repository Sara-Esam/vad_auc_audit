#!/usr/bin/env python3
"""Three-granularity AUC: global, per-category, within-video.

Supports --source scores (model predictions) or --source probe (pre-classifier
embeddings with zero-shot prototype scoring).
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
from vad_eval.granularity import (  # noqa: E402
    align_packs,
    build_probe_records,
    point_metrics,
)
from vad_eval.io import (  # noqa: E402
    collect_video_units,
    load_model_arrays,
    official_test_ids,
)
from vad_eval.resolution import paired_bootstrap  # noqa: E402


def _load_models_cfg(path, tags=None):
    with open(path) as fh:
        cfg = json.load(fh)
    models = cfg.get("models", cfg)
    out = {}
    for tag, entry in models.items():
        if tags and tag not in tags:
            continue
        out[tag] = entry
    return out


def _run_scores(args):
    models = _load_models_cfg(args.models, args.tags)
    test_ids = official_test_ids(args.annotation) if args.scope == "test" else None
    units = {}
    for tag, entry in models.items():
        pred = entry.get("pred")
        if not pred or not Path(pred).is_file():
            print(f"[skip] {tag}: missing pred", file=sys.stderr)
            continue
        flat_s, flat_g = load_model_arrays(
            pred,
            gt_path=entry.get("gt"),
            train_gt_path=entry.get("train_gt", args.train_gt),
            annotation_path=args.annotation,
        )
        u = collect_video_units(flat_s, flat_g)
        if test_ids is not None:
            u = [x for x in u if x["id"] in test_ids]
        units[tag] = u
        print(f"[{tag}] {len(u)} videos")
    packs, shared = align_packs(units, source="scores")
    print(f"shared support: {len(shared)}")
    return packs


def _run_probe(args):
    models = _load_models_cfg(args.models, args.tags)
    recs_by = {}
    for tag, entry in models.items():
        probe = entry.get("probe") or entry.get("features")
        gt_path = entry.get("gt") or args.gt
        ann = entry.get("annotation") or args.annotation
        if not probe or not gt_path or not ann:
            print(f"[skip] {tag}: need probe, gt, annotation", file=sys.stderr)
            continue
        gt_array = load_flat_gt(gt_path, ann)
        feats, gts = load_probe_features(
            probe,
            gt_array,
            ann,
            file_suffix=entry.get("file_suffix"),
            frame_level=bool(entry.get("frame_level", False)),
        )
        proto_feats = proto_gts = None
        if entry.get("train_probe") and entry.get("train_gt") and entry.get("train_ann"):
            train_gt = load_flat_gt(entry["train_gt"], entry["train_ann"])
            proto_feats, proto_gts = load_probe_features(
                entry["train_probe"],
                train_gt,
                entry["train_ann"],
                file_suffix=entry.get("file_suffix"),
                frame_level=bool(entry.get("frame_level", False)),
            )
        recs = build_probe_records(feats, gts, proto_feats, proto_gts)
        recs_by[tag] = recs
        print(f"[{tag}] {len(recs)} videos (probe)")
    packs, shared = align_packs(recs_by, source="probe")
    print(f"shared support: {len(shared)}")
    return packs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("scores", "probe"), default="scores")
    ap.add_argument("--models", required=True, help="JSON config of models")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--annotation", default=None)
    ap.add_argument("--gt", default=None, help="flat GT for probe mode")
    ap.add_argument("--train-gt", default=None)
    ap.add_argument("--scope", choices=("test", "full"), default="test")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("outputs/granularity"))
    args = ap.parse_args()

    if args.source == "scores":
        if args.scope == "test" and not args.annotation:
            sys.exit("--annotation required for scope=test")
        packs = _run_scores(args)
    else:
        if not args.annotation and not args.models:
            sys.exit("probe mode needs annotation paths in config or --annotation")
        packs = _run_probe(args)

    if not packs:
        sys.exit("no models loaded")

    per_model = {}
    rows = []
    for tag, pack in packs.items():
        pt = point_metrics(pack)
        per_model[tag] = {"point": pt, "ci": {}}
        rows.append({"model": tag, **{k: pt[k] for k in ("global", "per_category", "within_video", "delta_wv")}})
        print(
            f"  {tag}: G={pt['global']:.4f}  C={pt['per_category']:.4f}  "
            f"WV={pt['within_video']:.4f}"
            if pt["global"] is not None
            else f"  {tag}: incomplete"
        )

    paired = {}
    if args.n_boot > 0:
        ci, paired = paired_bootstrap(packs, n_boot=args.n_boot, seed=args.seed)
        for tag in packs:
            per_model[tag]["ci"] = {
                k: list(ci[tag][k]) for k in ("global", "per_category", "within_video")
            }
            for k, (lo, hi) in ci[tag].items():
                rows_by = {r["model"]: r for r in rows}
                rows_by[tag][f"{k}_ci_lo"] = lo
                rows_by[tag][f"{k}_ci_hi"] = hi

    args.out.mkdir(parents=True, exist_ok=True)
    scope = args.scope if args.source == "scores" else "probe"
    csv_path = args.out / f"granularity_{args.source}_{scope}.csv"
    fields = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    results = {
        "source": args.source,
        "scope": scope,
        "per_model": {t: {"test" if scope == "test" else scope: per_model[t]} for t in per_model},
        "paired": paired,
    }
    with open(args.out / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {csv_path}")
    print(f"wrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
