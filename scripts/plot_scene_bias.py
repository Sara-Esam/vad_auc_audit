#!/usr/bin/env python3
"""Dot plot and heatmap from scene-bias CSV output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vad_eval.plot_scene_bias import (  # noqa: E402
    load_bias_table,
    plot_bias_dotplot,
    plot_bias_heatmap,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True,
                    help="long-form CSV from evaluate_scene_bias, or wide *_pooled table")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out-prefix", default="scene_bias_auc")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    args = ap.parse_args()

    pooled = load_bias_table(args.csv)
    models = args.models
    png, pdf = plot_bias_dotplot(
        pooled, models=models, out_prefix=f"{args.out_prefix}_dotplot", out_dir=args.out
    )
    print(f"wrote {png} / {pdf}")
    png, pdf = plot_bias_heatmap(
        pooled, models=models, out_prefix=f"{args.out_prefix}_heatmap", out_dir=args.out
    )
    print(f"wrote {png} / {pdf}")


if __name__ == "__main__":
    main()
