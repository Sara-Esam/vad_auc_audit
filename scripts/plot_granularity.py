#!/usr/bin/env python3
"""Forest plot from evaluate_granularity results.json."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vad_eval.plot_granularity import load_granularity_results, plot_forest  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True, help="results.json from evaluate_granularity")
    ap.add_argument("--groups", type=Path, default=None,
                    help="optional JSON: {group_name: [model_tags,...]}")
    ap.add_argument("--display", type=Path, default=None, help="optional JSON tag->display name")
    ap.add_argument("--out-prefix", default="prediction_granularity_forest")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    args = ap.parse_args()

    data = load_granularity_results(args.results)
    if args.groups:
        with open(args.groups) as fh:
            groups = OrderedDict(json.load(fh))
    else:
        groups = OrderedDict([("models", sorted(data.keys()))])
    display = {}
    if args.display:
        with open(args.display) as fh:
            display = json.load(fh)

    png, pdf = plot_forest(data, groups, display=display, out_prefix=args.out_prefix, out_dir=args.out)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
