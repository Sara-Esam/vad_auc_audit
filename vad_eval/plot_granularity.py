"""Forest plot for three-granularity AUC with bootstrap CIs."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLUMNS = ["Global", "Per-category", "Within-video", r"$\Delta_{\mathrm{wv}}$"]
COL_KEYS = ["global", "per_category", "within_video", "delta_wv"]

BAND_COLOR = "#B7D3E8"
BAND_EDGE = "#5A87A8"
INT_COLOR = "#333333"
POINT_COLOR = "#C0392B"
POINT_EDGE = "white"


def load_granularity_results(path):
    """
    Accept either:
      * results.json from evaluate_granularity (per_model[tag][scope].point/ci)
      * flat CSV-style dict list is not used here
    """
    with open(path) as fh:
        blob = json.load(fh)

    if "per_model" in blob:
        per_model = blob["per_model"]
        scope = "test" if any("test" in v for v in per_model.values()) else next(
            iter(next(iter(per_model.values())).keys())
        )
        data = {}
        for tag, scopes in per_model.items():
            entry_src = scopes.get(scope, scopes)
            pt = entry_src.get("point", entry_src)
            ci = entry_src.get("ci", {})
            entry = {}
            for k in COL_KEYS:
                p = pt.get(k)
                lo_hi = ci.get(k)
                if lo_hi is None:
                    lo = pt.get(f"{k}_ci_lo")
                    hi = pt.get(f"{k}_ci_hi")
                else:
                    lo, hi = lo_hi
                if p is None or lo is None or hi is None:
                    entry[k] = None
                else:
                    entry[k] = (100 * lo, 100 * p, 100 * hi)
            data[tag] = entry
        return data

    # flat {tag: {global, ..., global_ci_lo, ...}}
    data = {}
    for tag, row in blob.items():
        entry = {}
        for k in COL_KEYS:
            p = row.get(k)
            lo, hi = row.get(f"{k}_ci_lo"), row.get(f"{k}_ci_hi")
            if p is None or lo is None or hi is None:
                entry[k] = None
            else:
                entry[k] = (100 * lo, 100 * p, 100 * hi)
        data[tag] = entry
    return data


def _axis_range(data, tags, colkey, pad_frac=0.06):
    los, his = [], []
    for m in tags:
        trip = data.get(m, {}).get(colkey)
        if trip is None:
            continue
        lo, _, hi = trip
        los.append(lo)
        his.append(hi)
    if not los:
        return 0.0, 1.0
    lo_all, hi_all = min(los), max(his)
    pad = max(0.8, pad_frac * (hi_all - lo_all))
    return lo_all - pad, hi_all + pad


def _point_span(data, tags, colkey):
    pts = [data[m][colkey][1] for m in tags if data.get(m, {}).get(colkey) is not None]
    if not pts:
        return 0.0, 1.0
    return min(pts), max(pts)


def draw_panel(ax, data, tags, colkey, colname, display):
    lo_ax, hi_ax = _axis_range(data, tags, colkey)
    pt_lo, pt_hi = _point_span(data, tags, colkey)
    band_width = pt_hi - pt_lo
    n = len(tags)
    y_positions = np.arange(n)

    ax.axvspan(pt_lo, pt_hi, facecolor=BAND_COLOR, edgecolor="none", alpha=0.55, zorder=0)
    for x in (pt_lo, pt_hi):
        ax.axvline(x, color=BAND_EDGE, linewidth=0.6, alpha=0.7, zorder=1)

    for y, tag in zip(y_positions, tags):
        trip = data.get(tag, {}).get(colkey)
        if trip is None:
            ax.text(0.5 * (lo_ax + hi_ax), y, "\u2014", ha="center", va="center",
                    fontsize=14, color="#888", zorder=3)
            continue
        lo, pt, hi = trip
        ax.hlines(y, lo, hi, color=INT_COLOR, linewidth=1.6, zorder=2, capstyle="round")
        ax.vlines([lo, hi], y - 0.18, y + 0.18, color=INT_COLOR, linewidth=1.0, zorder=2)
        ax.plot(pt, y, marker="o", markersize=8.5,
                markerfacecolor=POINT_COLOR, markeredgecolor=POINT_EDGE,
                markeredgewidth=1.0, linestyle="none", zorder=4)

    ax.set_xlim(lo_ax, hi_ax)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(y_positions)
    ax.tick_params(axis="both", which="both", length=3, labelsize=16)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    unit = " pts" if colkey == "delta_wv" else ""
    ax.set_title(f"{colname}\nmodel spread: {band_width:.2f}{unit}", fontsize=18, pad=10)
    ax.set_xlabel("AUC (%)" if colkey != "delta_wv" else r"$\Delta_{\mathrm{wv}}$ (pts)", fontsize=16)
    return band_width


def plot_forest(
    data,
    groups,
    display=None,
    out_prefix="prediction_granularity_forest",
    out_dir=".",
):
    """
    Parameters
    ----------
    data : {tag: {metric: (ci_lo, point, ci_hi) in percent}}
    groups : OrderedDict[group_name -> list of tags]
    display : optional {tag -> display name}
    """
    display = display or {}
    tags, seps, group_bounds = [], [], []
    for gname, models in groups.items():
        y0 = len(tags)
        tags.extend(models)
        group_bounds.append((gname, y0, len(tags) - 1))
        seps.append(len(tags) - 0.5)
    seps = seps[:-1]

    n_rows = len(tags)
    fig_h = max(5.6, 0.42 * n_rows + 2.4)
    fig, axes = plt.subplots(
        1, len(COL_KEYS), figsize=(14.8, fig_h), sharey=True,
        gridspec_kw={"wspace": 0.14, "left": 0.19, "right": 0.995, "top": 0.90, "bottom": 0.17},
    )
    for ax, ckey, cname in zip(axes, COL_KEYS, COLUMNS):
        draw_panel(ax, data, tags, ckey, cname, display)

    axes[0].set_yticks(np.arange(n_rows))
    axes[0].set_yticklabels([display.get(t, t) for t in tags], fontsize=16)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    for s in seps:
        for ax in axes:
            ax.axhline(s, color="#BBBBBB", linewidth=0.5, zorder=1)

    handles = [
        Patch(facecolor=BAND_COLOR, edgecolor=BAND_EDGE, label="min–max span of AUC across models"),
        Line2D([0], [0], color=INT_COLOR, linewidth=2.6, label="95% video-bootstrap CI"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=POINT_COLOR,
               markeredgecolor=POINT_EDGE, markeredgewidth=1.2, markersize=13, label="AUC"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=3, frameon=False, fontsize=19, handlelength=2.8, columnspacing=2.4)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{out_prefix}.png"
    pdf = out_dir / f"{out_prefix}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return str(png), str(pdf)
