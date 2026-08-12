"""Scene bias-AUC dot plot and heatmap."""

from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

CAPTURE_FACTORS = [
    "closeview_vs_wideview",
    "singleview_vs_multiview",
    "color_vs_grayscale",
    "highresolution_vs_lowresolution",
    "highmotion_vs_lowmotion",
    "highway_vs_sideroad",
]
CONTENT_FACTORS = [
    "outdoor_vs_indoor",
    "few-person_vs_crowd",
    "singleact_vs_multiact",
    "shopshelves_vs_cashier",
    "private_area_vs_gasstation",
    "private_area_vs_commercial",
    "private_area_vs_commrcial",
    "commercial_vs_gasstation",
]
LABEL_MAP = {
    "color_vs_grayscale": ("Color encoding", "colour / grayscale"),
    "highresolution_vs_lowresolution": ("Spatial resolution", "high / low"),
    "highmotion_vs_lowmotion": ("Motion energy", "high / low"),
    "singleview_vs_multiview": ("Camera continuity", "single-view / multi-view"),
    "closeview_vs_wideview": ("Field of view", "close / wide"),
    "outdoor_vs_indoor": ("Setting", "outdoor / indoor"),
    "few-person_vs_crowd": ("Crowd density", "sparse / crowded"),
    "singleact_vs_multiact": ("Concurrent activity", "single / multiple"),
    "highway_vs_sideroad": ("Camera elevation", "elevated / eye-level"),
    "shopshelves_vs_cashier": ("Retail viewpoint", "shelves / checkout"),
    "private_area_vs_commrcial": ("Location type -1", "residential / commercial"),
    "private_area_vs_commercial": ("Location type -1", "residential / commercial"),
    "private_area_vs_gasstation": ("Location type -2", "residential / fuel station"),
    "commercial_vs_gasstation": ("Location type -3", "commercial / fuel station"),
}
LOCATION_GROUP = [
    "private_area_vs_commercial",
    "private_area_vs_commrcial",
    "private_area_vs_gasstation",
    "commercial_vs_gasstation",
]
BAND = 0.05
REF = 0.5
_GOLDEN = 0.6180339887


def _palette(n):
    fills, texts = [], []
    for i in range(n):
        h = (0.02 + i * _GOLDEN) % 1.0
        fills.append(colorsys.hls_to_rgb(h, 0.78, 0.48))
        texts.append(colorsys.hls_to_rgb(h, 0.36, 0.58))
    return fills, texts


def load_bias_table(csv_path, value_col="bias_auc_pooled"):
    """
    Load long-form CSV from evaluate_scene_bias (model, factor, bias_auc_pooled)
    or wide-form (*_pooled columns).
    """
    df = pd.read_csv(csv_path)
    if "factor" in df.columns and "model" in df.columns and value_col in df.columns:
        wide = df.pivot_table(index="factor", columns="model", values=value_col, aggfunc="first")
        return wide
    if "factor" in df.columns:
        df = df.set_index("factor")
        cols = [c for c in df.columns if c.endswith("_pooled")]
        if cols:
            return df[cols].rename(columns={c: c.replace("_pooled", "") for c in cols})
    raise ValueError(f"unrecognized bias CSV layout: {csv_path}")


def order_by_effect(pooled, factors, models=None):
    factors = [f for f in factors if f in pooled.index]
    cols = list(pooled.columns) if models is None else [m for m in models if m in pooled.columns]
    if not factors or not cols:
        return factors
    means = pooled.loc[factors, cols].mean(axis=1)
    dev = (means - REF).abs()
    loc_in = [f for f in LOCATION_GROUP if f in factors]
    # de-dup commercial spelling
    seen = set()
    loc_in = [f for f in loc_in if not (f in seen or seen.add(f))]
    if len(loc_in) <= 1:
        return dev.sort_values(ascending=False).index.tolist()
    block_key = max(dev.loc[loc_in])
    ordered, inserted = [], False
    for f in dev.sort_values(ascending=False).index.tolist():
        if f in loc_in:
            if not inserted:
                ordered.extend(loc_in)
                inserted = True
            continue
        if not inserted and dev[f] <= block_key:
            ordered.extend(loc_in)
            inserted = True
        ordered.append(f)
    if not inserted:
        ordered.extend(loc_in)
    return ordered


def plot_bias_dotplot(pooled, models=None, out_prefix="scene_bias_auc_dotplot", out_dir="."):
    models = list(models) if models is not None else list(pooled.columns)
    models = [m for m in models if m in pooled.columns]
    fills, texts = _palette(len(CAPTURE_FACTORS) + len(CONTENT_FACTORS))
    all_f = CAPTURE_FACTORS + [f for f in CONTENT_FACTORS if f in pooled.index]
    factor_fill = {f: fills[i % len(fills)] for i, f in enumerate(all_f)}
    factor_text = {f: texts[i % len(texts)] for i, f in enumerate(all_f)}

    capture = order_by_effect(pooled, CAPTURE_FACTORS, models)
    content = order_by_effect(pooled, CONTENT_FACTORS, models)
    lane = {m: off for m, off in zip(models, np.linspace(-0.28, 0.28, len(models)))}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 8.6),
        gridspec_kw={"height_ratios": [len(capture), len(content)], "hspace": 0.45},
        sharex=True,
    )

    def _draw(ax, factors, title):
        n = len(factors)
        ax.axvspan(REF - BAND, REF + BAND, color="0.90", zorder=0)
        ax.axvline(REF, color="0.25", lw=1.4, ls="--", zorder=1)
        for row, factor in enumerate(factors):
            y0 = n - 1 - row
            for m in models:
                v = pooled.loc[factor, m]
                ax.scatter(
                    v, y0 + lane[m], marker="o", s=60,
                    facecolor=factor_fill.get(factor, "0.7"),
                    edgecolor=factor_text.get(factor, "0.25"),
                    linewidth=1.0, alpha=0.95, zorder=3,
                )
            ax.set_yticks([])
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            main, poles = LABEL_MAP.get(factor, (factor, ""))
            ax.text(-0.02, y0 + 0.24, main, transform=trans, ha="right", va="center",
                    fontsize=13.5, fontweight="bold", color=factor_text.get(factor, "0.2"), clip_on=False)
            ax.text(-0.02, y0 - 0.24, poles, transform=trans, ha="right", va="center",
                    fontsize=12, color="0.4", style="italic", clip_on=False)
        ax.set_ylim(-0.6, n - 1 + 0.6)
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=10)
        ax.grid(axis="x", color="0.9", lw=0.7, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    _draw(ax1, capture, "Capture factors")
    _draw(ax2, content, "Content factors")
    ax2.set_xlabel("bias-AUC (pooled)  —  0.5 = unbiased", labelpad=8)
    sub = pooled.loc[:, models].values.astype(float)
    ax2.set_xlim(min(0.2, np.nanmin(sub) - 0.03), max(0.8, np.nanmax(sub) + 0.03))

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="0.7",
                   markeredgecolor="0.25", markersize=9, label="model"),
            Patch(facecolor="0.90", label=f"|bias − 0.5| ≤ {BAND} band"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=10.5,
    )
    fig.subplots_adjust(top=0.96, bottom=0.16, left=0.24, right=0.97)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{out_prefix}.png"
    pdf = out_dir / f"{out_prefix}.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return str(png), str(pdf)


def plot_bias_heatmap(pooled, models=None, out_prefix="scene_bias_auc_heatmap", out_dir="."):
    models = list(models) if models is not None else list(pooled.columns)
    models = [m for m in models if m in pooled.columns]
    capture = order_by_effect(pooled, CAPTURE_FACTORS, models)
    content = order_by_effect(pooled, CONTENT_FACTORS, models)
    factor_order = capture + content
    mat = pooled.loc[factor_order, models].values.astype(float)
    vabs = max(abs(np.nanmin(mat) - REF), abs(np.nanmax(mat) - REF), 0.05)
    vmin, vmax = REF - vabs, REF + vabs

    fig, ax = plt.subplots(figsize=(max(11.0, 1.0 * len(models) + 3.0), 8.6))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=12.5)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fills, texts = _palette(len(factor_order))
    row_trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, factor in enumerate(factor_order):
        main, poles = LABEL_MAP.get(factor, (factor, ""))
        ax.text(-0.02, i - 0.18, main, transform=row_trans, ha="right", va="center",
                fontsize=13.5, fontweight="bold", color=texts[i % len(texts)], clip_on=False)
        ax.text(-0.02, i + 0.22, poles, transform=row_trans, ha="right", va="center",
                fontsize=12, color="0.45", style="italic", clip_on=False)
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = "white" if abs(v - REF) > 0.55 * vabs else "0.15"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11, color=txt)

    ax.axhline(len(capture) - 0.5, color="0.2", lw=2.0, zorder=3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.02)
    cbar.ax.invert_yaxis()
    cbar.ax.set_title("bias-AUC\n(pooled)", fontsize=12.5, pad=8)
    fig.subplots_adjust(top=0.94, bottom=0.14, left=0.26, right=0.90)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{out_prefix}.png"
    pdf = out_dir / f"{out_prefix}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return str(png), str(pdf)
