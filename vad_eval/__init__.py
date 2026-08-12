"""Public evaluation protocol for auditing frame-level AUC in WSVAD."""

from .granularity import align_packs, build_pack, build_probe_records, build_score_records, point_metrics
from .resolution import GRANULARITIES, paired_bootstrap, resolve_at_granularity
from .scene_bias import factor_bias
from .scene_conditioned import pair_matrix, scene_conditioned

__all__ = [
    "GRANULARITIES",
    "align_packs",
    "build_pack",
    "build_probe_records",
    "build_score_records",
    "factor_bias",
    "pair_matrix",
    "paired_bootstrap",
    "point_metrics",
    "resolve_at_granularity",
    "scene_conditioned",
]
