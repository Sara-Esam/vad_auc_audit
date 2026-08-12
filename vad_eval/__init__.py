"""Evaluation protocol for auditing frame-level AUC in WSVAD."""

from .io import (
    canonical_id,
    collect_video_units,
    flatten_scores,
    load_model_arrays,
    load_nested_dict,
    load_scene_pairs,
    load_video_list,
    merge_gt,
)
from .metrics import auc, category_of, to_snippets

__all__ = [
    "auc",
    "canonical_id",
    "category_of",
    "collect_video_units",
    "flatten_scores",
    "load_model_arrays",
    "load_nested_dict",
    "load_scene_pairs",
    "load_video_list",
    "merge_gt",
    "to_snippets",
]
