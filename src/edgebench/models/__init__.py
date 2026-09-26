"""Model construction, CIFAR adaptation and static cost analysis."""

from __future__ import annotations

from edgebench.models.adapt import (
    adapt_stem_to_cifar,
    replace_classifier_head,
    verify_head,
)
from edgebench.models.analysis import (
    ParameterBreakdown,
    count_multiply_accumulates,
    parameter_breakdown,
)
from edgebench.models.registry import (
    build_model,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "ParameterBreakdown",
    "adapt_stem_to_cifar",
    "build_model",
    "checkpoint_path",
    "count_multiply_accumulates",
    "load_checkpoint",
    "parameter_breakdown",
    "replace_classifier_head",
    "save_checkpoint",
    "verify_head",
]
