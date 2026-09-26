"""The deployment optimization ladder.

``LADDER`` maps a configuration id to an implementation. Adding an optimization
means adding one function and one dictionary entry; the benchmark, reporting and
CLI layers pick it up automatically because they only ever iterate over outcomes.

The baseline (``fp32``) is itself an entry in the ladder. That keeps the
comparison honest: every row in the results table, including the reference row,
travels the same code path and is measured by the same protocol.
"""

from __future__ import annotations

import re
from dataclasses import replace

from edgebench.config import OptimizationEntry
from edgebench.inference import TorchPredictor
from edgebench.optim.base import (
    OptimizationContext,
    OptimizationFn,
    OptimizationOutcome,
    clone_module,
    failed,
    unavailable,
)
from edgebench.optim.compilation import apply_compile
from edgebench.optim.onnx_runtime import apply_onnxruntime
from edgebench.optim.pruning import apply_unstructured_pruning
from edgebench.optim.quantization import (
    apply_dynamic_quantization,
    apply_qat_quantization,
    apply_static_quantization,
)

__all__ = [
    "LADDER",
    "LADDER_IDS",
    "OptimizationContext",
    "OptimizationOutcome",
    "apply_identity",
    "apply_ladder",
    "run_optimization",
]


def apply_identity(ctx: OptimizationContext, params: dict) -> OptimizationOutcome:
    """The FP32 reference: a plain eager model with no optimization applied.

    Every other row in the table is a delta against this one, so it is produced by
    the same machinery rather than by a special case in the reporting layer.
    """
    model = clone_module(ctx.model)
    return OptimizationOutcome(
        optimization_id=params.get("optimization_id", "fp32"),
        status="applied",
        params=dict(params),
        predictor=TorchPredictor(model, label="fp32"),
        metadata={"weight_bits": 32, "activation_bits": 32},
        notes=["eager PyTorch, no graph rewrite, no weight compression"],
    )


def _pruning_fn(ctx: OptimizationContext, params: dict) -> OptimizationOutcome:
    """Resolve a pruning amount from the optimization id when not given explicitly.

    ``prune_unstructured_50`` means 50% sparsity, which keeps the config readable
    and lets several sparsity levels coexist in the ladder.
    """
    if "amount" not in params:
        match = re.search(r"(\d+)$", str(params.get("optimization_id", "")))
        if match:
            params = {**params, "amount": int(match.group(1)) / 100.0}
    return apply_unstructured_pruning(ctx, params)


def _onnx_int8_fn(ctx: OptimizationContext, params: dict) -> OptimizationOutcome:
    return apply_onnxruntime(ctx, {"quantize": True, **params})


#: Registered optimizations. Keys are the ids used in ``configs/default.yaml``.
LADDER: dict[str, OptimizationFn] = {
    "fp32": apply_identity,
    "compile": apply_compile,
    "dynamic_int8": apply_dynamic_quantization,
    "static_int8": apply_static_quantization,
    "qat_int8": apply_qat_quantization,
    "prune_unstructured_30": _pruning_fn,
    "prune_unstructured_50": _pruning_fn,
    "prune_unstructured_70": _pruning_fn,
    "onnxruntime": apply_onnxruntime,
    "onnxruntime_int8": _onnx_int8_fn,
}

LADDER_IDS: tuple[str, ...] = tuple(LADDER)


def run_optimization(
    entry: OptimizationEntry,
    base_context: OptimizationContext,
) -> OptimizationOutcome:
    """Apply one optimization to an isolated copy of the trained model.

    The model is deep-copied here rather than inside each implementation so that
    no optimization can observe, or be contaminated by, another. ``torch.compile``
    and FX quantization both mutate module graphs in place; sharing one instance
    across the ladder would make results order-dependent.
    """
    implementation = LADDER.get(entry.id)
    if implementation is None:
        return unavailable(
            entry.id,
            f"unknown optimization id; registered ids: {', '.join(LADDER_IDS)}",
        )

    params = {"optimization_id": entry.id, **entry.params}
    isolated = replace(base_context, model=clone_module(base_context.model))

    try:
        outcome = implementation(isolated, params)
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        return failed(entry.id, error, params=params)

    # Normalise: an implementation may have chosen a generic label.
    outcome.optimization_id = entry.id
    return outcome


def apply_ladder(
    entries: list[OptimizationEntry],
    base_context: OptimizationContext,
) -> list[OptimizationOutcome]:
    """Run every enabled optimization, in the order given by the configuration."""
    return [run_optimization(entry, base_context) for entry in entries]
