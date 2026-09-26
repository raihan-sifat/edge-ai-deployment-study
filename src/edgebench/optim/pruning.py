"""Magnitude pruning, and an honest accounting of what it does and does not buy.

Global unstructured magnitude pruning removes the smallest weights by absolute
value, then the masks are made permanent with ``prune.remove``.

Four claims are tested explicitly in the report, because the folklore is
misleading:

1. **Sparsity does not shrink the shipped file.** ``torch.save`` writes dense
   tensors; zeroing half the weights changes the values, not the shape. The
   measured serialized size is therefore ~unchanged. A *theoretical* sparse
   footprint (nonzero values plus index overhead) is computed alongside it so the
   gap is visible rather than implied.
2. **The sparse estimate does not beat dense storage until past 50% sparsity.**
   Storing a float32 value with an int32 column index costs 8 bytes per surviving
   weight against 4 bytes dense, so the break-even is exactly 50% and the row
   pointers tip it marginally the wrong way there. This is the arithmetic reason
   unstructured pruning is not a compression technique as usually applied.
3. **Dense CPU kernels do not exploit sparsity.** MKL-DNN and oneDNN dense
   convolution paths have no sparsity-aware fast path for arbitrary masks. No
   speedup is expected, and none is claimed. This is verified empirically in
   ``scripts/check_measurement_order.py``, where a 50%-pruned model measures
   1.01x its FP32 baseline once measurement-order effects are controlled for.
4. **Zeroing weights still changes accuracy.** Sparsity is not free even when it
   is not cheap.

The only pruning variant that changes the arithmetic is structured (channel)
pruning, which requires dependency-aware surgery across residual adds and
concatenations. That is deliberately left as future work rather than implemented
incorrectly.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch.nn.utils import prune

from edgebench.inference import TorchPredictor
from edgebench.optim.base import (
    OptimizationContext,
    OptimizationOutcome,
    clone_module,
    failed,
    is_fatal,
    unavailable,
)
from edgebench.utils import get_logger

logger = get_logger("optim.prune")

#: Modules whose weights are eligible. Biases and normalization parameters are
#: excluded: pruning a bias to zero is a pure accuracy loss with no size benefit.
PRUNABLE_TYPES = (nn.Conv2d, nn.Linear)


def _prunable_parameters(module: nn.Module) -> list[tuple[nn.Module, str]]:
    """``(submodule, "weight")`` pairs eligible for pruning, in definition order."""
    targets: list[tuple[nn.Module, str]] = []
    for submodule in module.modules():
        if isinstance(submodule, PRUNABLE_TYPES) and submodule.weight is not None:
            targets.append((submodule, "weight"))
    return targets


def measure_sparsity(module: nn.Module) -> dict[str, float]:
    """Zero fractions, overall and restricted to prunable weights."""
    total = 0
    zeros = 0
    for parameter in module.parameters():
        total += parameter.numel()
        zeros += int((parameter.detach() == 0).sum().item())

    prunable_total = 0
    prunable_zeros = 0
    for submodule, attribute in _prunable_parameters(module):
        tensor = getattr(submodule, attribute).detach()
        prunable_total += tensor.numel()
        prunable_zeros += int((tensor == 0).sum().item())

    return {
        "zero_fraction_all": zeros / total if total else 0.0,
        "zero_fraction_prunable": prunable_zeros / prunable_total if prunable_total else 0.0,
        "total_parameters": float(total),
        "nonzero_parameters": float(total - zeros),
    }


def estimate_sparse_bytes(module: nn.Module, index_bytes: int = 4) -> int:
    """Theoretical size if every pruned weight matrix were stored sparsely (CSR).

    Per matrix: one 4-byte value plus one ``index_bytes`` column index per
    nonzero, plus one row pointer per output row. Non-prunable tensors (biases,
    normalization parameters, running statistics) stay dense.

    This is a *lower bound* on a real sparse format -- it ignores block padding
    and alignment -- and is always labelled as an estimate in the results.
    """
    total = 0
    prunable_submodule_ids: set[int] = set()

    for submodule, attribute in _prunable_parameters(module):
        tensor = getattr(submodule, attribute).detach()
        nonzero = int((tensor != 0).sum().item())
        rows = tensor.shape[0] if tensor.ndim >= 2 else 1
        total += nonzero * (4 + index_bytes) + rows * index_bytes
        prunable_submodule_ids.add(id(submodule))

    for name, parameter in module.named_parameters():
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = module.get_submodule(parent_name) if parent_name else module
        if id(parent) in prunable_submodule_ids and name.endswith("weight"):
            continue
        total += parameter.numel() * parameter.element_size()

    return total


def apply_unstructured_pruning(
    ctx: OptimizationContext, params: dict[str, Any]
) -> OptimizationOutcome:
    """Globally prune the smallest-magnitude weights by ``amount``.

    Supported params:
        ``amount``: fraction of *prunable* weights to zero, in ``[0, 1)``.
        ``scope``: ``global`` (default) ranks all prunable weights together;
            ``local`` applies ``amount`` per layer, which is harsher on small
            layers and generally worse.
    """
    optimization_id = str(
        params.get("optimization_id")
        or _default_id("prune_unstructured", float(params.get("amount", 0.5)))
    )
    amount = float(params.get("amount", 0.5))
    scope = str(params.get("scope", "global"))

    if not 0.0 <= amount < 1.0:
        return unavailable(
            optimization_id,
            f"pruning amount {amount} must lie in [0, 1)",
            params=params,
        )
    if scope not in {"global", "local"}:
        return unavailable(optimization_id, f"unknown scope {scope!r}", params=params)

    model = clone_module(ctx.model)
    targets = _prunable_parameters(model)
    if not targets:
        return unavailable(
            optimization_id,
            "model has no Conv2d or Linear weights to prune",
            params=params,
        )

    try:
        if scope == "global":
            prune.global_unstructured(
                targets,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        else:
            for submodule, attribute in targets:
                prune.l1_unstructured(submodule, name=attribute, amount=amount)

        # Turn the reparameterisation into real buffers: without this the mask is
        # applied at every forward pass and the model would be *slower*, which
        # would be an artifact of the implementation rather than of pruning.
        for submodule, attribute in targets:
            prune.remove(submodule, attribute)
    except BaseException as error:
        if is_fatal(error):
            raise
        return failed(optimization_id, error, params=params)

    model.eval()
    stats = measure_sparsity(model)
    dense_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    sparse_bytes = estimate_sparse_bytes(model)

    logger.info(
        "%s: pruned %.1f%% of prunable weights (global zero fraction %.1f%%)",
        ctx.model_id,
        100 * amount,
        100 * stats["zero_fraction_all"],
    )

    return OptimizationOutcome(
        optimization_id=optimization_id,
        status="applied",
        params={**params, "amount": amount, "scope": scope},
        predictor=TorchPredictor(model, label=f"prune-{round(amount * 100)}%"),
        metadata={
            **stats,
            "dense_weight_bytes": dense_bytes,
            "estimated_sparse_weight_bytes": sparse_bytes,
            "estimated_sparse_index_bytes": 4,
        },
        notes=[
            "dense storage is unchanged by magnitude pruning; the sparse estimate "
            "assumes a CSR layout and is a lower bound, not a measurement",
            "no latency improvement is expected on dense CPU kernels",
        ],
    )


def _default_id(prefix: str, amount: float) -> str:
    return f"{prefix}_{round(amount * 100)}"
