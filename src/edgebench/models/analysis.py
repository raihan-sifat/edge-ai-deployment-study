"""Architecture-level cost analysis: parameters, buffers and multiply-accumulates.

One convention is fixed globally: **one MAC is one multiply-accumulate, and
FLOPs = 2 x MACs.** Papers mix the two freely, so any comparison against
published numbers has to state which convention is in use. The study reports MACs
and labels them as such; :func:`flops_from_macs` converts when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from edgebench.utils import get_logger

logger = get_logger("models.analysis")


@dataclass(frozen=True)
class ParameterBreakdown:
    """Static size accounting for a model."""

    total: int
    trainable: int
    frozen: int
    buffers: int
    conv: int
    linear: int
    normalization: int
    other: int

    @property
    def non_conv_linear(self) -> int:
        """Parameters that structured pruning cannot remove."""
        return self.total - self.conv - self.linear

    def to_dict(self) -> dict[str, int]:
        return {
            "params_total": self.total,
            "params_trainable": self.trainable,
            "params_frozen": self.frozen,
            "params_buffers": self.buffers,
            "params_conv": self.conv,
            "params_linear": self.linear,
            "params_normalization": self.normalization,
            "params_other": self.other,
        }

    @property
    def size_fp32_bytes(self) -> int:
        return 4 * self.total

    @property
    def size_int8_bytes(self) -> int:
        """Approximate INT8 size: one byte per weight, plus FP32 scale/zero-point.

        The per-channel overhead is ~8 bytes per output channel, which is
        negligible at the granularity reported here but is why an INT8 model is
        slightly larger than exactly ``params`` bytes.
        """
        return self.total + 8 * 64


def parameter_breakdown(model: nn.Module) -> ParameterBreakdown:
    """Count parameters by category."""
    totals = {"conv": 0, "linear": 0, "norm": 0, "other": 0}
    trainable = 0
    frozen = 0
    buffers = 0

    seen: set[int] = set()
    for module in model.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))

        for name, parameter in module.named_parameters(recurse=False):
            count = parameter.numel()
            if parameter.requires_grad:
                trainable += count
            else:
                frozen += count

            if _is_normalization_parameter(module, name):
                totals["norm"] += count
            elif isinstance(module, nn.Conv2d | nn.ConvTranspose2d | nn.Conv1d):
                totals["conv"] += count
            elif isinstance(module, nn.Linear):
                totals["linear"] += count
            else:
                totals["other"] += count

        for buffer in module.buffers(recurse=False):
            buffers += buffer.numel()

    total = trainable + frozen
    return ParameterBreakdown(
        total=total,
        trainable=trainable,
        frozen=frozen,
        buffers=buffers,
        conv=totals["conv"],
        linear=totals["linear"],
        normalization=totals["norm"],
        other=totals["other"],
    )


def _is_normalization_parameter(module: nn.Module, name: str) -> bool:
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LocalResponseNorm,
    )
    return isinstance(module, norm_types) or name in {"running_mean", "running_var"}


def count_multiply_accumulates(
    model: nn.Module,
    input_size: int = 32,
    batch_size: int = 1,
) -> int | None:
    """Count MACs for a single forward pass.

    Uses :class:`torch.utils.flop_counter.FlopCounterMode`, available from
    PyTorch 2.1. ``FlopCounterMode`` reports FLOPs under the 2-per-MAC
    convention, which is divided out here.

    Returns ``None`` when the counter is unavailable or the model contains an
    operation the counter cannot handle, rather than inventing a number.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:  # pragma: no cover - very old torch
        logger.warning("torch.utils.flop_counter unavailable; MACs will be reported as null")
        return None

    was_training = model.training
    model.eval()
    example = torch.zeros(batch_size, 3, input_size, input_size)
    try:
        counter = FlopCounterMode(display=False)
        with counter, torch.inference_mode():
            model(example)
        total_flops = int(counter.get_total_flops())
    except Exception as error:  # pragma: no cover - counter support varies by op
        logger.warning("could not count MACs for %s: %s", type(model).__name__, error)
        return None
    finally:
        model.train(was_training)

    return total_flops // 2


def flops_from_macs(macs: int | None) -> int | None:
    """Convert MACs to FLOPs under the 2 FLOPs per MAC convention."""
    return None if macs is None else 2 * macs


def structure_summary(model: nn.Module, depth: int = 2) -> list[dict[str, Any]]:
    """Compact per-layer table used in the technical report appendix."""
    rows: list[dict[str, Any]] = []
    for index, (name, module) in enumerate(model.named_modules()):
        if index == 0:
            continue
        indent = "  " * (name.count(".") + 1)
        if name.count(".") > depth:
            continue
        params = sum(p.numel() for p in module.parameters(recurse=False))
        rows.append(
            {
                "name": f"{indent}{name.rpartition('.')[2]}",
                "type": type(module).__name__,
                "params": params,
            }
        )
    return rows
