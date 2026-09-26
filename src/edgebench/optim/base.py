"""Shared types for the deployment optimization ladder.

Every optimization is a function ``(OptimizationContext, params) -> OptimizationOutcome``.
The ladder is *fault tolerant by construction*: an optimization that cannot run on
the current machine (``torch.compile`` without Triton on Windows, ONNX Runtime
not installed, FX quantization hitting an unsupported op) produces an outcome
with ``status="unavailable"`` or ``status="failed"`` and a human-readable reason.
It never aborts the run and it never silently disappears from the results.

That property is what makes the study honest: the report can say "6 of 7
optimizations applied on this machine, and here is exactly why the seventh did
not" instead of quietly reporting a smaller matrix.
"""

from __future__ import annotations

import copy
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edgebench.config import BenchmarkConfig, TrainConfig
from edgebench.inference import Predictor
from edgebench.utils import get_logger

logger = get_logger("optim")

OutcomeStatus = Literal["applied", "unavailable", "failed"]

#: Exceptions that must propagate. The ladder is fault tolerant on purpose, but
#: a user pressing Ctrl-C should not be reported as a failed optimization.
FATAL_EXCEPTIONS: tuple[type[BaseException], ...] = (KeyboardInterrupt, SystemExit)


def is_fatal(error: BaseException) -> bool:
    """Whether ``error`` should be re-raised rather than recorded."""
    return isinstance(error, (*FATAL_EXCEPTIONS, MemoryError))


@dataclass
class OptimizationContext:
    """Everything an optimization is allowed to depend on."""

    #: A *fresh, independent copy* of the trained FP32 model. Optimizations may
    #: mutate this object freely; the ladder owns the copying so no optimizer can
    #: contaminate another.
    model: nn.Module
    model_id: str
    num_classes: int
    input_size: int
    device: torch.device
    seed: int
    artifacts_dir: Path
    train_cfg: TrainConfig
    bench_cfg: BenchmarkConfig

    train_loader: DataLoader[Any] | None = None
    val_loader: DataLoader[Any] | None = None
    #: Small, deterministic sample used to calibrate observers and export graphs.
    calib_loader: DataLoader[Any] | None = None

    def example_inputs(self, batch_size: int = 1) -> torch.Tensor:
        """A correctly shaped, correctly normalised dummy batch."""
        return torch.randn(batch_size, 3, self.input_size, self.input_size, device=self.device)

    def example_inputs_numpy(self, batch_size: int = 1):
        """NumPy view of :meth:`example_inputs`, for the ONNX toolchain."""
        return self.example_inputs(batch_size).cpu().numpy()


@dataclass
class OptimizationOutcome:
    """The result of attempting one optimization."""

    optimization_id: str
    status: OutcomeStatus
    reason: str | None = None
    predictor: Predictor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def applied(self) -> bool:
        return self.status == "applied" and self.predictor is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimization_id": self.optimization_id,
            "status": self.status,
            "reason": self.reason,
            "params": self.params,
            "notes": self.notes,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
            "predictor": self.predictor.describe() if self.predictor else None,
        }


#: Signature every optimization implementation must satisfy.
OptimizationFn = Callable[[OptimizationContext, dict[str, Any]], OptimizationOutcome]


def clone_module(module: nn.Module) -> nn.Module:
    """Deep-copy a model so optimizations cannot alias one another.

    ``copy.deepcopy`` is used rather than ``state_dict`` round-tripping because
    some quantized modules hold non-tensor state that ``load_state_dict`` does
    not restore.
    """
    cloned = copy.deepcopy(module)
    cloned.eval()
    return cloned


def unavailable(
    optimization_id: str,
    reason: str,
    params: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> OptimizationOutcome:
    """Build an outcome for an optimization that is not applicable here."""
    logger.info("%s: unavailable (%s)", optimization_id, reason)
    return OptimizationOutcome(
        optimization_id=optimization_id,
        status="unavailable",
        reason=reason,
        params=params or {},
        notes=notes or [],
    )


def failed(
    optimization_id: str,
    error: BaseException,
    params: dict[str, Any] | None = None,
    include_traceback: bool = True,
) -> OptimizationOutcome:
    """Build an outcome for an optimization that raised."""
    detail = f"{type(error).__name__}: {error}".strip().rstrip(":")
    logger.warning("%s: failed (%s)", optimization_id, detail)
    notes: list[str] = []
    if include_traceback:
        # Truncated: the full trace is in the run log, this is for the report.
        notes.append("".join(traceback.format_exception_only(type(error), error)).strip())
    return OptimizationOutcome(
        optimization_id=optimization_id,
        status="failed",
        reason=detail,
        params=params or {},
        notes=notes,
    )
