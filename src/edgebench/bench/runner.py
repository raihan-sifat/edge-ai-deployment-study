"""Orchestration: turn an optimization outcome into a complete measurement record.

One record is produced per ``(model, optimization)`` pair. It is the atomic unit
of the study and the only thing the reporting layer reads. Because everything the
charts need is in the record, a figure can always be regenerated without re-running
a single forward pass -- which matters when a full run takes hours.

Record layout::

    {
      "run_id": ..., "model_id": ..., "optimization_id": ...,
      "accuracy":   {top1, top5, loss, ece, per_class_accuracy, confusion},
      "footprint":  {weight_bytes, parameters, macs, rss_peak_delta_bytes, ...},
      "latency":    [ {resolution, batch_size, num_threads, latency_ms, ...}, ... ],
      "environment": {...}, "config": {...}
    }
"""

from __future__ import annotations

import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch
from torch.utils.data import DataLoader

from edgebench import RESULTS_SCHEMA_VERSION, __version__
from edgebench.bench.energy import EnergyMeter
from edgebench.bench.latency import LatencyMeasurement, measure_latency
from edgebench.config import BenchmarkConfig
from edgebench.inference import Predictor
from edgebench.models.analysis import count_multiply_accumulates, parameter_breakdown
from edgebench.optim.base import OptimizationOutcome
from edgebench.training import AccuracyMetrics, evaluate
from edgebench.utils import (
    describe_cpu,
    environment_fingerprint,
    get_logger,
    git_commit_hash,
)

logger = get_logger("bench.runner")

#: Static architcture costs copied from the model card into a record's footprint.
_STATIC_FOOTPRINT_KEYS = (
    "parameters",
    "params_total",
    "params_trainable",
    "params_conv",
    "params_linear",
    "params_normalization",
    "params_other",
    "macs",
    "flops",
    "macs_millions",
    "parameter_bytes_fp32",
    "macs_note",
)


def new_run_id() -> str:
    """Short, sortable identifier for a benchmark run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass
class BenchmarkRecord:
    """A complete measurement of one (model, optimization) pair."""

    run_id: str
    model_id: str
    optimization_id: str
    status: str
    reason: str | None = None

    accuracy: dict[str, Any] | None = None
    footprint: dict[str, Any] = field(default_factory=dict)
    latency: list[dict[str, Any]] = field(default_factory=list)

    optimization_params: dict[str, Any] = field(default_factory=dict)
    optimization_metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    predictor: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    model_card: dict[str, Any] = field(default_factory=dict)

    schema_version: int = RESULTS_SCHEMA_VERSION
    edgebench_version: str = __version__
    created_at: str = ""

    # -- derived accessors used by the reporting layer ---------------------

    @property
    def top1(self) -> float | None:
        return None if not self.accuracy else self.accuracy.get("top1")

    def latency_cell(
        self, resolution: int, batch_size: int, num_threads: int | None
    ) -> dict[str, Any] | None:
        for cell in self.latency:
            if (
                cell["resolution"] == resolution
                and cell["batch_size"] == batch_size
                and cell["num_threads"] == num_threads
                and cell["status"] == "ok"
            ):
                return cell
        return None

    @property
    def headline_latency_ms(self) -> float | None:
        """Latency quoted in summary tables: batch 1, single thread, 32px.

        Single-thread batch-of-one is the least flattering configuration and the
        closest analogue to a small edge core running one request at a time. It is
        the right default for a headline number because it cannot be improved by
        giving the benchmark more hardware.
        """
        cell = self.latency_cell(resolution=32, batch_size=1, num_threads=1)
        if cell is None:
            cell = self.latency_cell(resolution=32, batch_size=1, num_threads=None)
        return None if cell is None else cell.get("latency_ms")

    @property
    def weight_bytes(self) -> int | None:
        return self.footprint.get("weight_bytes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "edgebench_version": self.edgebench_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "model_id": self.model_id,
            "optimization_id": self.optimization_id,
            "status": self.status,
            "reason": self.reason,
            "accuracy": self.accuracy,
            "footprint": self.footprint,
            "latency": self.latency,
            "optimization_params": self.optimization_params,
            "optimization_metadata": self.optimization_metadata,
            "notes": self.notes,
            "artifacts": self.artifacts,
            "predictor": self.predictor,
            "environment": self.environment,
            "model_card": self.model_card,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkRecord:
        known = set(cls.__annotations__)
        return cls(**{key: value for key, value in payload.items() if key in known})


def build_model_card(
    model_id: str,
    torchvision_name: str,
    num_classes: int,
    input_size: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Static description of the architecture, independent of any optimization."""
    return {
        "model_id": model_id,
        "torchvision_name": torchvision_name,
        "num_classes": num_classes,
        "input_size": input_size,
        **params,
    }


def static_cost(
    model: torch.nn.Module,
    input_size: int,
    num_classes: int,
) -> dict[str, Any]:
    """Parameter breakdown and MAC count, computed once per architecture."""
    breakdown = parameter_breakdown(model)
    macs = count_multiply_accumulates(model, input_size=input_size, batch_size=1)
    return {
        **breakdown.to_dict(),
        # Alias so downstream code does not need to know that the canonical name
        # is `params_total`.
        "parameters": breakdown.total,
        "macs": macs,
        "flops": None if macs is None else 2 * macs,
        "macs_millions": None if macs is None else macs / 1e6,
        "macs_note": "one MAC = one multiply-accumulate; FLOPs = 2 x MACs",
        "parameter_bytes_fp32": breakdown.size_fp32_bytes,
    }


def measure_accuracy(
    predictor: Predictor,
    test_loader: DataLoader[Any],
    device: torch.device,
    num_classes: int,
    with_details: bool = True,
) -> AccuracyMetrics:
    """Top-1/top-5/loss/ECE on the held-out test split."""
    return evaluate(
        predictor,
        test_loader,
        device,
        num_classes,
        with_details=with_details,
    )


def benchmark_one(
    *,
    run_id: str,
    model_id: str,
    outcome: OptimizationOutcome,
    test_loader: DataLoader[Any],
    device: torch.device,
    num_classes: int,
    bench_cfg: BenchmarkConfig,
    seed: int,
    model_card: dict[str, Any],
    energy_meter: EnergyMeter | None = None,
    measure_memory: bool = True,
    with_accuracy_details: bool = True,
) -> BenchmarkRecord:
    """Measure a single optimized model end to end."""
    record = BenchmarkRecord(
        run_id=run_id,
        model_id=model_id,
        optimization_id=outcome.optimization_id,
        status=outcome.status,
        reason=outcome.reason,
        optimization_params=outcome.params,
        optimization_metadata=outcome.metadata,
        notes=list(outcome.notes),
        artifacts=dict(outcome.artifacts),
        model_card=model_card,
        environment=_environment_block(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    if not outcome.applied or outcome.predictor is None:
        logger.info(
            "%s: skipping measurement for %s (status=%s)",
            model_id,
            outcome.optimization_id,
            outcome.status,
        )
        return record

    predictor = outcome.predictor
    record.predictor = predictor.describe()

    # ---- accuracy ---------------------------------------------------------
    try:
        metrics = measure_accuracy(
            predictor,
            test_loader,
            device,
            num_classes,
            with_details=with_accuracy_details,
        )
        record.accuracy = metrics.to_dict(include_matrices=with_accuracy_details)
        logger.info(
            "%s/%s: test top-1 %.2f%% top-5 %.2f%% ece %.4f",
            model_id,
            outcome.optimization_id,
            100 * metrics.top1,
            100 * metrics.top5,
            metrics.ece or float("nan"),
        )
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt | SystemExit):
            raise
        record.accuracy = None
        record.status = "accuracy_failed"
        record.reason = f"{type(error).__name__}: {error}"
        logger.warning("%s/%s: accuracy failed: %s", model_id, outcome.optimization_id, error)

    # ---- footprint --------------------------------------------------------
    try:
        from edgebench.bench.footprint import footprint_summary

        model = getattr(predictor, "module", None)
        summary_input = torch.rand(1, 3, bench_cfg.resolutions[0], bench_cfg.resolutions[0]) * 2 - 1
        record.footprint = footprint_summary(
            predictor,
            model if isinstance(model, torch.nn.Module) else None,
            summary_input,
            input_size=bench_cfg.resolutions[0],
            measure_memory=measure_memory,
        )
    except BaseException as error:
        record.footprint = {"error": f"{type(error).__name__}: {error}"}

    # Backfill architecture-level cost from the model card. Measured values win;
    # this only fills gaps, which matters for backends with no ``nn.Module``
    # (ONNX Runtime) where parameter counts cannot be introspected from the object.
    for key in _STATIC_FOOTPRINT_KEYS:
        if key in model_card:
            record.footprint.setdefault(key, model_card[key])

    # ---- latency grid -----------------------------------------------------
    cells: list[dict[str, Any]] = []
    for resolution in bench_cfg.resolutions:
        for batch_size in bench_cfg.batch_sizes:
            for thread_count in bench_cfg.thread_counts:
                measurement: LatencyMeasurement = measure_latency(
                    predictor,
                    resolution=resolution,
                    batch_size=batch_size,
                    warmup_iters=bench_cfg.warmup_iters,
                    timed_iters=bench_cfg.timed_iters,
                    repeats=bench_cfg.repeats,
                    num_threads=thread_count,
                    seed=seed,
                    measure_memory=measure_memory,
                    energy_meter=energy_meter,
                )
                cells.append(measurement.to_dict())
    record.latency = cells

    return record


def _environment_block() -> dict[str, Any]:
    """Environment metadata embedded in every record."""
    return {
        "platform": platform.platform(),
        "cpu": describe_cpu(),
        "git_commit": git_commit_hash(),
        "fingerprint": environment_fingerprint(),
    }
