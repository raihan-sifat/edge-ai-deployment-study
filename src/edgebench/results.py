"""Loading and flattening benchmark records.

The reporting layer reads JSON and nothing else. No forward pass is executed to
build a chart, which means figures can be regenerated on a laptop from a results
directory produced on another machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from edgebench.bench.runner import BenchmarkRecord
from edgebench.utils import get_logger

logger = get_logger("results")

#: The configuration used for headline latency numbers in summary tables.
HEADLINE_RESOLUTION = 32
HEADLINE_BATCH_SIZE = 1
HEADLINE_THREADS: int | None = 1

#: Coefficient of variation above which a latency cell is treated as unstable.
#: Calibrated empirically: a quiet machine and a 1-thread configuration measure
#: at 5-8% CV, so 15% indicates that something else was competing for the CPU.
UNSTABLE_CV_THRESHOLD = 0.15


@dataclass
class ResultStore:
    """A directory of benchmark records plus the derived frames."""

    directory: Path
    records: list[BenchmarkRecord]

    @property
    def run_ids(self) -> list[str]:
        return sorted({record.run_id for record in self.records})

    def model_ids(self) -> list[str]:
        return sorted({record.model_id for record in self.records})

    def optimization_ids(self) -> list[str]:
        seen: list[str] = []
        for record in self.records:
            if record.optimization_id not in seen:
                seen.append(record.optimization_id)
        return seen

    def environment(self) -> dict[str, Any]:
        for record in self.records:
            if record.environment:
                return record.environment
        return {}

    def summary_frame(
        self,
        resolution: int = HEADLINE_RESOLUTION,
        batch_size: int = HEADLINE_BATCH_SIZE,
        num_threads: int | None = HEADLINE_THREADS,
    ) -> pd.DataFrame:
        """One row per (model, optimization) with the headline latency attached."""
        rows: list[dict[str, Any]] = []
        for record in self.records:
            cell = record.latency_cell(resolution, batch_size, num_threads)
            if cell is None:
                cell = record.latency_cell(resolution, batch_size, None)

            per_iteration = (cell or {}).get("per_iteration") or {}
            energy = (cell or {}).get("energy") or {}

            rows.append(
                {
                    "run_id": record.run_id,
                    "model_id": record.model_id,
                    "optimization_id": record.optimization_id,
                    "status": record.status,
                    "reason": record.reason,
                    "top1": None if record.accuracy is None else record.accuracy.get("top1"),
                    "top5": None if record.accuracy is None else record.accuracy.get("top5"),
                    "loss": None if record.accuracy is None else record.accuracy.get("loss"),
                    "ece": None if record.accuracy is None else record.accuracy.get("ece"),
                    "per_class_accuracy": (
                        None
                        if record.accuracy is None
                        else record.accuracy.get("per_class_accuracy")
                    ),
                    "confusion": None
                    if record.accuracy is None
                    else record.accuracy.get("confusion"),
                    "weight_bytes": record.footprint.get("weight_bytes"),
                    "weight_mib": record.footprint.get("weight_mib"),
                    "parameters": record.footprint.get("parameters")
                    or record.footprint.get("params_total"),
                    "macs": record.footprint.get("macs"),
                    "compression_ratio": record.footprint.get("compression_ratio"),
                    "rss_peak_delta_bytes": record.footprint.get("rss_peak_delta_bytes"),
                    "latency_ms": (cell or {}).get("latency_ms"),
                    "latency_mean_ms": per_iteration.get("mean_ms"),
                    "latency_p90_ms": per_iteration.get("p90_ms"),
                    "latency_p95_ms": per_iteration.get("p95_ms"),
                    "latency_cv": per_iteration.get("cv"),
                    "throughput_samples_per_s": (cell or {}).get("throughput_samples_per_s"),
                    "energy_per_sample_mj": energy.get("energy_per_sample_millijoules"),
                    "quantized_modules": record.optimization_metadata.get("quantized_modules"),
                    "zero_fraction_all": record.optimization_metadata.get("zero_fraction_all"),
                    "weight_bits": record.optimization_metadata.get("weight_bits"),
                    "backend": "pytorch" if record.predictor else None,
                    "predictor_label": (record.predictor or {}).get("label"),
                }
            )

        frame = pd.DataFrame(rows)
        if not frame.empty and frame["model_id"].notna().any():
            frame = _attach_baselines(frame)
        return frame

    def latency_frame(self) -> pd.DataFrame:
        """One row per latency cell: the full (resolution, batch, threads) grid."""
        rows: list[dict[str, Any]] = []
        for record in self.records:
            for cell in record.latency:
                per_iteration = cell.get("per_iteration") or {}
                rows.append(
                    {
                        "model_id": record.model_id,
                        "optimization_id": record.optimization_id,
                        "status": cell.get("status"),
                        "reason": cell.get("reason"),
                        "resolution": cell.get("resolution"),
                        "batch_size": cell.get("batch_size"),
                        "num_threads": cell.get("num_threads"),
                        "latency_ms": cell.get("latency_ms"),
                        "mean_ms": per_iteration.get("mean_ms"),
                        "p90_ms": per_iteration.get("p90_ms"),
                        "p95_ms": per_iteration.get("p95_ms"),
                        "p99_ms": per_iteration.get("p99_ms"),
                        "cv": per_iteration.get("cv"),
                        "std_ms": per_iteration.get("std_ms"),
                        "throughput_samples_per_s": cell.get("throughput_samples_per_s"),
                        "rss_peak_delta_bytes": (cell.get("memory") or {}).get(
                            "rss_peak_delta_bytes"
                        ),
                        "samples_ms": cell.get("samples_ms") or [],
                    }
                )
        return pd.DataFrame(rows)

    def status_frame(self) -> pd.DataFrame:
        """Which optimizations applied, which did not, and why."""
        rows = [
            {
                "model_id": record.model_id,
                "optimization_id": record.optimization_id,
                "status": record.status,
                "reason": record.reason,
                "notes": " ".join(record.notes),
            }
            for record in self.records
        ]
        return pd.DataFrame(rows)


def _attach_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Add per-model deltas against that model's own FP32 baseline.

    Deltas are computed against the *same model's* baseline rather than a global
    one, because the interesting question is "what does this optimization cost
    this architecture", not "which architecture is fastest".
    """
    frame = frame.copy()
    baseline = frame[frame["optimization_id"] == "fp32"].set_index("model_id")

    frame["baseline_top1"] = frame["model_id"].map(baseline["top1"])
    frame["baseline_latency_ms"] = frame["model_id"].map(baseline["latency_ms"])
    frame["baseline_weight_bytes"] = frame["model_id"].map(baseline["weight_bytes"])

    frame["top1_delta"] = frame["top1"] - frame["baseline_top1"]
    frame["top1_delta_points"] = 100.0 * frame["top1_delta"]

    frame["speedup_vs_fp32"] = frame["baseline_latency_ms"] / frame["latency_ms"]
    frame["latency_reduction_pct"] = (
        100.0 * (frame["baseline_latency_ms"] - frame["latency_ms"]) / frame["baseline_latency_ms"]
    )

    frame["size_reduction_pct"] = (
        100.0
        * (frame["baseline_weight_bytes"] - frame["weight_bytes"])
        / frame["baseline_weight_bytes"]
    )

    # A single "is this worth it" ratio: accuracy points sacrificed per 2x speedup.
    frame["accuracy_per_speedup"] = frame["top1_delta_points"] / (
        frame["speedup_vs_fp32"].replace(0, pd.NA)
    )

    # Flag unstable measurements instead of averaging them away. Sequential
    # benchmarking over several minutes lets machine state drift, and a drifted
    # cell can differ from a clean measurement by 4x -- far more than any
    # optimization in the ladder is worth. Surfacing the flag is what makes that
    # visible rather than silently wrong.
    frame["latency_unstable"] = frame["latency_cv"].gt(UNSTABLE_CV_THRESHOLD)
    frame["measurement_quality"] = [_quality_label(cv) for cv in frame["latency_cv"]]

    return frame


def _quality_label(cv: float | None) -> str:
    """Human-readable stability label for a coefficient of variation."""
    if cv is None or (isinstance(cv, float) and pd.isna(cv)):
        return "not measured"
    if cv > UNSTABLE_CV_THRESHOLD:
        return f"unstable (cv={100 * cv:.0f}%)"
    return "stable"


def load_records(results_dir: str | Path) -> list[BenchmarkRecord]:
    """Load every record under ``results_dir/raw`` (or ``results_dir`` itself)."""
    directory = Path(results_dir)
    raw_dir = directory / "raw"
    search_dir = raw_dir if raw_dir.exists() else directory

    if not search_dir.exists():
        raise FileNotFoundError(f"no results directory at {search_dir}")

    records: list[BenchmarkRecord] = []
    for path in sorted(search_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            logger.warning("skipping malformed record %s: %s", path.name, error)
            continue

        # A file may hold one record or a list of them.
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict) or "model_id" not in item:
                continue
            records.append(BenchmarkRecord.from_dict(item))

    if not records:
        logger.warning("no benchmark records found under %s", search_dir)

    return records


def load_store(results_dir: str | Path) -> ResultStore:
    """Load records and wrap them in a :class:`ResultStore`."""
    directory = Path(results_dir)
    return ResultStore(directory=directory, records=load_records(directory))
