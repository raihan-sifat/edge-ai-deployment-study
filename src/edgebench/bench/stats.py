"""Summary statistics for latency samples.

Raw means are not enough for a latency claim. CPU scheduling, turbo behaviour and
allocator state produce a right-skewed distribution, so the study reports the
median and the 90th/95th percentiles alongside the mean, plus a coefficient of
variation so that an unstable measurement is visible rather than averaged away.

A high ``cv`` is treated as a warning sign in the report, not as noise to hide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimingStats:
    """Distribution of a set of latency samples, in milliseconds."""

    count: int
    mean: float
    std: float
    minimum: float
    p50: float
    p90: float
    p95: float
    p99: float
    maximum: float
    iqr: float
    cv: float

    @classmethod
    def from_samples_ms(cls, samples: list[float] | np.ndarray) -> TimingStats:
        """Summarise latency samples expressed in milliseconds."""
        array = np.asarray(samples, dtype=np.float64)
        if array.size == 0:
            raise ValueError("cannot summarise an empty sample set")

        mean = float(array.mean())
        std = float(array.std(ddof=1)) if array.size > 1 else 0.0
        p25, p50, p75, p90, p95, p99 = np.percentile(array, [25, 50, 75, 90, 95, 99])

        return cls(
            count=int(array.size),
            mean=mean,
            std=std,
            minimum=float(array.min()),
            p50=float(p50),
            p90=float(p90),
            p95=float(p95),
            p99=float(p99),
            maximum=float(array.max()),
            iqr=float(p75 - p25),
            # Guard against a degenerate zero-variance sample set.
            cv=std / mean if mean > 0 else 0.0,
        )

    @property
    def throughput_per_second(self) -> float:
        """Inferences per second implied by the median latency."""
        return 1000.0 / self.p50 if self.p50 > 0 else float("nan")

    def to_dict(self, prefix: str = "") -> dict[str, float | int]:
        """Flatten to a dict, optionally prefixed, for tidy CSV/DataFrame columns."""
        return {
            f"{prefix}count": self.count,
            f"{prefix}mean_ms": self.mean,
            f"{prefix}std_ms": self.std,
            f"{prefix}min_ms": self.minimum,
            f"{prefix}p50_ms": self.p50,
            f"{prefix}p90_ms": self.p90,
            f"{prefix}p95_ms": self.p95,
            f"{prefix}p99_ms": self.p99,
            f"{prefix}max_ms": self.maximum,
            f"{prefix}iqr_ms": self.iqr,
            f"{prefix}cv": self.cv,
        }


def coefficient_of_variation(samples: list[float]) -> float:
    """Coefficient of variation, used to flag unstable measurements."""
    array = np.asarray(samples, dtype=np.float64)
    mean = array.mean()
    if mean <= 0:
        return float("nan")
    return float(array.std(ddof=1) / mean) if array.size > 1 else 0.0


def summarise_speedup(baseline_ms: float, candidate_ms: float) -> dict[str, Any]:
    """Speedup and relative reduction of a candidate against a baseline."""
    if baseline_ms <= 0 or candidate_ms <= 0:
        return {"speedup": float("nan"), "latency_reduction_pct": float("nan")}
    return {
        "speedup": baseline_ms / candidate_ms,
        "latency_reduction_pct": 100.0 * (baseline_ms - candidate_ms) / baseline_ms,
    }
