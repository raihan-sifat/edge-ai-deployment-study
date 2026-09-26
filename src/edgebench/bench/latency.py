"""The latency measurement protocol.

The protocol is fixed here once, and every number in the study is produced by it:

1. **Thread count is set before anything else.** On a CPU, going from 4 to 1
   threads changes latency by more than most of the optimizations in the ladder,
   so it is an explicit axis rather than an ambient condition.
2. **Warmup iterations are discarded.** Allocators, oneDNN primitive caches and
   ``torch.compile`` all do one-time work on the first calls.
3. **Each iteration is timed individually**, and all the samples are kept. That is
   what makes the latency-distribution figure possible and what lets a skewed
   distribution be seen rather than averaged.
4. **A separate block-timing pass** runs ``timed_iters`` forwards inside one timer
   and divides. Per-iteration timing pays timer overhead and prevents any
   pipelining; the block number is the throughput bound. Reporting only one of the
   two would overstate or understate performance depending on the batch size.
5. **Repeats are interleaved with warmups** so that thermal drift and background
   load affect all configurations similarly.

Inputs are uniform in ``[-1, 1]`` rather than Gaussian. Random normal inputs
occasionally produce denormal activations, and denormal handling on x86 can add
tens of percent to latency for reasons that have nothing to do with the model.
Uniform inputs in the range the network actually sees avoid that artifact.

Known limitation: below roughly 0.1 ms per inference the per-iteration timer
itself becomes a significant fraction of the measurement, and the reported
coefficient of variation rises accordingly. The smallest models in this study at
32x32 reach that regime, which is precisely why every measurement also has a
*block* number (``timed_iters`` forwards inside one timer). The block figure is
the trustworthy one at the fast end; the per-iteration figure is what makes the
distribution visible for the slower cells.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from edgebench.bench.energy import EnergyMeter, EnergyReading, create_energy_meter
from edgebench.bench.footprint import measure_peak_memory
from edgebench.bench.stats import TimingStats
from edgebench.inference import Predictor
from edgebench.utils import get_logger

logger = get_logger("bench.latency")

#: Samples are rounded before storage to keep result files reviewable.
_SAMPLE_PRECISION = 5


def make_inputs(batch_size: int, resolution: int, seed: int = 0) -> torch.Tensor:
    """Deterministic benchmark input tensor.

    Seeded so that two runs see byte-identical inputs, which removes a small but
    real source of run-to-run variation (data-dependent kernel selection).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.rand(batch_size, 3, resolution, resolution, generator=generator) * 2.0 - 1.0


@dataclass
class LatencyMeasurement:
    """One cell of the latency grid: (resolution, batch size, thread count)."""

    resolution: int
    batch_size: int
    num_threads: int | None
    status: str = "ok"
    reason: str | None = None

    latency_ms: float | None = None
    per_iteration: dict[str, Any] | None = None
    block: dict[str, Any] | None = None
    throughput_samples_per_s: float | None = None
    samples_ms: list[float] = field(default_factory=list)

    memory: dict[str, Any] = field(default_factory=dict)
    energy: dict[str, Any] = field(default_factory=dict)

    warmup_iters: int = 0
    timed_iters: int = 0
    repeats: int = 0
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution,
            "batch_size": self.batch_size,
            "num_threads": self.num_threads,
            "status": self.status,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "per_iteration": self.per_iteration,
            "block": self.block,
            "throughput_samples_per_s": self.throughput_samples_per_s,
            "samples_ms": self.samples_ms,
            "memory": self.memory,
            "energy": self.energy,
            "warmup_iters": self.warmup_iters,
            "timed_iters": self.timed_iters,
            "repeats": self.repeats,
            "wall_time_s": round(self.wall_time_s, 4),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LatencyMeasurement:
        return cls(**{key: payload[key] for key in payload if key in cls.__annotations__})


def measure_latency(
    predictor: Predictor,
    resolution: int,
    batch_size: int,
    warmup_iters: int,
    timed_iters: int,
    repeats: int,
    num_threads: int | None,
    seed: int = 0,
    measure_memory: bool = True,
    energy_meter: EnergyMeter | None = None,
) -> LatencyMeasurement:
    """Measure latency/throughput/memory/energy for one configuration.

    Never raises for a backend limitation: an unsupported batch size or a device
    that cannot allocate the activations produces ``status="failed"`` with the
    reason attached, so one bad cell does not invalidate the whole grid.
    """
    measurement = LatencyMeasurement(
        resolution=resolution,
        batch_size=batch_size,
        num_threads=num_threads,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        repeats=repeats,
    )

    started = time.perf_counter()

    try:
        predictor.configure_threads(num_threads)
        inputs = make_inputs(batch_size, resolution, seed=seed)

        # ---- warmup ----------------------------------------------------------
        with torch.inference_mode():
            for _ in range(max(0, warmup_iters)):
                predictor(inputs)

        # ---- timed, per-iteration -------------------------------------------
        samples: list[float] = []
        block_samples: list[float] = []
        for _ in range(max(1, repeats)):
            with torch.inference_mode():
                for _ in range(timed_iters):
                    tick = time.perf_counter_ns()
                    predictor(inputs)
                    samples.append((time.perf_counter_ns() - tick) / 1e6)

                block_tick = time.perf_counter_ns()
                for _ in range(timed_iters):
                    predictor(inputs)
                block_samples.append(
                    (time.perf_counter_ns() - block_tick) / 1e6 / max(1, timed_iters)
                )

        per_iteration = TimingStats.from_samples_ms(samples)
        block = TimingStats.from_samples_ms(block_samples)

        measurement.per_iteration = per_iteration.to_dict()
        measurement.block = block.to_dict()
        measurement.latency_ms = per_iteration.p50
        measurement.throughput_samples_per_s = 1000.0 * batch_size / per_iteration.p50
        measurement.samples_ms = [round(value, _SAMPLE_PRECISION) for value in samples]

    except BaseException as error:
        if isinstance(error, KeyboardInterrupt | SystemExit):
            raise
        measurement.status = "failed"
        measurement.reason = f"{type(error).__name__}: {error}"
        measurement.wall_time_s = time.perf_counter() - started
        logger.warning(
            "%s: latency measurement failed at %dpx batch=%d threads=%s (%s)",
            predictor.label,
            resolution,
            batch_size,
            num_threads,
            measurement.reason,
        )
        return measurement

    # ---- secondary measurements ----------------------------------------------
    try:
        if measure_memory:
            reading = measure_peak_memory(predictor, inputs, iterations=max(1, timed_iters // 5))
            measurement.memory = reading.to_dict()
        else:
            measurement.memory = {
                "rss_peak_delta_bytes": None,
                "note": "memory measurement disabled",
            }
    except BaseException as error:
        measurement.memory = {"error": f"{type(error).__name__}: {error}"}

    meter = energy_meter or create_energy_meter()
    try:
        if meter.available:
            reading = _measure_energy(meter, predictor, inputs, max(1, timed_iters // 5))
            measurement.energy = {
                **reading.to_dict(),
                "energy_per_iteration_joules": reading.per_iteration_joules(
                    max(1, timed_iters // 5)
                ),
                "energy_per_sample_millijoules": (
                    (reading.per_iteration_joules(max(1, timed_iters // 5)) or 0.0)
                    * 1000.0
                    / batch_size
                    if reading.available
                    else None
                ),
            }
        else:
            measurement.energy = EnergyReading(
                available=False,
                joules=None,
                duration_s=0.0,
                source=meter.source,
                reason=meter.reason,
            ).to_dict()
    except BaseException as error:
        measurement.energy = {"energy_available": False, "energy_unavailable_reason": str(error)}

    measurement.wall_time_s = time.perf_counter() - started

    logger.info(
        "%s @%dpx batch=%d threads=%s: p50=%.2fms p95=%.2fms cv=%.1f%% (%.1f samples/s)",
        predictor.label,
        resolution,
        batch_size,
        num_threads if num_threads is not None else "auto",
        measurement.latency_ms or float("nan"),
        (measurement.per_iteration or {}).get("p95_ms", float("nan")),
        100 * (measurement.per_iteration or {}).get("cv", 0.0),
        measurement.throughput_samples_per_s or float("nan"),
    )

    return measurement


def _measure_energy(
    meter: EnergyMeter,
    predictor: Predictor,
    inputs: torch.Tensor,
    iterations: int,
) -> EnergyReading:
    def work() -> None:
        predictor(inputs)

    return meter.measure(work, iterations)
