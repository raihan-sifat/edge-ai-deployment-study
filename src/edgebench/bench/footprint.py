"""Memory accounting for edge deployment.

Two numbers are reported, and they answer different questions:

``weight_bytes``
    The serialized size of the parameters. This is what occupies flash and what a
    device manifest quotes.

``peak_rss_delta_bytes``
    The peak resident-set growth observed while running inference. This is what
    competes for RAM with the rest of the appliance, and for a small model it is
    dominated by activation tensors rather than by weights.

The RSS measurement is process-wide and therefore includes Python, PyTorch and
allocator overhead. It is a *delta* against a baseline taken immediately before the
run, which removes most of that constant. It is sampled rather than exact -- CPU
Torch offers no peak-allocation hook -- so it is reported as an estimate with the
sampling interval recorded, never as a precise figure.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from edgebench.inference import Predictor, measure_state_dict_bytes
from edgebench.utils import human_bytes

DEFAULT_SAMPLE_INTERVAL_S = 0.0005


@dataclass(frozen=True)
class MemoryReading:
    """Outcome of a peak-memory measurement."""

    baseline_bytes: int
    peak_bytes: int
    delta_bytes: int
    samples: int
    interval_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rss_baseline_bytes": self.baseline_bytes,
            "rss_peak_bytes": self.peak_bytes,
            "rss_peak_delta_bytes": self.delta_bytes,
            "rss_samples": self.samples,
            "rss_sample_interval_s": self.interval_s,
        }


class PeakMemoryMonitor:
    """Context manager that samples resident-set size on a background thread."""

    def __init__(self, interval_s: float = DEFAULT_SAMPLE_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self._peak = 0
        self._baseline = 0
        self._samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any = None

    def __enter__(self) -> PeakMemoryMonitor:
        try:
            import psutil
        except ImportError:  # pragma: no cover - psutil is a declared dependency
            self._process = None
            return self

        self._process = psutil.Process()
        self._baseline = int(self._process.memory_info().rss)
        self._peak = self._baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                rss = int(self._process.memory_info().rss)
            except Exception:  # pragma: no cover - process may be closing
                return
            if rss > self._peak:
                self._peak = rss
            self._samples += 1
            time.sleep(self.interval_s)

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def reading(self) -> MemoryReading:
        return MemoryReading(
            baseline_bytes=self._baseline,
            peak_bytes=self._peak,
            delta_bytes=max(0, self._peak - self._baseline),
            samples=self._samples,
            interval_s=self.interval_s,
        )


def measure_peak_memory(
    predictor: Predictor,
    example_inputs: torch.Tensor,
    iterations: int = 5,
    interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
) -> MemoryReading:
    """Run inference ``iterations`` times and report the peak RSS delta."""
    with PeakMemoryMonitor(interval_s=interval_s) as monitor:
        for _ in range(max(1, iterations)):
            predictor(example_inputs)  # the return value is deliberately discarded

    return monitor.reading


def footprint_summary(
    predictor: Predictor,
    model: nn.Module | None,
    example_inputs: torch.Tensor,
    input_size: int,
    measure_memory: bool = True,
) -> dict[str, Any]:
    """Collect the full footprint block for one (model, optimization) pair."""
    weight_bytes = predictor.state_bytes()

    summary: dict[str, Any] = {
        "weight_bytes": weight_bytes,
        "weight_mib": round(weight_bytes / (1024 * 1024), 4),
        "weight_human": human_bytes(weight_bytes),
    }

    if model is not None:
        parameters = sum(p.numel() for p in model.parameters())
        buffers = sum(b.numel() for b in model.buffers())
        summary.update(
            {
                "parameters": parameters,
                "buffers": buffers,
                "parameter_bytes_fp32": parameters * 4,
                "compression_ratio": (parameters * 4) / weight_bytes if weight_bytes else None,
            }
        )

    if measure_memory:
        reading = measure_peak_memory(predictor, example_inputs)
        summary.update(reading.to_dict())
    else:
        summary.update(
            {
                "rss_baseline_bytes": None,
                "rss_peak_bytes": None,
                "rss_peak_delta_bytes": None,
                "rss_samples": 0,
                "rss_sample_interval_s": None,
            }
        )

    # Activation memory scales with pixels per image, and that scaling is the real
    # reason a 224x224 deployment needs more RAM than a 32x32 one.
    summary["input_pixels"] = int(example_inputs.shape[-2] * example_inputs.shape[-1])
    summary["batch_size_for_memory"] = int(example_inputs.shape[0])

    return summary


def measure_state_bytes(module: nn.Module) -> int:
    """Convenience re-export so callers do not need to import ``inference``."""
    return measure_state_dict_bytes(module)
