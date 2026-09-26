"""Measurement: statistics, footprint, energy and the latency protocol."""

from __future__ import annotations

from edgebench.bench.energy import (
    EnergyMeter,
    EnergyReading,
    RaplEnergyMeter,
    UnavailableEnergyMeter,
    create_energy_meter,
    measure_energy,
)
from edgebench.bench.footprint import (
    MemoryReading,
    PeakMemoryMonitor,
    footprint_summary,
    measure_peak_memory,
    measure_state_bytes,
)
from edgebench.bench.latency import LatencyMeasurement, make_inputs, measure_latency
from edgebench.bench.runner import BenchmarkRecord, benchmark_one, new_run_id, static_cost
from edgebench.bench.stats import TimingStats, summarise_speedup

__all__ = [
    "BenchmarkRecord",
    "EnergyMeter",
    "EnergyReading",
    "LatencyMeasurement",
    "MemoryReading",
    "PeakMemoryMonitor",
    "RaplEnergyMeter",
    "TimingStats",
    "UnavailableEnergyMeter",
    "benchmark_one",
    "create_energy_meter",
    "footprint_summary",
    "make_inputs",
    "measure_energy",
    "measure_latency",
    "measure_peak_memory",
    "measure_state_bytes",
    "new_run_id",
    "static_cost",
    "summarise_speedup",
]
