"""Energy measurement, including an honest answer when it is impossible.

Energy per inference is arguably the metric that matters most on a battery- or
thermally-limited device, and it is the one most often faked by reporting CPU
utilisation or a TDP-derived estimate. This module refuses to do that.

It measures energy only when the platform exposes a real counter -- Intel/AMD
RAPL via ``/sys/class/powercap`` on Linux -- and otherwise returns an explicit
``available: False`` with the reason. On Windows and macOS there is no
unprivileged, per-process energy counter, so the study reports energy as
"not measured" and says so in the results table instead of inventing a number.

That refusal is a feature. A benchmark whose weakest metric is fabricated is not
a benchmark.
"""

from __future__ import annotations

import platform
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edgebench.utils import get_logger

logger = get_logger("bench.energy")

RAPL_ROOT = Path("/sys/class/powercap")


@dataclass(frozen=True)
class EnergyReading:
    """Energy consumed over a measured interval."""

    available: bool
    joules: float | None
    duration_s: float
    source: str
    reason: str | None = None

    def per_iteration_joules(self, iterations: int) -> float | None:
        if not self.available or self.joules is None or iterations <= 0:
            return None
        return self.joules / iterations

    def average_watts(self) -> float | None:
        if not self.available or self.joules is None or self.duration_s <= 0:
            return None
        return self.joules / self.duration_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy_available": self.available,
            "energy_joules": self.joules,
            "energy_duration_s": self.duration_s,
            "energy_source": self.source,
            "energy_unavailable_reason": self.reason,
            "energy_avg_watts": self.average_watts(),
        }


class EnergyMeter(ABC):
    """Measure energy consumed while running a callable."""

    source: str = "none"

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this meter can produce a real measurement."""

    @property
    def reason(self) -> str | None:
        """Why measurement is impossible, when it is."""
        return None

    @abstractmethod
    def measure(self, work: Callable[[], None], iterations: int) -> EnergyReading:
        """Run ``work`` ``iterations`` times and report the energy consumed."""


class UnavailableEnergyMeter(EnergyMeter):
    """Meter used whenever no hardware counter exists."""

    source = "unavailable"

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def available(self) -> bool:
        return False

    @property
    def reason(self) -> str | None:
        return self._reason

    def measure(self, work: Callable[[], None], iterations: int) -> EnergyReading:
        started = time.perf_counter()
        for _ in range(max(1, iterations)):
            work()
        return EnergyReading(
            available=False,
            joules=None,
            duration_s=time.perf_counter() - started,
            source=self.source,
            reason=self._reason,
        )


class RaplEnergyMeter(EnergyMeter):
    """Intel/AMD Running Average Power Limit counter, via ``powercap`` sysfs.

    Reads every ``intel-rapl:N`` package domain, sums the deltas, and handles
    counter wrap-around using the advertised ``max_energy_range_uj``. The reading
    covers the whole CPU package, not this process, so it is only meaningful when
    the machine is otherwise idle -- which is why the benchmark runs single-threaded
    and the report notes that the host must be quiet.
    """

    source = "intel-rapl"

    def __init__(self, domains: list[tuple[Path, int]]) -> None:
        self._domains = domains

    @classmethod
    def create(cls) -> RaplEnergyMeter | None:
        """Return a meter, or ``None`` when RAPL is not readable."""
        if not RAPL_ROOT.exists():
            return None

        domains: list[tuple[Path, int]] = []
        for energy_file in sorted(RAPL_ROOT.glob("intel-rapl:*/energy_uj")):
            try:
                energy_file.read_text(encoding="utf-8")
            except (PermissionError, OSError):
                logger.debug("RAPL domain %s is not readable", energy_file)
                continue

            range_file = energy_file.with_name("max_energy_range_uj")
            try:
                max_range = int(range_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                max_range = 2**32
            domains.append((energy_file, max_range))

        return cls(domains) if domains else None

    @property
    def available(self) -> bool:
        return bool(self._domains)

    def _read_total_uj(self) -> int:
        total = 0
        for energy_file, _ in self._domains:
            total += int(energy_file.read_text(encoding="utf-8").strip())
        return total

    def measure(self, work: Callable[[], None], iterations: int) -> EnergyReading:
        try:
            before = self._read_total_uj()
            started = time.perf_counter()
            for _ in range(max(1, iterations)):
                work()
            duration = time.perf_counter() - started
            after = self._read_total_uj()
        except (OSError, PermissionError) as error:
            return EnergyReading(
                available=False,
                joules=None,
                duration_s=0.0,
                source=self.source,
                reason=f"RAPL read failed: {error}",
            )

        delta_uj = after - before
        if delta_uj < 0:
            # Counter wrapped; recover using the advertised range.
            delta_uj += sum(max_range for _, max_range in self._domains)

        return EnergyReading(
            available=True,
            joules=delta_uj / 1e6,
            duration_s=duration,
            source=self.source,
        )


def create_energy_meter() -> EnergyMeter:
    """Best available energy meter for this platform, with a documented fallback."""
    if sys.platform.startswith("linux"):
        meter = RaplEnergyMeter.create()
        if meter is not None:
            return meter
        return UnavailableEnergyMeter(
            "no readable RAPL domain under /sys/class/powercap (needs an Intel/AMD "
            "package with powercap enabled and read permission)"
        )

    if sys.platform == "darwin":
        return UnavailableEnergyMeter(
            "macOS exposes no unprivileged per-package energy counter; "
            "use `powermetrics` as root for an external measurement"
        )

    if sys.platform.startswith("win"):
        return UnavailableEnergyMeter(
            "Windows exposes no per-package energy counter through a supported API; "
            "an external wall-plug meter is required"
        )

    return UnavailableEnergyMeter(f"no energy counter implementation for {platform.system()}")


def measure_energy(
    work: Callable[[], None],
    iterations: int,
    meter: EnergyMeter | None = None,
) -> EnergyReading:
    """Convenience wrapper around :meth:`EnergyMeter.measure`."""
    return (meter or create_energy_meter()).measure(work, iterations)
