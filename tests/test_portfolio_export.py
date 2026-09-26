"""Tests for the portfolio exporter's honesty properties.

The exporter is the last thing between the results and a public claim, so the
numbers it reports about *scope* are tested rather than trusted.

The bug this file exists to prevent: the exporter used to report the harness's
capability (5 architectures, 10 rungs, 12 latency cells per configuration) as
though it described the run. A study covering 3 models and 6 rungs therefore put
"5 architectures" and "12 latency cells" on the case study. That is a false claim
about work that was never done, and it is harder to spot than the provisional
banner because it looks like ordinary metadata.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = REPO_ROOT / "scripts" / "export_portfolio_assets.py"


def load_exporter():
    """Import the exporter as a module, without executing `main`."""
    spec = importlib.util.spec_from_file_location("export_portfolio_assets", EXPORTER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load_exporter()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeRecord:
    model_id: str
    optimization_id: str
    latency: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FakeStore:
    records: list[FakeRecord]


def cell(
    resolution: int = 32,
    batch_size: int = 1,
    num_threads: int | None = 1,
    status: str = "ok",
    *,
    omit_threads: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "resolution": resolution,
        "batch_size": batch_size,
        "status": status,
    }
    if not omit_threads:
        payload["num_threads"] = num_threads
    return payload


# ---------------------------------------------------------------------------
# _measured_scope
# ---------------------------------------------------------------------------


class TestMeasuredScope:
    def test_counts_only_the_models_present(self):
        store = FakeStore(
            [
                FakeRecord("resnet18", "fp32", [cell()]),
                FakeRecord("mobilenet_v2", "fp32", [cell()]),
            ]
        )
        assert exporter._measured_scope(store)["architectures"] == 2

    def test_counts_a_model_once_however_many_rungs_it_has(self):
        """Distinct architectures, not record count."""
        store = FakeStore(
            [
                FakeRecord("resnet18", "fp32", [cell()]),
                FakeRecord("resnet18", "static_int8", [cell()]),
                FakeRecord("resnet18", "qat_int8", [cell()]),
            ]
        )
        assert exporter._measured_scope(store)["architectures"] == 1
        assert exporter._measured_scope(store)["optimizations"] == 3

    def test_counts_unavailable_rungs_as_covered(self):
        """An unavailable rung is still part of the reported scope.

        It appears in the results as an explicit gap with a reason, so the case
        study does discuss it. Excluding it would understate the sweep.
        """
        store = FakeStore(
            [
                FakeRecord("resnet18", "fp32", [cell()]),
                FakeRecord("resnet18", "compile", []),
            ]
        )
        assert exporter._measured_scope(store)["optimizations"] == 2

    def test_reports_the_actual_scope_not_the_harness_capability(self):
        """The regression that matters: 3 models must not be reported as 5."""
        store = FakeStore(
            [
                FakeRecord("resnet18", "fp32", [cell()]),
                FakeRecord("mobilenet_v2", "fp32", [cell()]),
                FakeRecord("mobilenet_v3_small", "fp32", [cell()]),
            ]
        )
        scope = exporter._measured_scope(store)

        assert scope["architectures"] == 3
        assert scope["architectures"] != exporter.HARNESS_CAPABILITY["architectures"]

    def test_counts_auto_threads_as_a_distinct_setting(self):
        """`num_threads is None` means "let the runtime choose", not "unknown"."""
        store = FakeStore(
            [FakeRecord("resnet18", "fp32", [cell(num_threads=1), cell(num_threads=None)])]
        )
        assert exporter._measured_scope(store)["thread_counts"] == 2

    def test_an_absent_thread_field_is_not_read_as_auto(self):
        """A missing key is not a measurement setting; `None` is."""
        store = FakeStore([FakeRecord("resnet18", "fp32", [cell(omit_threads=True)])])
        scope = exporter._measured_scope(store)

        # No usable thread setting was recorded, so it falls back rather than
        # inventing an "auto" measurement.
        assert scope["thread_counts"] == exporter.HARNESS_CAPABILITY["thread_counts"]

    def test_counts_latency_dimensions_from_completed_cells(self):
        store = FakeStore(
            [
                FakeRecord(
                    "resnet18",
                    "fp32",
                    [
                        cell(resolution=32, batch_size=1, num_threads=1),
                        cell(resolution=32, batch_size=8, num_threads=1),
                    ],
                )
            ]
        )
        scope = exporter._measured_scope(store)

        assert scope["resolutions"] == 1
        assert scope["batch_sizes"] == 2
        assert scope["thread_counts"] == 1
        assert scope["resolutions"] * scope["batch_sizes"] * scope["thread_counts"] == 2

    def test_ignores_cells_that_did_not_complete(self):
        """A cell that errored was not measured, so it must not widen the scope."""
        store = FakeStore(
            [
                FakeRecord(
                    "resnet18",
                    "fp32",
                    [
                        cell(resolution=32, status="ok"),
                        cell(resolution=224, status="failed"),
                    ],
                )
            ]
        )
        assert exporter._measured_scope(store)["resolutions"] == 1

    def test_falls_back_to_capability_with_no_records(self):
        scope = exporter._measured_scope(FakeStore([]))

        assert scope == exporter.HARNESS_CAPABILITY
        # A copy, not the constant itself: mutating the result must not corrupt
        # the fallback used by every later call.
        scope["architectures"] = 999
        assert exporter.HARNESS_CAPABILITY["architectures"] == 5

    def test_tolerates_a_store_without_records(self):
        assert exporter._measured_scope(object()) == exporter.HARNESS_CAPABILITY

    def test_latency_cells_per_config_matches_the_study_scope(self):
        """The CI study measures 1 resolution x 2 batch sizes x 2 thread settings."""
        store = FakeStore(
            [
                FakeRecord(
                    "resnet18",
                    "fp32",
                    [
                        cell(resolution=32, batch_size=1, num_threads=1),
                        cell(resolution=32, batch_size=8, num_threads=1),
                        cell(resolution=32, batch_size=1, num_threads=None),
                        cell(resolution=32, batch_size=8, num_threads=None),
                    ],
                )
            ]
        )
        scope = exporter._measured_scope(store)

        cells = scope["resolutions"] * scope["batch_sizes"] * scope["thread_counts"]
        assert cells == 4
        # Not the capability figure, which is what used to be published.
        assert cells != 12


# ---------------------------------------------------------------------------
# _display_path
# ---------------------------------------------------------------------------


class TestDisplayPath:
    def test_relative_inside_the_repo(self):
        assert exporter._display_path(REPO_ROOT / "results") == str(Path("results"))

    def test_absolute_outside_the_repo(self, tmp_path):
        """`--out` may point outside the repo; printing must not crash the export."""
        outside = tmp_path / "elsewhere"
        assert exporter._display_path(outside) == str(outside)


# ---------------------------------------------------------------------------
# Documented invariants
# ---------------------------------------------------------------------------


def test_harness_capability_is_not_described_as_a_run_scope():
    """The fallback constant must not be nameable as a claim about a run."""
    assert hasattr(exporter, "HARNESS_CAPABILITY")
    assert not hasattr(exporter, "IMPLEMENTATION_FACTS"), (
        "the old name implied these counts described an implementation/run"
    )


def test_scope_keys_are_stable():
    """The case study reads these by name; renaming one breaks the page silently."""
    assert set(exporter.HARNESS_CAPABILITY) == {
        "architectures",
        "optimizations",
        "resolutions",
        "batch_sizes",
        "thread_counts",
    }


@pytest.mark.parametrize("key", ["architectures", "optimizations"])
def test_counts_are_positive_integers(key):
    assert isinstance(exporter.HARNESS_CAPABILITY[key], int)
    assert exporter.HARNESS_CAPABILITY[key] > 0
