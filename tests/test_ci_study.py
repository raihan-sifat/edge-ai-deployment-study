"""Tests for the real-data study CI scripts.

`ci_assert_study.py` is the single check standing between a synthetic run and a
public claim about real accuracy, so its failure modes are tested individually
rather than trusted. `ci_summarise_study.py` is tested for the properties that
matter to a GitHub job summary: it must be valid UTF-8 (the step-summary file
requires it) and its warnings must actually fire when the data warrants them.

The fixtures deliberately include the run-metadata file (`_run__<id>.json`) that
lives beside the measurements. It matches the same `*__*.json` glob, and an early
version of the assertion script treated it as a measurement and failed with a
confusing message, so a regression test keeps it ignored.
"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

STUDY_ASSERT = SCRIPTS / "ci_assert_study.py"
STUDY_SUMMARY = SCRIPTS / "ci_summarise_study.py"

CIFAR10_TEST_SAMPLES = 10_000


def run_script(script: Path, *argv: str) -> int:
    """Execute a script in-process and return its exit code.

    `runpy` is used rather than a subprocess so a failure surfaces as ordinary
    pytest output. The caller is responsible for `monkeypatch.chdir`, since the
    assertion script resolves its paths relative to the working directory -- which
    is how CI invokes it.
    """
    original_argv = sys.argv
    sys.argv = [str(script), *argv]

    added = False
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        added = True

    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    finally:
        sys.argv = original_argv
        if added:
            sys.path.remove(script_dir)
    return 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def write_study_record(
    raw_dir: Path,
    optimization_id: str,
    *,
    status: str = "applied",
    reason: str | None = None,
    num_samples: int = CIFAR10_TEST_SAMPLES,
    top1: float = 0.25,
    latency: bool = True,
    cv: float = 0.05,
    model_id: str = "resnet18",
) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": "study-run",
        "model_id": model_id,
        "optimization_id": optimization_id,
        "status": status,
        "reason": reason,
        "accuracy": {"top1": top1, "top5": 0.9, "loss": 1.0, "num_samples": num_samples},
        "footprint": {"weight_bytes": 42_000_000},
        "environment": {
            "cpu": "Test CPU",
            "fingerprint": {"torch": "2.14.0+cpu", "python": "3.14.5", "onnxruntime": "1.30.0"},
        },
        "latency": (
            [
                {
                    "resolution": 32,
                    "batch_size": 1,
                    "num_threads": 1,
                    "status": "ok",
                    "latency_ms": 10.0,
                    "per_iteration": {"p50_ms": 10.0, "cv": cv},
                    "samples_ms": [9.8, 10.1, 10.0],
                }
            ]
            if latency
            else []
        ),
    }
    path = raw_dir / f"{model_id}__{optimization_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_study_export(
    root: Path,
    *,
    dataset: str = "cifar10",
    provisional: bool = False,
    provisional_reason: str | None = None,
    applied_rows: int = 3,
    unstable_rows: int = 0,
) -> None:
    """Write the portfolio export that the assertion reads last."""
    portfolio = root / "portfolio" / "data"
    portfolio.mkdir(parents=True, exist_ok=True)

    headline = []
    for index in range(applied_rows):
        headline.append(
            {
                "model": "ResNet-18",
                "optimization": f"opt{index}",
                "status": "applied",
                "unstable": index < unstable_rows,
            }
        )

    payload = {
        "dataset": dataset,
        "provisional": provisional,
        "provisional_reason": provisional_reason,
        "headline": headline,
    }
    (portfolio / "results.json").write_text(json.dumps(payload), encoding="utf-8")


def build_study_tree(
    root: Path,
    *,
    num_samples: int = CIFAR10_TEST_SAMPLES,
    top1: float = 0.25,
    latency: bool = True,
    cv: float = 0.05,
    unavailable_without_reason: bool = False,
    with_run_metadata: bool = True,
    with_figures: bool = True,
    with_tables: bool = True,
    with_export: bool = True,
    export_dataset: str = "cifar10",
    export_provisional: bool = False,
    export_provisional_reason: str | None = None,
) -> None:
    """Fabricate a real-CIFAR-10 study tree, including the portfolio export."""
    raw = root / "results" / "study" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    write_study_record(raw, "fp32", num_samples=num_samples, top1=top1, latency=latency, cv=cv)
    write_study_record(
        raw, "static_int8", num_samples=num_samples, top1=top1, latency=latency, cv=cv
    )

    if unavailable_without_reason:
        write_study_record(raw, "compile", status="unavailable", reason=None)
    else:
        write_study_record(
            raw, "compile", status="unavailable", reason="no working backend on this machine"
        )

    if with_run_metadata:
        # Share the `*__*.json` shape with the measurements. Not a measurement.
        (raw / "_run__20260926T000000Z.json").write_text(
            json.dumps({"run_id": "study-run", "started_at": "2026-09-26T00:00:00Z"}),
            encoding="utf-8",
        )

    figures = root / "results" / "study" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if with_figures:
        (figures / "accuracy_vs_latency.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    tables = root / "results" / "study" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if with_tables:
        (tables / "main_comparison.csv").write_text("model,opt\nresnet18,fp32\n", encoding="utf-8")

    if with_export:
        write_study_export(
            root,
            dataset=export_dataset,
            provisional=export_provisional,
            provisional_reason=export_provisional_reason,
        )


# ---------------------------------------------------------------------------
# ci_assert_study.py
# ---------------------------------------------------------------------------


class TestStudyAssertions:
    def test_passes_on_a_healthy_tree(self, tmp_path, monkeypatch):
        build_study_tree(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 0

    def test_ignores_the_run_metadata_file(self, tmp_path, monkeypatch):
        """`_run__<id>.json` matches the measurement glob but is not a measurement.

        Regression test: treating it as one made every field look absent, and the
        script reported the unrelated "not applied but no reason recorded".
        """
        build_study_tree(tmp_path, with_run_metadata=True)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 0

    def test_fails_without_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_on_synthetic_sample_count(self, tmp_path, monkeypatch):
        """512 samples is the synthetic path's signature, not CIFAR-10's test split."""
        build_study_tree(tmp_path, num_samples=512)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_when_export_still_says_synthetic(self, tmp_path, monkeypatch):
        """The critical guard: a synthetic run must never be published as real."""
        build_study_tree(tmp_path, export_dataset="synthetic", export_provisional=True)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_when_export_is_provisional_despite_real_data(self, tmp_path, monkeypatch):
        """Real data plus a provisional flag means fallback or broken detection."""
        build_study_tree(
            tmp_path,
            export_dataset="cifar10",
            export_provisional=True,
            export_provisional_reason="synthetic tensors detected",
        )
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_on_inconsistent_export(self, tmp_path, monkeypatch):
        """A stale `provisional_reason` alongside `provisional: false` is incoherent."""
        build_study_tree(
            tmp_path,
            export_dataset="cifar10",
            export_provisional=False,
            export_provisional_reason="left over from an earlier run",
        )
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_without_a_portfolio_export(self, tmp_path, monkeypatch):
        build_study_tree(tmp_path, with_export=False)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_when_export_has_no_applied_rows(self, tmp_path, monkeypatch):
        build_study_tree(tmp_path)
        write_study_export(tmp_path, applied_rows=0)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_when_an_unavailable_rung_has_no_reason(self, tmp_path, monkeypatch):
        """An unexplained gap in the results matrix is the thing this prevents."""
        build_study_tree(tmp_path, unavailable_without_reason=True)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_when_accuracy_is_at_chance(self, tmp_path, monkeypatch):
        build_study_tree(tmp_path, top1=0.05)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_when_accuracy_is_implausibly_high(self, tmp_path, monkeypatch):
        """The test-set-leak detector: a from-scratch run cannot reach this."""
        build_study_tree(tmp_path, top1=0.99)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_when_the_latency_grid_is_empty(self, tmp_path, monkeypatch):
        build_study_tree(tmp_path, latency=False)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_without_figures(self, tmp_path, monkeypatch):
        build_study_tree(tmp_path, with_figures=False)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_fails_without_tables(self, tmp_path, monkeypatch):
        build_study_tree(tmp_path, with_tables=False)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 1

    def test_passes_with_unstable_latency(self, tmp_path, monkeypatch):
        """Dispersion on a shared runner is reported, not treated as failure.

        Accuracy is unaffected by a noisy clock, and failing here would make the
        workflow flaky for a reason that has nothing to do with correctness.
        """
        build_study_tree(tmp_path, cv=0.42)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 0

    def test_passes_at_the_edges_of_the_plausible_band(self, tmp_path, monkeypatch):
        build_study_tree(tmp_path, top1=0.16)
        monkeypatch.chdir(tmp_path)
        assert run_script(STUDY_ASSERT) == 0


# ---------------------------------------------------------------------------
# ci_summarise_study.py
# ---------------------------------------------------------------------------


def run_summary(cwd: Path, *argv: str) -> subprocess.CompletedProcess[bytes]:
    """Run the summary script, capturing raw bytes.

    Bytes rather than text, because the encoding *is* the property under test.
    """
    return subprocess.run(
        [sys.executable, str(STUDY_SUMMARY), *argv],
        cwd=cwd,
        capture_output=True,
        check=False,
    )


class TestStudySummary:
    def test_output_is_valid_utf8(self, tmp_path):
        """`$GITHUB_STEP_SUMMARY` requires UTF-8, and the tables use non-ASCII.

        The Windows console default (cp1252) cannot encode the delta sign, which
        made this fail locally while passing in CI.
        """
        build_study_tree(tmp_path)
        result = run_summary(tmp_path)

        assert result.returncode == 0
        text = result.stdout.decode("utf-8")  # raises if not UTF-8
        assert "\u0394" in text  # the delta sign, present as real UTF-8

    def test_reports_the_machine_and_stack(self, tmp_path):
        build_study_tree(tmp_path)
        text = run_summary(tmp_path).stdout.decode("utf-8")

        assert "Test CPU" in text
        assert "2.14.0+cpu" in text
        assert "1.30.0" in text

    def test_counts_applied_and_not_applied(self, tmp_path):
        build_study_tree(tmp_path)
        text = run_summary(tmp_path).stdout.decode("utf-8")

        assert "2 applied, 1 not applied" in text

    def test_lists_the_reason_for_unapplied_rungs(self, tmp_path):
        """The reason is the useful part; omitting it leaves an unexplained gap."""
        build_study_tree(tmp_path)
        text = run_summary(tmp_path).stdout.decode("utf-8")

        assert "Not applied" in text
        assert "no working backend on this machine" in text

    def test_warns_when_latency_is_unstable(self, tmp_path):
        build_study_tree(tmp_path, cv=0.40)
        text = run_summary(tmp_path).stdout.decode("utf-8")

        assert "unstable latency" in text
        assert "indicative" in text

    def test_does_not_warn_when_latency_is_stable(self, tmp_path):
        build_study_tree(tmp_path, cv=0.02)
        text = run_summary(tmp_path).stdout.decode("utf-8")

        assert "unstable latency" not in text

    def test_ignores_the_run_metadata_file(self, tmp_path):
        """Same regression as the assertion script: metadata is not a measurement."""
        build_study_tree(tmp_path, with_run_metadata=True)
        text = run_summary(tmp_path).stdout.decode("utf-8")

        assert "2 applied, 1 not applied" in text

    def test_succeeds_when_there_are_no_results(self, tmp_path):
        """A missing results tree is reported, not raised.

        The summary step runs even when training failed, and the job summary is
        where a reader looks to find out why. Crashing there would hide the cause.
        """
        result = run_summary(tmp_path)

        assert result.returncode == 0
        assert "No results found" in result.stdout.decode("utf-8")

    def test_succeeds_when_nothing_applied(self, tmp_path):
        raw = tmp_path / "results" / "study" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        write_study_record(raw, "compile", status="unavailable", reason="no backend")

        result = run_summary(tmp_path)

        assert result.returncode == 0
        assert "Nothing applied" in result.stdout.decode("utf-8")

    def test_accepts_an_alternative_results_directory(self, tmp_path):
        build_study_tree(tmp_path)
        moved = tmp_path / "elsewhere"
        (tmp_path / "results" / "study").rename(moved)

        result = run_summary(tmp_path, "--results", "elsewhere")

        assert result.returncode == 0
        assert "2 applied, 1 not applied" in result.stdout.decode("utf-8")
