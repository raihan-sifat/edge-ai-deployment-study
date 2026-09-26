"""Tests for the CI assertion scripts.

These scripts are the difference between CI that is meaningful and CI that is
decorative. `edgebench run_all` and `edgebench report` both exit zero when they
produce nothing -- deliberately, so a partial sweep still yields partial output
and a long run is not thrown away. That means an exit-code check proves nothing,
and the assertion scripts are what actually verify content.

Untested assertion scripts are the same category of problem they exist to
prevent, so their logic is tested here against fabricated fixtures, in both the
passing and failing directions.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

SMOKE_ASSERT = SCRIPTS / "ci_assert_smoke.py"
REAL_ASSERT = SCRIPTS / "ci_assert_real_data.py"
REPORT_ASSERT = SCRIPTS / "ci_assert_report.py"


def run_assertion(script: Path) -> int:
    """Execute an assertion script and return its exit code.

    ``runpy`` is used rather than a subprocess so failures surface as ordinary
    pytest output instead of a captured process result. The caller is responsible
    for ``monkeypatch.chdir``, because the scripts resolve their paths relative to
    the working directory -- which is exactly how CI invokes them.
    """
    original_argv = sys.argv
    sys.argv = [str(script)]

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
# Fixtures: fabricate just enough of a results tree
# ---------------------------------------------------------------------------


def write_smoke_record(
    raw_dir: Path,
    optimization_id: str,
    *,
    status: str = "applied",
    reason: str | None = None,
    num_samples: int = 512,
    latency: bool = True,
    weight_bytes: int = 42_000_000,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": "test-run",
        "model_id": "resnet18",
        "optimization_id": optimization_id,
        "status": status,
        "reason": reason,
        "accuracy": {"top1": 0.5, "top5": 0.9, "loss": 1.0, "num_samples": num_samples},
        "footprint": {"weight_bytes": weight_bytes},
        "latency": (
            [
                {
                    "resolution": 32,
                    "batch_size": 1,
                    "num_threads": 1,
                    "status": "ok",
                    "latency_ms": 10.0,
                    "per_iteration": {"p50_ms": 10.0, "cv": 0.05},
                    "samples_ms": [9.8, 10.1, 10.0],
                }
            ]
            if latency
            else []
        ),
    }
    path = raw_dir / f"resnet18__{optimization_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def build_smoke_tree(root: Path, applied: int = 4, skipped: int = 1) -> None:
    raw = root / "results" / "offline" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    for index in range(applied):
        write_smoke_record(raw, f"opt{index}")
    for index in range(skipped):
        write_smoke_record(raw, f"skip{index}", status="unavailable", reason="not supported here")

    (root / "results" / "offline" / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results" / "offline" / "figures" / "accuracy_vs_latency.png").write_bytes(
        b"\x89PNG\r\n\x1a\n"
    )

    tables = root / "results" / "offline" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "main_comparison.csv").write_text("model,opt\nresnet18,fp32\n", encoding="utf-8")
    (tables / "manifest.json").write_text("{}", encoding="utf-8")

    portfolio = root / "portfolio" / "data"
    portfolio.mkdir(parents=True, exist_ok=True)
    (portfolio / "results.json").write_text(
        json.dumps(
            {
                "provisional": True,
                "dataset": "synthetic",
                "headline": [{"model": "ResNet-18", "optimization": "FP32"}],
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# ci_assert_smoke.py
# ---------------------------------------------------------------------------


class TestSmokeAssertions:
    def test_passes_on_a_healthy_tree(self, tmp_path, monkeypatch):
        build_smoke_tree(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 0

    def test_fails_without_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 1

    def test_fails_when_too_few_records(self, tmp_path, monkeypatch):
        build_smoke_tree(tmp_path, applied=1, skipped=0)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 1

    def test_fails_when_no_optimization_applied(self, tmp_path, monkeypatch):
        build_smoke_tree(tmp_path, applied=0, skipped=5)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 1

    def test_fails_when_skipped_record_has_no_reason(self, tmp_path, monkeypatch):
        """A gap in the results must be explainable, not silent."""
        build_smoke_tree(tmp_path)
        raw = tmp_path / "results" / "offline" / "raw"
        write_smoke_record(raw, "skip0", status="unavailable", reason=None)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 1

    def test_fails_when_latency_grid_is_empty(self, tmp_path, monkeypatch):
        build_smoke_tree(tmp_path)
        raw = tmp_path / "results" / "offline" / "raw"
        write_smoke_record(raw, "opt0", latency=False)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 1

    def test_fails_when_weight_bytes_missing(self, tmp_path, monkeypatch):
        build_smoke_tree(tmp_path)
        raw = tmp_path / "results" / "offline" / "raw"
        write_smoke_record(raw, "opt0", weight_bytes=0)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 1

    def test_fails_on_the_wrong_sample_count(self, tmp_path, monkeypatch):
        """Synthetic CIFAR-10 is 512 test images; anything else means a bad split."""
        build_smoke_tree(tmp_path)
        raw = tmp_path / "results" / "offline" / "raw"
        write_smoke_record(raw, "opt0", num_samples=10_000)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 1

    def test_fails_when_provisional_flag_is_missing(self, tmp_path, monkeypatch):
        """The placeholder-data guard is the reason this check exists."""
        build_smoke_tree(tmp_path)
        portfolio = tmp_path / "portfolio" / "data" / "results.json"
        portfolio.write_text(
            json.dumps({"provisional": False, "dataset": "synthetic", "headline": [{"a": 1}]}),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 1

    def test_fails_when_no_figures_were_generated(self, tmp_path, monkeypatch):
        build_smoke_tree(tmp_path)
        for figure in (tmp_path / "results" / "offline" / "figures").glob("*.png"):
            figure.unlink()
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 1

    def test_fails_when_portfolio_export_is_missing(self, tmp_path, monkeypatch):
        build_smoke_tree(tmp_path)
        (tmp_path / "portfolio" / "data" / "results.json").unlink()
        monkeypatch.chdir(tmp_path)
        assert run_assertion(SMOKE_ASSERT) == 1


# ---------------------------------------------------------------------------
# ci_assert_report.py
# ---------------------------------------------------------------------------


class TestReportAssertions:
    def _write_report(self, root: Path, body: str) -> None:
        docs = root / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "report.html").write_text(body, encoding="utf-8")

    def test_passes_on_a_complete_report(self, tmp_path, monkeypatch, repo_root):
        # Reuse the real rendered report if it exists; otherwise skip.
        real = repo_root / "docs" / "report.html"
        if not real.exists():
            pytest.skip("docs/report.html not built; run scripts/build_report.py first")
        self._write_report(tmp_path, real.read_text(encoding="utf-8"))
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REPORT_ASSERT) == 0

    def test_fails_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REPORT_ASSERT) == 1

    def test_fails_when_too_small(self, tmp_path, monkeypatch):
        self._write_report(tmp_path, "<html><body><h1>x</h1></body></html>")
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REPORT_ASSERT) == 1

    def test_fails_when_the_title_is_absent(self, tmp_path, monkeypatch, repo_root):
        """A large file with no expected content means the wrong document rendered."""
        real = repo_root / "docs" / "report.html"
        if not real.exists():
            pytest.skip("docs/report.html not built")
        body = real.read_text(encoding="utf-8").replace(
            "Benchmarking Efficient Deep Learning Models", "Some Other Project"
        )
        self._write_report(tmp_path, body)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REPORT_ASSERT) == 1

    def test_fails_when_tables_were_dropped(self, tmp_path, monkeypatch, repo_root):
        """Catches a renderer that silently stops emitting pipe tables."""
        real = repo_root / "docs" / "report.html"
        if not real.exists():
            pytest.skip("docs/report.html not built")
        body = real.read_text(encoding="utf-8").replace("<table", "<div")
        self._write_report(tmp_path, body)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REPORT_ASSERT) == 1


# ---------------------------------------------------------------------------
# ci_assert_real_data.py
# ---------------------------------------------------------------------------


def build_real_tree(
    root: Path,
    *,
    top1: float = 0.25,
    num_samples: int = 10_000,
    train_top1: float = 0.30,
    with_history: bool = True,
    with_checkpoint: bool = True,
) -> None:
    """Fabricate a real-CIFAR-10 result tree, including a checkpoint."""
    import torch

    raw = root / "results" / "smoke" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    write_smoke_record(raw, "fp32", num_samples=num_samples)
    record = json.loads((raw / "resnet18__fp32.json").read_text(encoding="utf-8"))
    record["accuracy"]["top1"] = top1
    (raw / "resnet18__fp32.json").write_text(json.dumps(record), encoding="utf-8")

    if not with_checkpoint:
        return

    checkpoints = root / "checkpoints" / "smoke"
    checkpoints.mkdir(parents=True, exist_ok=True)

    epochs = [{"epoch": 1.0, "train_top1": train_top1, "val_top1": top1}] if with_history else []
    metadata = {"training_history": {"epochs": epochs}, "saved_at": "2026-09-26T00:00:00Z"}

    torch.save(
        {
            "format_version": 2,
            "model_id": "resnet18",
            "num_classes": 10,
            "input_size": 32,
            "state_dict": {"linear.weight": torch.zeros(2, 2)},
            "metadata": metadata,
        },
        checkpoints / "resnet18__fp32.pt",
    )


class TestRealDataAssertions:
    def test_passes_on_a_healthy_tree(self, tmp_path, monkeypatch):
        build_real_tree(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REAL_ASSERT) == 0

    def test_fails_without_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REAL_ASSERT) == 1

    def test_fails_on_wrong_test_sample_count(self, tmp_path, monkeypatch):
        """A different sample count means the wrong dataset or a changed split."""
        build_real_tree(tmp_path, num_samples=512)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REAL_ASSERT) == 1

    def test_fails_when_checkpoint_is_missing(self, tmp_path, monkeypatch):
        build_real_tree(tmp_path, with_checkpoint=False)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REAL_ASSERT) == 1

    def test_fails_when_checkpoint_has_no_training_history(self, tmp_path, monkeypatch):
        """A cache hit means the run loaded weights instead of training them."""
        build_real_tree(tmp_path, with_history=False)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REAL_ASSERT) == 1

    def test_fails_when_accuracy_is_at_chance(self, tmp_path, monkeypatch):
        """Near-chance top-1 means the training loop is not learning."""
        build_real_tree(tmp_path, top1=0.05, train_top1=0.05)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REAL_ASSERT) == 1

    def test_fails_when_accuracy_is_implausibly_high(self, tmp_path, monkeypatch):
        """The test-set-leak detector: one epoch cannot reach 90%."""
        build_real_tree(tmp_path, top1=0.93)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REAL_ASSERT) == 1

    def test_fails_when_training_accuracy_is_at_chance(self, tmp_path, monkeypatch):
        build_real_tree(tmp_path, top1=0.25, train_top1=0.10)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REAL_ASSERT) == 1

    def test_passes_at_the_edges_of_the_plausible_band(self, tmp_path, monkeypatch):
        build_real_tree(tmp_path, top1=0.12, train_top1=0.30)
        monkeypatch.chdir(tmp_path)
        assert run_assertion(REAL_ASSERT) == 0
