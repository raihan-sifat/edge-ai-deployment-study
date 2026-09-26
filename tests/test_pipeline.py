"""End-to-end pipeline and reporting, using the offline synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgebench.bench.runner import BenchmarkRecord, build_model_card
from edgebench.config import (
    BenchmarkConfig,
    Config,
    DataConfig,
    ModelEntry,
    OptimizationEntry,
    PathsConfig,
    TrainConfig,
)
from edgebench.reporting import build_report
from edgebench.results import load_store
from edgebench.utils import json_dump


def offline_config(tmp_path: Path, optimizations: tuple[str, ...] = ("fp32",)) -> Config:
    return Config(
        seed=0,
        data=DataConfig(
            name="synthetic",
            root=str(tmp_path / "data"),
            val_size=0,
            num_workers=0,
            augment=False,
            download=False,
        ),
        train=TrainConfig(
            epochs=1,
            batch_size=16,
            lr=0.05,
            scheduler="none",
            warmup_epochs=0,
            label_smoothing=0.0,
            log_interval=0,
            qat_epochs=1,
        ),
        benchmark=BenchmarkConfig(
            device="cpu",
            resolutions=(32,),
            batch_sizes=(1,),
            warmup_iters=1,
            timed_iters=2,
            repeats=1,
            thread_counts=(1,),
        ),
        paths=PathsConfig(
            checkpoints=str(tmp_path / "checkpoints"),
            results=str(tmp_path / "results"),
            figures=str(tmp_path / "results" / "figures"),
            tables=str(tmp_path / "results" / "tables"),
            raw=str(tmp_path / "results" / "raw"),
        ),
        optimizations=tuple(OptimizationEntry(id=name) for name in optimizations),
        models=(ModelEntry(id="resnet18"),),
        root=tmp_path,
    )


@pytest.mark.network
def test_offline_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    """The whole harness, on synthetic data, in a few seconds."""
    from edgebench import pipeline

    config = offline_config(tmp_path, optimizations=("fp32", "prune_unstructured_50"))
    summary = pipeline.run_all(config, measure_memory=False)

    assert len(summary.records) == 2
    assert summary.applied == 2

    raw_files = sorted(config.raw_dir.glob("resnet18__*.json"))
    assert len(raw_files) == 2

    record = BenchmarkRecord.from_dict(json.loads(raw_files[0].read_text(encoding="utf-8")))
    assert record.status == "applied"
    assert record.accuracy is not None
    assert record.latency, "latency grid must not be empty"
    assert record.footprint.get("parameters", 0) > 0
    assert record.environment

    del monkeypatch


def test_records_survive_a_report_round_trip(tmp_path):
    """Reporting must be pure: build a record by hand, then derive charts from it."""
    record = BenchmarkRecord(
        run_id="test-run",
        model_id="resnet18",
        optimization_id="fp32",
        status="applied",
        accuracy={
            "top1": 0.9,
            "top5": 0.99,
            "loss": 0.3,
            "num_samples": 10000,
            "ece": 0.04,
            "per_class_accuracy": [0.9] * 10,
        },
        footprint={"weight_bytes": 44_000_000, "parameters": 11_000_000, "macs": 5.5e8},
        latency=[
            {
                "resolution": 32,
                "batch_size": 1,
                "num_threads": 1,
                "status": "ok",
                "latency_ms": 10.0,
                "per_iteration": {"p50_ms": 10.0, "p95_ms": 14.0, "cv": 0.08},
                "throughput_samples_per_s": 100.0,
                "samples_ms": [9.5, 10.2, 10.1, 9.8],
                "memory": {},
                "energy": {},
            }
        ],
        model_card=build_model_card("resnet18", "resnet18", 10, 32, {"macs": 5.5e8}),
        environment={"cpu": "test cpu", "platform": "test", "fingerprint": {"torch": "2.14"}},
    )

    json_dump(record.to_dict(), tmp_path / "raw" / "resnet18__fp32.json")

    store = load_store(tmp_path)
    assert len(store.records) == 1
    assert store.model_ids() == ["resnet18"]

    frame = store.summary_frame()
    assert len(frame) == 1
    assert frame.iloc[0]["top1"] == pytest.approx(0.9)
    assert frame.iloc[0]["latency_ms"] == pytest.approx(10.0)

    artifacts = build_report(tmp_path)
    assert artifacts.figures, "at least one figure should be produced"
    assert (tmp_path / "tables" / "main_comparison.md").exists()
    assert (tmp_path / "tables" / "manifest.json").exists()


def test_report_handles_an_empty_results_directory(tmp_path):
    artifacts = build_report(tmp_path)
    assert artifacts.figures == []
    assert "no records found" in artifacts.skipped


# ---------------------------------------------------------------------------
# Sweeps that need more than one point on their axis
# ---------------------------------------------------------------------------


def write_sweep_record(
    root: Path,
    cells: list[tuple[int, int]],
    *,
    model_id: str = "resnet18",
    optimization_id: str = "fp32",
) -> None:
    """Write one record whose latency grid covers the given (resolution, batch) cells."""
    latency = [
        {
            "resolution": resolution,
            "batch_size": batch_size,
            "num_threads": 1,
            "status": "ok",
            "latency_ms": 10.0 * resolution / 32,
            "per_iteration": {"p50_ms": 10.0, "p95_ms": 14.0, "cv": 0.08},
            "throughput_samples_per_s": 100.0 / batch_size,
            "samples_ms": [9.5, 10.2, 10.1, 9.8],
            "memory": {},
            "energy": {},
        }
        for resolution, batch_size in cells
    ]

    record = BenchmarkRecord(
        run_id="sweep-run",
        model_id=model_id,
        optimization_id=optimization_id,
        status="applied",
        accuracy={"top1": 0.9, "top5": 0.99, "loss": 0.3, "num_samples": 10000},
        footprint={"weight_bytes": 44_000_000, "parameters": 11_000_000, "macs": 5.5e8},
        latency=latency,
        model_card=build_model_card(model_id, model_id, 10, 32, {"macs": 5.5e8}),
        environment={"cpu": "test cpu", "platform": "test", "fingerprint": {"torch": "2.14"}},
    )
    json_dump(record.to_dict(), root / "raw" / f"{model_id}__{optimization_id}.json")


def skipped_reason(artifacts, name: str) -> str | None:
    for entry in artifacts.skipped:
        if entry.startswith(f"{name}:"):
            return entry
    return None


def test_single_resolution_skips_the_sensitivity_figure_with_a_reason(tmp_path):
    """One resolution cannot show resolution sensitivity.

    Drawing a single bar per model under a "Resolution sensitivity" title would
    imply a comparison that was never made. The figure must be recorded as an
    explicit gap instead, which is this project's pattern everywhere else.
    """
    write_sweep_record(tmp_path, [(32, 1), (32, 8)])

    artifacts = build_report(tmp_path)
    reason = skipped_reason(artifacts, "resolution_sensitivity")

    assert reason is not None, "a single-resolution sweep must not render the figure"
    assert "at least two measured resolutions" in reason
    assert not (tmp_path / "figures" / "resolution_sensitivity.png").exists()


def test_two_resolutions_render_the_sensitivity_figure(tmp_path):
    write_sweep_record(tmp_path, [(32, 1), (224, 1)])

    artifacts = build_report(tmp_path)

    assert skipped_reason(artifacts, "resolution_sensitivity") is None
    assert (tmp_path / "figures" / "resolution_sensitivity.png").exists()


def test_single_batch_size_skips_the_batch_scaling_figure_with_a_reason(tmp_path):
    """Batch scaling is a trend across batch sizes; one point is not a trend."""
    write_sweep_record(tmp_path, [(32, 1), (224, 1)])

    artifacts = build_report(tmp_path)
    reason = skipped_reason(artifacts, "batch_scaling")

    assert reason is not None
    assert "at least two measured batch sizes" in reason


def test_two_batch_sizes_render_the_batch_scaling_figure(tmp_path):
    write_sweep_record(tmp_path, [(32, 1), (32, 8)])

    artifacts = build_report(tmp_path)

    assert skipped_reason(artifacts, "batch_scaling") is None
    assert (tmp_path / "figures" / "batch_scaling.png").exists()


def test_the_published_scope_still_produces_the_headline_figure(tmp_path):
    """32x32 at batch 1/8/32 — the suite that is actually published.

    The two guarded sweeps cover a single axis each, so this checks the guards did
    not accidentally take the headline figure down with them.
    """
    write_sweep_record(tmp_path, [(32, 1), (32, 8), (32, 32)])

    artifacts = build_report(tmp_path)

    assert skipped_reason(artifacts, "accuracy_vs_latency") is None
    assert (tmp_path / "figures" / "accuracy_vs_latency.png").exists()
    assert skipped_reason(artifacts, "batch_scaling") is None
    # Only the resolution sweep is expected to be missing.
    assert skipped_reason(artifacts, "resolution_sensitivity") is not None


def test_malformed_record_is_skipped_not_fatal(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    (raw / "broken.json").write_text("{ this is not json", encoding="utf-8")
    json_dump(
        {
            "model_id": "resnet18",
            "optimization_id": "fp32",
            "status": "applied",
            "run_id": "ok",
            "accuracy": {"top1": 0.5},
            "latency": [],
            "footprint": {},
        },
        raw / "good.json",
    )

    store = load_store(tmp_path)
    assert len(store.records) == 1, "the good record must still load"


def test_unavailable_optimization_is_recorded_with_a_reason(tmp_path):
    """The central honesty property: a skipped optimization stays visible."""
    record = BenchmarkRecord(
        run_id="test-run",
        model_id="resnet18",
        optimization_id="compile",
        status="unavailable",
        reason="no working torch.compile backend on this machine",
        optimization_params={},
    )
    json_dump(record.to_dict(), tmp_path / "raw" / "resnet18__compile.json")

    store = load_store(tmp_path)
    status = store.status_frame()

    assert len(status) == 1
    assert status.iloc[0]["status"] == "unavailable"
    assert "torch.compile" in status.iloc[0]["reason"]


def test_baseline_deltas_are_computed_per_model(tmp_path):
    """Deltas must be relative to the same model's FP32 row, not a global baseline."""
    raw = tmp_path / "raw"
    for model_id, top1, latency in (
        ("model_a", 0.90, 10.0),
        ("model_b", 0.70, 20.0),
    ):
        json_dump(
            {
                "run_id": "r",
                "model_id": model_id,
                "optimization_id": "fp32",
                "status": "applied",
                "accuracy": {"top1": top1},
                "footprint": {"weight_bytes": 1000},
                "latency": [
                    {
                        "resolution": 32,
                        "batch_size": 1,
                        "num_threads": 1,
                        "status": "ok",
                        "latency_ms": latency,
                        "per_iteration": {"cv": 0.05},
                    }
                ],
            },
            raw / f"{model_id}__fp32.json",
        )
        json_dump(
            {
                "run_id": "r",
                "model_id": model_id,
                "optimization_id": "prune_unstructured_50",
                "status": "applied",
                "accuracy": {"top1": top1 - 0.01},
                "footprint": {"weight_bytes": 1000},
                "latency": [
                    {
                        "resolution": 32,
                        "batch_size": 1,
                        "num_threads": 1,
                        "status": "ok",
                        "latency_ms": latency / 2,
                        "per_iteration": {"cv": 0.05},
                    }
                ],
            },
            raw / f"{model_id}__prune_unstructured_50.json",
        )

    frame = load_store(tmp_path).summary_frame()
    pruned = frame[frame["optimization_id"] == "prune_unstructured_50"]

    assert len(pruned) == 2
    assert (pruned["top1_delta"] < 0).all(), "accuracy must drop slightly"
    # `.tolist()` is required, not cosmetic: `pandas.Series == pytest.approx(x)`
    # returns all-False even when every value equals x, because pandas does not
    # delegate its element-wise __eq__ to the approx object.
    assert pruned["speedup_vs_fp32"].tolist() == pytest.approx([2.0, 2.0])


def test_unstable_latency_is_flagged(tmp_path):
    raw = tmp_path / "raw"
    for name, cv in (("fp32", 0.05), ("noisy", 0.40)):
        json_dump(
            {
                "run_id": "r",
                "model_id": "m",
                "optimization_id": name,
                "status": "applied",
                "accuracy": {"top1": 0.5},
                "footprint": {"weight_bytes": 100},
                "latency": [
                    {
                        "resolution": 32,
                        "batch_size": 1,
                        "num_threads": 1,
                        "status": "ok",
                        "latency_ms": 5.0,
                        "per_iteration": {"cv": cv},
                    }
                ],
            },
            raw / f"m__{name}.json",
        )

    frame = load_store(tmp_path).summary_frame()
    flags = dict(zip(frame["optimization_id"], frame["latency_unstable"], strict=True))

    assert flags["fp32"] is False
    assert flags["noisy"] is True


def test_config_summary_is_embedded_and_json_safe(tmp_path):
    config = offline_config(tmp_path)
    payload = config.summary()
    text = json.dumps(payload)
    assert "synthetic" in text


def test_synthetic_dataset_never_downloads(tmp_path):
    from edgebench.config import DataConfig
    from edgebench.data import build_datasets

    bundle = build_datasets(
        DataConfig(name="synthetic", root=str(tmp_path), download=False), seed=0
    )
    assert bundle.name == "synthetic"
    assert bundle.num_classes == 10
    assert bundle.train_size > 0
    # The synthetic loader must be constructible with zero workers offline.
    images, labels = bundle.train[0]  # type: ignore[index]
    assert images.shape == (3, 32, 32)
    assert 0 <= int(labels) < 10
