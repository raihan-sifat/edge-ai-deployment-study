"""End-to-end orchestration: train, optimize, measure, persist.

One JSON record per ``(model, optimization)`` pair is written as soon as it is
produced. This is deliberate: a full sweep is long enough that crashing at hour
three and losing everything is a real risk, and a partially complete results
directory is still a usable results directory.

Checkpoints are cached under ``checkpoints/`` and reused unless ``retrain`` is set,
so re-running the benchmark to add an optimization does not mean retraining five
networks.
"""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from edgebench.bench.energy import EnergyMeter, create_energy_meter
from edgebench.bench.runner import (
    BenchmarkRecord,
    benchmark_one,
    build_model_card,
    new_run_id,
    static_cost,
)
from edgebench.config import Config, ModelEntry
from edgebench.data import build_dataloaders, build_datasets
from edgebench.models import build_model, checkpoint_path, load_checkpoint, save_checkpoint
from edgebench.optim import OptimizationContext, apply_ladder
from edgebench.training import evaluate, fit
from edgebench.utils import ensure_dir, get_logger, json_dump, set_seed

logger = get_logger("pipeline")

#: Examples used to calibrate PTQ observers. Small on purpose: activation ranges
#: stabilise quickly, and a large calibration set makes the run slow without
#: improving the result.
CALIBRATION_SAMPLES = 512
CALIBRATION_BATCH_SIZE = 32


@dataclass
class RunSummary:
    """What a full sweep produced."""

    run_id: str
    records: list[BenchmarkRecord] = field(default_factory=list)
    training: dict[str, dict[str, Any]] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def applied(self) -> int:
        return sum(1 for record in self.records if record.status == "applied")

    @property
    def skipped(self) -> int:
        return sum(1 for record in self.records if record.status != "applied")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "models": self.models,
            "records": len(self.records),
            "applied": self.applied,
            "skipped": self.skipped,
            "elapsed_s": round(self.elapsed_s, 2),
            "training": self.training,
        }


def build_calibration_loader(bundle: Any, seed: int) -> DataLoader[Any]:
    """A deterministic calibration subset drawn from the *validation* split.

    Using validation data for calibration keeps the test split untouched, which is
    the same rule that governs model selection.
    """
    count = min(CALIBRATION_SAMPLES, len(bundle.val))  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(bundle.val), generator=generator)[:count].tolist()  # type: ignore[arg-type]
    subset = Subset(bundle.val, indices)
    return DataLoader(subset, batch_size=CALIBRATION_BATCH_SIZE, shuffle=False, drop_last=False)


def train_or_load(
    entry: ModelEntry,
    config: Config,
    model: torch.nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    device: torch.device,
    retrain: bool,
) -> dict[str, Any]:
    """Train the model, or load a cached checkpoint when one exists."""
    path = checkpoint_path(config.checkpoints_dir, entry.id, "fp32")
    seed = config.seed

    if path.exists() and not retrain:
        logger.info("%s: loading checkpoint from %s", entry.id, path)
        metadata = load_checkpoint(model, path)
        val_metrics = evaluate(model, val_loader, device, config.data.num_classes)
        test_metrics = evaluate(model, test_loader, device, config.data.num_classes)
        return {
            "source": "checkpoint",
            "checkpoint": str(path),
            "val_top1": val_metrics.top1,
            "test_top1": test_metrics.top1,
            "test_top5": test_metrics.top5,
            "test_ece": test_metrics.ece,
            "saved_at": metadata.get("saved_at"),
            "history": metadata.get("training_history"),
        }

    logger.info("%s: training for %d epoch(s)", entry.id, config.train.epochs)
    set_seed(seed, deterministic=config.train.deterministic)

    history = fit(
        model,
        train_loader,
        val_loader,
        config.train,
        device,
        config.data.num_classes,
        tag=entry.id,
    )

    val_metrics = evaluate(model, val_loader, device, config.data.num_classes)
    test_metrics = evaluate(model, test_loader, device, config.data.num_classes)

    save_checkpoint(
        model,
        path,
        model_id=entry.id,
        num_classes=config.data.num_classes,
        input_size=config.data.input_size,
        metadata={
            "training_history": history.to_dict(),
            "val_top1": val_metrics.top1,
            "test_top1": test_metrics.top1,
            "seed": seed,
        },
    )

    return {
        "source": "trained",
        "checkpoint": str(path),
        "val_top1": val_metrics.top1,
        "test_top1": test_metrics.top1,
        "test_top5": test_metrics.top5,
        "test_ece": test_metrics.ece,
        "best_val_top1": history.best_top1,
        "best_epoch": history.best_epoch,
        "history": history.to_dict(),
    }


def run_all(
    config: Config,
    model_ids: list[str] | None = None,
    optimization_ids: list[str] | None = None,
    download: bool | None = None,
    retrain: bool = False,
    measure_memory: bool = True,
    energy_meter: EnergyMeter | None = None,
) -> RunSummary:
    """Train (or load) each model, apply the ladder, and measure every rung."""
    started = time.perf_counter()
    run_id = new_run_id()

    entries = config.enabled_models(model_ids)
    optimizations = config.enabled_optimizations(optimization_ids)

    raw_dir = ensure_dir(config.raw_dir)
    ensure_dir(config.checkpoints_dir)

    logger.info(
        "run %s: %d model(s) x %d optimization(s)",
        run_id,
        len(entries),
        len(optimizations),
    )

    set_seed(config.seed, deterministic=config.train.deterministic)

    bundle = build_datasets(config.data, config.seed, download=download)
    loaders, bundle = build_dataloaders(
        config.data,
        config.seed,
        train_batch_size=config.train.batch_size,
        eval_batch_size=max(256, config.train.batch_size),
        bundle=bundle,
    )
    calib_loader = build_calibration_loader(bundle, config.seed)

    device = torch.device(config.benchmark.device)
    meter = energy_meter or create_energy_meter()
    if not meter.available:
        logger.info("energy measurement unavailable: %s", meter.reason)

    summary = RunSummary(run_id=run_id, models=[entry.id for entry in entries])

    for entry in entries:
        logger.info("=" * 72)
        logger.info("model %s (%s)", entry.id, entry.torchvision_name)
        logger.info("=" * 72)

        set_seed(config.seed, deterministic=config.train.deterministic)
        model = build_model(entry, config.data.num_classes, input_size=config.data.input_size)
        model = model.to(device)

        # Static architecture cost is a property of the FP32 graph, measured once.
        card = build_model_card(
            entry.id,
            entry.torchvision_name,
            config.data.num_classes,
            config.data.input_size,
            {
                **static_cost(model, config.data.input_size, config.data.num_classes),
                "family": entry.family,
                "adapt": entry.adapt,
                "notes": entry.notes,
            },
        )

        try:
            training_info = train_or_load(
                entry,
                config,
                model,
                loaders["train"],
                loaders["val"],
                loaders["test"],
                device,
                retrain,
            )
        except Exception as error:
            # Log the full stack: a bare message here makes a failed sweep
            # impossible to debug after the fact.
            logger.exception("%s: training failed; skipping model", entry.id)
            summary.training[entry.id] = {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            del model
            gc.collect()
            continue

        model.eval()
        summary.training[entry.id] = training_info

        context = OptimizationContext(
            model=model,
            model_id=entry.id,
            num_classes=config.data.num_classes,
            input_size=config.data.input_size,
            device=device,
            seed=config.seed,
            artifacts_dir=ensure_dir(config.raw_dir / "onnx"),
            train_cfg=config.train,
            bench_cfg=config.benchmark,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            calib_loader=calib_loader,
        )

        outcomes = apply_ladder(optimizations, context)

        for outcome in outcomes:
            record = benchmark_one(
                run_id=run_id,
                model_id=entry.id,
                outcome=outcome,
                test_loader=loaders["test"],
                device=device,
                num_classes=config.data.num_classes,
                bench_cfg=config.benchmark,
                seed=config.seed,
                model_card=card,
                energy_meter=meter,
                measure_memory=measure_memory,
            )
            summary.records.append(record)

            # Persist immediately: a long sweep that dies later still has value.
            record_path = raw_dir / f"{entry.id}__{outcome.optimization_id}.json"
            json_dump(record.to_dict(), record_path)

            if outcome.predictor is not None:
                outcome.predictor.close()
            del outcome

        del model, context, outcomes
        gc.collect()

    summary.elapsed_s = time.perf_counter() - started

    json_dump(
        {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": config.summary(),
            "summary": summary.to_dict(),
        },
        raw_dir / f"_run__{run_id}.json",
    )

    logger.info(
        "run %s complete in %.1fs: %d applied, %d skipped/unavailable",
        run_id,
        summary.elapsed_s,
        summary.applied,
        summary.skipped,
    )
    return summary


def train_only(
    config: Config,
    model_ids: list[str] | None = None,
    download: bool | None = None,
    retrain: bool = False,
) -> dict[str, Any]:
    """Train and checkpoint every selected model without benchmarking."""
    entries = config.enabled_models(model_ids)
    ensure_dir(config.checkpoints_dir)
    set_seed(config.seed, deterministic=config.train.deterministic)

    bundle = build_datasets(config.data, config.seed, download=download)
    loaders, _ = build_dataloaders(
        config.data,
        config.seed,
        train_batch_size=config.train.batch_size,
        eval_batch_size=max(256, config.train.batch_size),
        bundle=bundle,
    )
    device = torch.device(config.benchmark.device)

    results: dict[str, Any] = {}
    for entry in entries:
        set_seed(config.seed, deterministic=config.train.deterministic)
        model = build_model(entry, config.data.num_classes, input_size=config.data.input_size)
        model = model.to(device)
        results[entry.id] = train_or_load(
            entry,
            config,
            model,
            loaders["train"],
            loaders["val"],
            loaders["test"],
            device,
            retrain,
        )
        del model
        gc.collect()

    json_dump(results, Path(config.raw_dir) / "training_summary.json")
    return results
