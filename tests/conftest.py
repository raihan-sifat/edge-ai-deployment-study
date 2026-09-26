"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from edgebench.config import (
    BenchmarkConfig,
    Config,
    DataConfig,
    ModelEntry,
    OptimizationEntry,
    PathsConfig,
    TrainConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
CONFIGS_DIR = REPO_ROOT / "configs"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return CONFIGS_DIR


@pytest.fixture
def fast_bench_config() -> BenchmarkConfig:
    """A benchmark config small enough for unit tests (sub-second)."""
    return BenchmarkConfig(
        device="cpu",
        resolutions=(32,),
        batch_sizes=(1, 2),
        warmup_iters=1,
        timed_iters=3,
        repeats=2,
        thread_counts=(1,),
    )


@pytest.fixture
def tiny_config(tmp_path: Path) -> Config:
    """A complete Config pointed at a temporary directory, using synthetic data.

    Fully offline: the dataset is generated from a seeded RNG, so no download and
    no network access are required.
    """
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
            batch_size=8,
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
        optimizations=(OptimizationEntry(id="fp32"),),
        models=(ModelEntry(id="resnet18"),),
        root=tmp_path,
    )


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def _cap_threads() -> None:
    """Keep unit tests fast and their timings stable."""
    torch.set_num_threads(1)
