"""Deterministic dataset construction.

The CIFAR-10 *test* split is treated as a held-out set that is touched exactly
once per configuration, after all model selection is finished. Selection uses a
deterministic carve-out of the official train split instead. This matters: it is
the difference between a benchmark and a leaderboard entry.

A ``synthetic`` dataset is also provided so the full pipeline (and CI) can run
without network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from edgebench.config import DataConfig
from edgebench.utils import get_logger

logger = get_logger("data")


@dataclass(frozen=True)
class DatasetBundle:
    """The three splits used by the study, plus their metadata."""

    train: Dataset[Any]
    val: Dataset[Any]
    test: Dataset[Any]
    num_classes: int
    input_size: int
    name: str

    @property
    def train_size(self) -> int:
        return len(self.train)  # type: ignore[arg-type]

    @property
    def val_size(self) -> int:
        return len(self.val)  # type: ignore[arg-type]

    @property
    def test_size(self) -> int:
        return len(self.test)  # type: ignore[arg-type]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
        }


def build_train_transform(cfg: DataConfig) -> transforms.Compose:
    """Augmentation used during training.

    Deliberately conservative: random crop with reflection-free zero padding plus
    a horizontal flip. Stronger recipes (CutMix, RandAugment, autoaugment) would
    raise absolute accuracy but also add variance between runs, which makes it
    harder to attribute differences to the deployment optimization under test.
    """
    base: list[Any] = []
    if cfg.augment:
        base.extend(
            [
                transforms.RandomCrop(cfg.input_size, padding=cfg.input_size // 8),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    base.extend([transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])
    return transforms.Compose(base)


def build_eval_transform(cfg: DataConfig) -> transforms.Compose:
    """Deterministic preprocessing used for validation and test."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])


def _build_cifar(cfg: DataConfig, download: bool, seed: int) -> DatasetBundle:
    try:
        train_full = datasets.CIFAR10(
            root=cfg.root,
            train=True,
            download=download,
            transform=build_train_transform(cfg),
        )
        eval_full = datasets.CIFAR10(
            root=cfg.root,
            train=True,
            download=download,
            transform=build_eval_transform(cfg),
        )
        test_set = datasets.CIFAR10(
            root=cfg.root,
            train=False,
            download=download,
            transform=build_eval_transform(cfg),
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"could not prepare CIFAR-10 under {cfg.root!r}: {error}. "
            "Run once with `--download` and an internet connection, or use "
            "`--dataset synthetic` for an offline smoke run."
        ) from error

    total = len(train_full)  # type: ignore[arg-type]
    if cfg.val_size > total:
        raise ValueError(
            f"data.val_size={cfg.val_size} exceeds the {total} available training images"
        )

    # A fixed permutation gives a stable val split across processes and machines.
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total)
    val_indices = np.sort(permutation[: cfg.val_size]).tolist()
    train_indices = np.sort(permutation[cfg.val_size :]).tolist()

    logger.info(
        "CIFAR-10 split: %d train / %d val / %d test (seed=%d)",
        len(train_indices),
        len(val_indices),
        len(test_set),  # type: ignore[arg-type]
        seed,
    )

    return DatasetBundle(
        train=Subset(train_full, train_indices),
        val=Subset(eval_full, val_indices),
        test=test_set,
        num_classes=10,
        input_size=32,
        name="cifar10",
    )


def _build_synthetic(cfg: DataConfig, seed: int) -> DatasetBundle:
    """Random tensors shaped like CIFAR-10, for offline pipeline validation.

    Accuracy on this dataset is meaningless by construction. It exists so that
    the benchmark, reporting and CI paths can be exercised without a download.
    """
    generator = torch.Generator().manual_seed(seed)
    sizes = {"train": 2048, "val": 512, "test": 512}

    def make(count: int) -> TensorDataset:
        images = torch.randn(count, 3, cfg.input_size, cfg.input_size, generator=generator)
        labels = torch.randint(0, cfg.num_classes, (count,), generator=generator)
        return TensorDataset(images, labels)

    logger.warning(
        "using the synthetic dataset: accuracy figures are NOT meaningful and must never be published"
    )
    return DatasetBundle(
        train=make(sizes["train"]),
        val=make(sizes["val"]),
        test=make(sizes["test"]),
        num_classes=cfg.num_classes,
        input_size=cfg.input_size,
        name="synthetic",
    )


def build_datasets(cfg: DataConfig, seed: int, download: bool | None = None) -> DatasetBundle:
    """Build the dataset bundle described by ``cfg``.

    ``download`` overrides ``cfg.download`` so a run can force offline mode.
    """
    effective_download = cfg.download if download is None else download
    name = cfg.name.strip().lower()

    if name == "synthetic":
        return _build_synthetic(cfg, seed)
    if name in {"cifar10", "cifar100"}:
        if name == "cifar100":
            raise NotImplementedError(
                "cifar100 is declared in the config schema but not wired up yet; "
                "the study targets cifar10"
            )
        return _build_cifar(cfg, effective_download, seed)

    raise ValueError(f"unsupported dataset {cfg.name!r}")


def build_dataloaders(
    cfg: DataConfig,
    seed: int,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    download: bool | None = None,
    bundle: DatasetBundle | None = None,
) -> tuple[dict[str, DataLoader[Any]], DatasetBundle]:
    """Build loaders for all three splits.

    Returns the loaders and the underlying bundle, because the benchmark stage
    needs the raw test set for accuracy evaluation.

    Only the ``train`` loader shuffles. Shuffling an evaluation loader would make
    the (deterministic) accuracy identical but the timing noisy.
    """
    bundle = bundle or build_datasets(cfg, seed, download=download)

    generator = torch.Generator()
    generator.manual_seed(seed)

    common: dict[str, Any] = {
        "num_workers": cfg.num_workers,
        # CPU-only study: there is no device to pin host memory for.
        "pin_memory": False,
    }
    if cfg.num_workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = 2

    loaders: dict[str, DataLoader[Any]] = {
        "train": DataLoader(
            bundle.train,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
            **common,
        ),
        "val": DataLoader(
            bundle.val,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        "test": DataLoader(
            bundle.test,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }
    return loaders, bundle
