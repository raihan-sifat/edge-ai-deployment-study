"""Typed configuration loading with strict validation.

Configuration lives in YAML and is parsed into frozen dataclasses. Unknown keys
are rejected rather than ignored, so a typo in a config file surfaces
immediately instead of silently producing a run with default behaviour.

Precedence, lowest to highest::

    configs/*.yaml  <  --set key=value overrides  <  explicit CLI flags
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import yaml

from edgebench.utils import get_logger

logger = get_logger("config")

T = TypeVar("T")

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_MODELS_PATH = Path("configs/models.yaml")


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    name: str = "cifar10"
    root: str = "data"
    val_size: int = 5000
    num_workers: int = 2
    augment: bool = True
    download: bool = True
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

    #: Number of classes is derived from ``name`` at load time.
    num_classes: int = 10
    input_size: int = 32

    def __post_init__(self) -> None:
        if self.val_size < 0:
            raise ConfigError("data.val_size must be >= 0")
        if self.num_workers < 0:
            raise ConfigError("data.num_workers must be >= 0")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ConfigError("data.mean and data.std must each contain 3 values")
        if any(channel <= 0 for channel in self.std):
            raise ConfigError("data.std values must be positive")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5.0e-4
    nesterov: bool = True
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    label_smoothing: float = 0.1
    grad_clip: float | None = 1.0
    log_interval: int = 100
    qat_epochs: int = 3
    qat_lr: float = 1.0e-3
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError("train.epochs must be >= 1")
        if self.batch_size < 1:
            raise ConfigError("train.batch_size must be >= 1")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise ConfigError(
                f"train.optimizer must be one of sgd/adam/adamw, got {self.optimizer!r}"
            )
        if self.scheduler not in {"cosine", "step", "none"}:
            raise ConfigError(
                f"train.scheduler must be one of cosine/step/none, got {self.scheduler!r}"
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigError("train.label_smoothing must lie in [0, 1)")
        if self.warmup_epochs > self.epochs:
            raise ConfigError("train.warmup_epochs cannot exceed train.epochs")
        if self.qat_epochs < 1:
            raise ConfigError("train.qat_epochs must be >= 1")


@dataclass(frozen=True)
class BenchmarkConfig:
    device: str = "cpu"
    resolutions: tuple[int, ...] = (32, 224)
    batch_sizes: tuple[int, ...] = (1, 8, 32)
    warmup_iters: int = 10
    timed_iters: int = 50
    repeats: int = 5
    thread_counts: tuple[int | None, ...] = (1, None)
    report_affinity: bool = True

    def __post_init__(self) -> None:
        if not self.resolutions:
            raise ConfigError("benchmark.resolutions must not be empty")
        if not self.batch_sizes:
            raise ConfigError("benchmark.batch_sizes must not be empty")
        if any(res < 8 for res in self.resolutions):
            raise ConfigError("benchmark.resolutions entries must be >= 8 px")
        if any(batch < 1 for batch in self.batch_sizes):
            raise ConfigError("benchmark.batch_sizes entries must be >= 1")
        if self.warmup_iters < 0:
            raise ConfigError("benchmark.warmup_iters must be >= 0")
        if self.timed_iters < 1:
            raise ConfigError("benchmark.timed_iters must be >= 1")
        if self.repeats < 1:
            raise ConfigError("benchmark.repeats must be >= 1")
        if any(thread is not None and thread < 1 for thread in self.thread_counts):
            raise ConfigError("benchmark.thread_counts entries must be >= 1 or null")
        if not self.thread_counts:
            raise ConfigError("benchmark.thread_counts must not be empty")


@dataclass(frozen=True)
class PathsConfig:
    checkpoints: str = "checkpoints"
    results: str = "results"
    figures: str = "results/figures"
    tables: str = "results/tables"
    raw: str = "results/raw"


@dataclass(frozen=True)
class ModelEntry:
    id: str
    enabled: bool = True
    family: str = "cnn"
    torchvision_name: str = ""
    adapt: str = "cifar"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ConfigError("model entries require a non-empty id")
        if self.adapt not in {"cifar", "none"}:
            raise ConfigError(f"model {self.id!r}: adapt must be 'cifar' or 'none'")
        if not self.torchvision_name:
            object.__setattr__(self, "torchvision_name", self.id)


@dataclass(frozen=True)
class OptimizationEntry:
    id: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ConfigError("optimization entries require a non-empty id")


@dataclass(frozen=True)
class Config:
    seed: int = 1234
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    optimizations: tuple[OptimizationEntry, ...] = ()
    models: tuple[ModelEntry, ...] = ()

    #: Absolute path of the configuration file this object was loaded from.
    source: Path | None = None
    #: Root directory that relative paths are resolved against.
    root: Path = field(default_factory=Path.cwd)

    # -- convenience ------------------------------------------------------

    def enabled_models(self, only: list[str] | None = None) -> list[ModelEntry]:
        """Return enabled model entries, optionally filtered by id."""
        selected = [entry for entry in self.models if entry.enabled]
        if only:
            requested = {name.strip() for name in only if name.strip()}
            known = {entry.id for entry in self.models}
            unknown = requested - known
            if unknown:
                raise ConfigError(f"unknown model id(s): {', '.join(sorted(unknown))}")
            selected = [entry for entry in selected if entry.id in requested]
        if not selected:
            raise ConfigError("no models selected; enable at least one in configs/models.yaml")
        return selected

    def enabled_optimizations(self, only: list[str] | None = None) -> list[OptimizationEntry]:
        """Return enabled optimization entries, optionally filtered by id."""
        selected = [entry for entry in self.optimizations if entry.enabled]
        if only:
            requested = {name.strip() for name in only if name.strip()}
            known = {entry.id for entry in self.optimizations}
            unknown = requested - known
            if unknown:
                raise ConfigError(f"unknown optimization id(s): {', '.join(sorted(unknown))}")
            selected = [entry for entry in selected if entry.id in requested]
        return selected

    def path(self, key: str) -> Path:
        """Resolve a configured path relative to the project root."""
        raw = getattr(self.paths, key)
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else (self.root / candidate)

    @property
    def checkpoints_dir(self) -> Path:
        return self.path("checkpoints")

    @property
    def results_dir(self) -> Path:
        return self.path("results")

    @property
    def figures_dir(self) -> Path:
        return self.path("figures")

    @property
    def tables_dir(self) -> Path:
        return self.path("tables")

    @property
    def raw_dir(self) -> Path:
        return self.path("raw")

    def with_updates(self, **updates: Any) -> Config:
        """Return a copy with top-level fields replaced."""
        return replace(self, **updates)

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-safe description of the effective configuration."""
        return {
            "source": str(self.source) if self.source else None,
            "root": str(self.root),
            "seed": self.seed,
            "data": _as_dict(self.data),
            "train": _as_dict(self.train),
            "benchmark": _as_dict(self.benchmark),
            "paths": _as_dict(self.paths),
            "models": [_as_dict(entry) for entry in self.models],
            "optimizations": [
                {"id": entry.id, "enabled": entry.enabled, "params": entry.params}
                for entry in self.optimizations
            ],
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(
    config_path: str | Path | None = None,
    models_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    root: str | Path | None = None,
) -> Config:
    """Load, validate and merge a configuration.

    ``models_path`` defaults to a ``models.yaml`` sitting next to the main
    configuration file, which is how the shipped configs are arranged.
    """
    resolved_config = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not resolved_config.exists():
        raise ConfigError(
            f"configuration file not found: {resolved_config}. "
            f"Pass --config explicitly or run from the project root."
        )

    raw = _read_yaml(resolved_config)
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved_config} must contain a YAML mapping at the top level")

    if overrides:
        _apply_overrides(raw, overrides)

    project_root = Path(root).resolve() if root else _infer_root(resolved_config)

    resolved_models = _resolve_models_path(models_path, resolved_config, project_root)
    models_raw = _read_yaml(resolved_models)
    if not isinstance(models_raw, dict) or "models" not in models_raw:
        raise ConfigError(f"{resolved_models} must contain a top-level 'models' list")

    model_entries = _build_many(ModelEntry, models_raw["models"], context="models")

    data_cfg = _build(DataConfig, raw.get("data", {}), context="data")
    class_name, num_classes, input_size = _dataset_metadata(data_cfg.name)
    data_cfg = replace(data_cfg, num_classes=num_classes, input_size=input_size)
    logger.debug("dataset %s -> %d classes, %dpx inputs", class_name, num_classes, input_size)

    train_cfg = _build(TrainConfig, raw.get("train", {}), context="train")
    bench_cfg = _build(BenchmarkConfig, raw.get("benchmark", {}), context="benchmark")
    paths_cfg = _build(PathsConfig, raw.get("paths", {}), context="paths")

    optimizations_raw = raw.get("optimizations", [])
    if not isinstance(optimizations_raw, list):
        raise ConfigError("optimizations must be a list")
    optimization_entries = _build_many(
        OptimizationEntry, optimizations_raw, context="optimizations"
    )

    known_top_level = {"seed", "data", "train", "benchmark", "paths", "optimizations"}
    unknown = set(raw) - known_top_level
    if unknown:
        raise ConfigError(
            f"unknown top-level configuration key(s): {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(sorted(known_top_level))}"
        )

    return Config(
        seed=int(raw.get("seed", 1234)),
        data=data_cfg,
        train=train_cfg,
        benchmark=bench_cfg,
        paths=paths_cfg,
        optimizations=tuple(optimization_entries),
        models=tuple(model_entries),
        source=resolved_config.resolve(),
        root=project_root,
    )


def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ConfigError(f"could not parse {path}: {error}") from error


def _infer_root(config_path: Path) -> Path:
    """Guess the project root from the location of the config file."""
    resolved = config_path.resolve()
    if resolved.parent.name == "configs":
        return resolved.parent.parent
    return resolved.parent


def _resolve_models_path(
    explicit: str | Path | None, config_path: Path, project_root: Path
) -> Path:
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if not candidate.exists():
            raise ConfigError(f"model zoo file not found: {candidate}")
        return candidate

    siblings = [config_path.parent / "models.yaml", project_root / DEFAULT_MODELS_PATH]
    for candidate in siblings:
        if candidate.exists():
            return candidate
    raise ConfigError(
        "no models.yaml found next to the config file or under configs/; "
        "pass --models-file to point at one"
    )


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply dotted-key overrides such as ``{"train.epochs": 5}`` in place."""
    for dotted_key, value in overrides.items():
        cursor = raw
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = cursor.get(part)
            if not isinstance(node, dict):
                node = {}
                cursor[part] = node
            cursor = node
        cursor[parts[-1]] = value


def parse_set_arguments(pairs: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--set key=value`` CLI arguments into a dict.

    Values are parsed as YAML so that ``--set train.epochs=5`` yields an int and
    ``--set benchmark.resolutions=[32,64]`` yields a list.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ConfigError(f"--set expects key=value, got {pair!r}")
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"--set expects a non-empty key, got {pair!r}")
        try:
            overrides[key] = yaml.safe_load(raw_value)
        except yaml.YAMLError as error:
            raise ConfigError(f"could not parse value for --set {key}: {error}") from error
    return overrides


def _dataset_metadata(name: str) -> tuple[str, int, int]:
    """Return ``(canonical_name, num_classes, input_size)`` for a dataset id."""
    registry = {
        "cifar10": ("cifar10", 10, 32),
        "cifar100": ("cifar100", 100, 32),
        "synthetic": ("synthetic", 10, 32),
    }
    key = name.strip().lower()
    if key not in registry:
        raise ConfigError(f"unknown dataset {name!r}; supported: {', '.join(sorted(registry))}")
    return registry[key]


def _build(cls: type[T], payload: Any, *, context: str) -> T:
    """Instantiate a dataclass from a mapping, rejecting unknown keys."""
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration section {context!r} must be a mapping")

    allowed = {f.name for f in fields(cls) if f.init}
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {context}: {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    kwargs: dict[str, Any] = {}
    for spec in fields(cls):
        if not spec.init or spec.name not in payload:
            continue
        value = payload[spec.name]
        # Tuples in the schema are frequently written as YAML lists.
        if spec.default is not None and isinstance(spec.default, tuple) and isinstance(value, list):
            value = tuple(value)
        if spec.name in {"mean", "std"} and isinstance(value, list):
            value = tuple(value)
        kwargs[spec.name] = value
    return cls(**kwargs)


def _build_many(cls: type[T], payload: Any, *, context: str) -> list[T]:
    if not isinstance(payload, list):
        raise ConfigError(f"configuration section {context!r} must be a list")
    return [_build(cls, item, context=f"{context}[{index}]") for index, item in enumerate(payload)]


def _as_dict(instance: Any) -> dict[str, Any]:
    """Recursively convert a (possibly nested) dataclass to a plain dict."""
    if not is_dataclass(instance) or isinstance(instance, type):
        return instance
    result: dict[str, Any] = {}
    for spec in fields(instance):
        value = getattr(instance, spec.name)
        if is_dataclass(value) and not isinstance(value, type):
            result[spec.name] = _as_dict(value)
        elif isinstance(value, tuple):
            result[spec.name] = list(value)
        else:
            result[spec.name] = value
    return result
