"""Configuration loading and validation."""

from __future__ import annotations

import pytest
import yaml

from edgebench.config import (
    BenchmarkConfig,
    ConfigError,
    DataConfig,
    ModelEntry,
    TrainConfig,
    load_config,
    parse_set_arguments,
)


def test_loads_shipped_default_config(configs_dir):
    config = load_config(configs_dir / "default.yaml")

    assert config.seed == 1234
    assert config.data.name == "cifar10"
    assert config.data.num_classes == 10
    assert config.data.input_size == 32
    assert config.train.epochs == 30
    assert 224 in config.benchmark.resolutions
    assert config.models, "the default model zoo must not be empty"
    assert {entry.id for entry in config.models} >= {
        "resnet18",
        "mobilenet_v2",
        "mobilenet_v3_small",
        "shufflenet_v2",
        "efficientnet_b0",
    }
    assert config.paths.results == "results"


def test_loads_smoke_config(configs_dir):
    config = load_config(configs_dir / "smoke.yaml")
    assert config.train.epochs == 1
    assert config.benchmark.resolutions == (32,)
    assert len(config.enabled_optimizations()) < len(config.optimizations)


def test_path_resolution_is_relative_to_root(configs_dir, repo_root):
    config = load_config(configs_dir / "default.yaml")
    assert config.results_dir == repo_root / "results"
    assert config.checkpoints_dir == repo_root / "checkpoints"


def test_missing_config_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_unknown_top_level_key_rejected(tmp_path):
    (tmp_path / "models.yaml").write_text("models:\n  - id: resnet18\n", encoding="utf-8")
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("seed: 1\nnot_a_real_key: 5\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="not_a_real_key"):
        load_config(config_path)


def test_unknown_section_key_rejected(tmp_path):
    (tmp_path / "models.yaml").write_text("models:\n  - id: resnet18\n", encoding="utf-8")
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("train:\n  epochz: 5\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="epochz"):
        load_config(config_path)


class TestValidation:
    def test_negative_val_size_rejected(self):
        with pytest.raises(ConfigError, match="val_size"):
            DataConfig(val_size=-1)

    def test_zero_std_rejected(self):
        with pytest.raises(ConfigError, match="std"):
            DataConfig(std=(0.0, 1.0, 1.0))

    def test_unknown_optimizer_rejected(self):
        with pytest.raises(ConfigError, match="optimizer"):
            TrainConfig(optimizer="rmsprop")

    def test_unknown_scheduler_rejected(self):
        with pytest.raises(ConfigError, match="scheduler"):
            TrainConfig(scheduler="triangular")

    def test_warmup_longer_than_run_rejected(self):
        with pytest.raises(ConfigError, match="warmup_epochs"):
            TrainConfig(epochs=2, warmup_epochs=5)

    def test_empty_resolutions_rejected(self):
        with pytest.raises(ConfigError, match="resolutions"):
            BenchmarkConfig(resolutions=())

    def test_zero_repeats_rejected(self):
        with pytest.raises(ConfigError, match="repeats"):
            BenchmarkConfig(repeats=0)

    def test_zero_threads_rejected(self):
        with pytest.raises(ConfigError, match="thread_counts"):
            BenchmarkConfig(thread_counts=(0,))

    def test_none_thread_means_automatic(self):
        assert BenchmarkConfig(thread_counts=(None,)).thread_counts == (None,)


class TestModelFiltering:
    def test_enabled_filter(self, tmp_path, configs_dir):
        config = load_config(configs_dir / "default.yaml")
        selected = config.enabled_models(["resnet18"])
        assert [entry.id for entry in selected] == ["resnet18"]

    def test_unknown_model_id_rejected(self, configs_dir):
        config = load_config(configs_dir / "default.yaml")
        with pytest.raises(ConfigError, match="unknown model id"):
            config.enabled_models(["no_such_model"])

    def test_empty_selection_rejected(self, configs_dir):
        config = load_config(configs_dir / "default.yaml")
        disabled = tuple(ModelEntry(id=entry.id, enabled=False) for entry in config.models)
        with pytest.raises(ConfigError, match="no models selected"):
            config.with_updates(models=disabled).enabled_models()


class TestOverrides:
    def test_dotted_override_applies(self, configs_dir):
        config = load_config(configs_dir / "default.yaml", overrides={"train.epochs": 3})
        assert config.train.epochs == 3

    def test_override_creates_missing_section(self, tmp_path):
        (tmp_path / "models.yaml").write_text("models:\n  - id: resnet18\n", encoding="utf-8")
        config_path = tmp_path / "min.yaml"
        config_path.write_text("seed: 1\n", encoding="utf-8")

        config = load_config(config_path, overrides={"benchmark.timed_iters": 7})
        assert config.benchmark.timed_iters == 7

    def test_parse_set_arguments_types(self):
        parsed = parse_set_arguments(["train.epochs=5", "seed=42", "benchmark.resolutions=[32,64]"])
        assert parsed["train.epochs"] == 5
        assert parsed["seed"] == 42
        assert parsed["benchmark.resolutions"] == [32, 64]

    def test_parse_set_rejects_missing_equals(self):
        with pytest.raises(ConfigError, match="key=value"):
            parse_set_arguments(["train.epochs"])

    def test_parse_set_rejects_empty_key(self):
        with pytest.raises(ConfigError, match="non-empty key"):
            parse_set_arguments(["=5"])


def test_summary_is_json_serialisable(configs_dir):
    import json

    config = load_config(configs_dir / "default.yaml")
    json.dumps(config.summary())


def test_tuples_survive_yaml_round_trip(tmp_path):
    """YAML has no tuple type; a list in the file must become a tuple in the schema."""
    (tmp_path / "models.yaml").write_text("models:\n  - id: resnet18\n", encoding="utf-8")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {"resolutions": [32, 64], "batch_sizes": [1]},
                "data": {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.benchmark.resolutions == (32, 64)
    assert config.data.mean == (0.1, 0.2, 0.3)


def test_unknown_optimization_id_rejected(configs_dir):
    config = load_config(configs_dir / "default.yaml")
    with pytest.raises(ConfigError, match="unknown optimization id"):
        config.enabled_optimizations(["not_an_optimization"])
