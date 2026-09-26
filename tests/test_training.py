"""Training loop, metrics and calibration."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edgebench.config import TrainConfig
from edgebench.training import (
    AccuracyMetrics,
    ConstantSchedule,
    StepDecaySchedule,
    TrainingHistory,
    WarmupCosineSchedule,
    build_optimizer,
    build_scheduler,
    evaluate,
    expected_calibration_error,
    fit,
    train_one_epoch,
)


class ToyNet(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_loader(num_samples: int = 32, num_classes: int = 4, batch_size: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, 8, generator=generator)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)


class TestCalibrationError:
    def test_perfectly_calibrated_confidence_is_near_zero(self):
        """Confidence 1.0 with 100% accuracy in that bin -> zero error."""
        confidences = torch.ones(100)
        correct = torch.ones(100, dtype=torch.bool)
        assert expected_calibration_error(confidences, correct) == pytest.approx(0.0, abs=1e-6)

    def test_fully_overconfident_scores_high(self):
        """Always 100% confident but always wrong -> maximum error."""
        confidences = torch.ones(100)
        correct = torch.zeros(100, dtype=torch.bool)
        assert expected_calibration_error(confidences, correct) == pytest.approx(1.0, abs=1e-6)

    def test_confidence_equal_to_accuracy_is_calibrated(self):
        confidences = torch.full((100,), 0.7)
        correct = torch.cat([torch.ones(70, dtype=torch.bool), torch.zeros(30, dtype=torch.bool)])
        assert expected_calibration_error(confidences, correct) == pytest.approx(0.0, abs=1e-6)

    def test_bounded_between_zero_and_one(self):
        generator = torch.Generator().manual_seed(0)
        confidences = torch.rand(500, generator=generator)
        correct = torch.rand(500, generator=generator) > 0.5
        value = expected_calibration_error(confidences, correct)
        assert 0.0 <= value <= 1.0

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            expected_calibration_error(torch.ones(3), torch.ones(4, dtype=torch.bool))

    def test_empty_input_is_nan(self):
        assert math.isnan(
            expected_calibration_error(torch.zeros(0), torch.zeros(0, dtype=torch.bool))
        )

    def test_confidence_of_one_lands_in_the_final_bin(self):
        """Boundary handling: 1.0 must be counted, not dropped."""
        confidences = torch.ones(10)
        correct = torch.zeros(10, dtype=torch.bool)
        # If the top bin were excluded the result would be 0.0 instead of 1.0.
        assert expected_calibration_error(confidences, correct, num_bins=5) == pytest.approx(
            1.0, abs=1e-6
        )


class TestEvaluation:
    def test_reports_accuracy_and_loss(self):
        torch.manual_seed(0)
        model = ToyNet()
        loader = make_loader()

        metrics = evaluate(model, loader, torch.device("cpu"), num_classes=4)

        assert isinstance(metrics, AccuracyMetrics)
        assert 0.0 <= metrics.top1 <= 1.0
        assert 0.0 <= metrics.top5 <= 1.0
        assert metrics.loss > 0.0
        assert metrics.num_samples == 32
        assert metrics.ece is not None

    def test_details_adds_per_class_and_confusion(self):
        torch.manual_seed(0)
        model = ToyNet()
        loader = make_loader(num_samples=64)

        metrics = evaluate(model, loader, torch.device("cpu"), num_classes=4, with_details=True)

        assert metrics.per_class_accuracy is not None
        assert len(metrics.per_class_accuracy) == 4
        assert metrics.confusion is not None
        assert len(metrics.confusion) == 4

        # The confusion matrix must account for every sample exactly once.
        total = sum(sum(row) for row in metrics.confusion)
        assert total == 64

    def test_perfect_classifier_scores_one(self):
        """A model that always predicts the truth must reach 100%."""
        labels = torch.arange(8) % 4
        features = torch.nn.functional.one_hot(labels, 4).float() @ torch.eye(4)[:, :8] * 10
        loader = DataLoader(TensorDataset(features, labels), batch_size=8)

        class Oracle(nn.Module):
            def forward(self, x):
                return x[:, :4]

        metrics = evaluate(Oracle(), loader, torch.device("cpu"), num_classes=4)
        assert metrics.top1 == pytest.approx(1.0)

    def test_empty_loader_rejected(self):
        empty = DataLoader(TensorDataset(torch.zeros(0, 8), torch.zeros(0, dtype=torch.long)))
        with pytest.raises(ValueError, match="no samples"):
            evaluate(ToyNet(), empty, torch.device("cpu"), num_classes=4)

    def test_works_with_a_plain_callable(self):
        """Backend-agnostic evaluation: any callable returning logits must work."""

        class Wrapper:
            def __init__(self, module: nn.Module) -> None:
                self.module = module

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return self.module(x)

        torch.manual_seed(0)
        metrics = evaluate(Wrapper(ToyNet()), make_loader(), torch.device("cpu"), num_classes=4)
        assert metrics.num_samples == 32

    def test_evaluation_restores_training_mode(self):
        model = ToyNet()
        model.train()
        evaluate(model, make_loader(), torch.device("cpu"), num_classes=4)
        assert model.training is True, "evaluate must restore the previous mode"

    def test_tuple_output_is_unwrapped(self):
        class TupleNet(nn.Module):
            def forward(self, x):
                return self.linear(x), "aux"

        TupleNet.linear = nn.Linear(8, 4)
        metrics = evaluate(TupleNet(), make_loader(), torch.device("cpu"), num_classes=4)
        assert metrics.num_samples == 32


class TestOptimizer:
    def test_biases_and_norms_excluded_from_weight_decay(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Flatten(), nn.Linear(4, 2))
        optimizer = build_optimizer(model, TrainConfig(optimizer="sgd", weight_decay=0.01))

        decays = {group["weight_decay"] for group in optimizer.param_groups}
        assert decays == {0.01, 0.0}

        decayed = optimizer.param_groups[0]["params"]
        undecayed = optimizer.param_groups[1]["params"]

        assert all(p.ndim > 1 for p in decayed), "only multi-dim weights may be decayed"
        assert any(p.ndim <= 1 for p in undecayed)

    @pytest.mark.parametrize("name", ["sgd", "adam", "adamw"])
    def test_supported_optimizers(self, name):
        optimizer = build_optimizer(nn.Linear(4, 2), TrainConfig(optimizer=name))
        assert isinstance(optimizer, torch.optim.SGD | torch.optim.Adam | torch.optim.AdamW)

    def test_frozen_parameters_excluded(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        for parameter in model[0].parameters():
            parameter.requires_grad = False

        optimizer = build_optimizer(model, TrainConfig())
        selected = [p for group in optimizer.param_groups for p in group["params"]]
        assert all(p.requires_grad for p in selected)
        assert len(selected) == 2  # one weight + one bias from the second layer


class TestSchedules:
    def test_warmup_increases_then_cosine_decays(self):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        schedule = WarmupCosineSchedule(optimizer, total_steps=100, warmup_steps=10, base_lr=0.1)

        first = schedule.lr_at(0)
        peak = schedule.lr_at(10)
        last = schedule.lr_at(99)

        assert first < peak, "warmup must start low"
        assert peak == pytest.approx(0.1)
        assert last < peak, "cosine must decay"
        assert last >= schedule.min_lr

    def test_schedule_reaches_minimum_at_the_end(self):
        optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
        schedule = WarmupCosineSchedule(optimizer, total_steps=50, warmup_steps=0, base_lr=0.1)
        assert schedule.lr_at(50) == pytest.approx(schedule.min_lr, abs=1e-6)

    def test_step_returns_the_rate_it_applied(self):
        """The contract that makes scheduling in the training loop safe."""
        optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
        schedule = ConstantSchedule(optimizer, base_lr=0.05)

        returned = schedule.step()

        assert returned is not None
        assert returned == pytest.approx(0.05)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)

    def test_all_schedules_obey_the_contract(self):
        """Guards against the StepLR.step() -> None trap."""
        for schedule in (
            ConstantSchedule(torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1), 0.1),
            WarmupCosineSchedule(
                torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1), total_steps=10
            ),
            StepDecaySchedule(
                torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1),
                steps_per_epoch=2,
                base_lr=0.1,
            ),
        ):
            value = schedule.step()
            assert isinstance(value, float), f"{type(schedule).__name__}.step() returned {value!r}"

    def test_step_decay_drops_at_the_right_epoch(self):
        optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
        schedule = StepDecaySchedule(
            optimizer, steps_per_epoch=5, base_lr=0.1, step_every_epochs=2, gamma=0.1
        )

        assert schedule.lr_at(0) == pytest.approx(0.1)  # epoch 0
        assert schedule.lr_at(9) == pytest.approx(0.1)  # epoch 1
        assert schedule.lr_at(10) == pytest.approx(0.01)  # epoch 2
        assert schedule.lr_at(20) == pytest.approx(0.001)  # epoch 4

    def test_build_scheduler_respects_epoch_override(self):
        """QAT runs fewer epochs; the cosine must not be stretched over the wrong span."""
        optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
        config = TrainConfig(scheduler="cosine", epochs=30, lr=0.1)

        short = build_scheduler(optimizer, config, steps_per_epoch=10, epochs=3)
        long = build_scheduler(optimizer, config, steps_per_epoch=10, epochs=30)

        assert isinstance(short, WarmupCosineSchedule)
        assert isinstance(long, WarmupCosineSchedule)
        assert short.total_steps == 30
        assert long.total_steps == 300

    def test_scheduler_state_round_trip(self):
        optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
        schedule = WarmupCosineSchedule(optimizer, total_steps=20, base_lr=0.1)
        for _ in range(5):
            schedule.step()

        state = schedule.state_dict()
        fresh = WarmupCosineSchedule(optimizer, total_steps=20, base_lr=0.1)
        fresh.load_state_dict(state)

        assert fresh.step_index == 5


class TestTrainingLoop:
    def test_one_epoch_reduces_loss_on_a_learnable_task(self):
        """Sanity: the loop actually optimizes."""
        torch.manual_seed(0)
        model = nn.Linear(8, 2)
        features = torch.randn(256, 8)
        labels = (features[:, 0] > 0).long()
        loader = DataLoader(TensorDataset(features, labels), batch_size=32)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        first = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        for _ in range(5):
            last = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

        assert last["loss"] < first["loss"]
        assert last["top1"] > 0.8

    def test_fit_returns_history_and_selects_best(self):
        torch.manual_seed(0)
        model = nn.Linear(8, 2)
        features = torch.randn(128, 8)
        labels = (features[:, 0] > 0).long()
        loader = DataLoader(TensorDataset(features, labels), batch_size=32)

        config = TrainConfig(
            epochs=3,
            lr=0.1,
            scheduler="cosine",
            warmup_epochs=1,
            label_smoothing=0.0,
            log_interval=0,
        )
        history = fit(model, loader, loader, config, torch.device("cpu"), num_classes=2)

        assert len(history.epochs) == 3
        assert history.best_epoch >= 0
        assert 0.0 <= history.best_top1 <= 1.0
        # The recorded trace must be complete enough to plot.
        for key in ("epoch", "train_loss", "train_top1", "val_top1", "lr"):
            assert key in history.epochs[0]

    def test_fit_honours_epoch_override(self):
        torch.manual_seed(0)
        model = nn.Linear(8, 2)
        loader = make_loader(num_samples=32, num_classes=2)
        config = TrainConfig(epochs=10, scheduler="none", log_interval=0)

        history = fit(model, loader, loader, config, torch.device("cpu"), num_classes=2, epochs=2)
        assert len(history.epochs) == 2

    def test_fit_applies_learning_rate_override(self):
        torch.manual_seed(0)
        model = nn.Linear(8, 2)
        loader = make_loader(num_samples=32, num_classes=2)
        config = TrainConfig(epochs=1, lr=0.1, scheduler="none", log_interval=0)

        history = fit(
            model,
            loader,
            loader,
            config,
            torch.device("cpu"),
            num_classes=2,
            learning_rate=0.001,
        )
        assert history.epochs[0]["lr"] == pytest.approx(0.001, rel=1e-3)

    def test_gradient_clipping_prevents_explosion(self):
        torch.manual_seed(0)
        model = nn.Linear(8, 2)
        features = torch.randn(64, 8) * 1e4  # pathological inputs
        labels = torch.randint(0, 2, (64,))
        loader = DataLoader(TensorDataset(features, labels), batch_size=16)

        config = TrainConfig(grad_clip=1.0, log_interval=0)
        metrics = train_one_epoch(
            model,
            loader,
            torch.optim.SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            grad_clip=config.grad_clip,
        )
        assert math.isfinite(metrics["loss"])

    def test_best_state_is_restored(self):
        """fit() must leave the model at its best-validation weights, not the last."""
        torch.manual_seed(0)
        model = nn.Linear(8, 2)
        loader = make_loader(num_samples=64, num_classes=2)

        config = TrainConfig(epochs=4, lr=0.5, scheduler="none", log_interval=0)
        history = fit(model, loader, loader, config, torch.device("cpu"), num_classes=2)

        assert history.best_epoch >= 0
        # Re-evaluating the returned model should reproduce the recorded best score,
        # within the noise of a fixed (non-shuffled) loader it should match exactly.
        restored = evaluate(model, loader, torch.device("cpu"), num_classes=2)
        assert restored.top1 == pytest.approx(history.best_top1, abs=1e-6)


class TestHistory:
    def test_tracks_best(self):
        history = TrainingHistory()
        history.record({"epoch": 1, "val_top1": 0.5})
        history.record({"epoch": 2, "val_top1": 0.7})
        history.record({"epoch": 3, "val_top1": 0.6})

        assert history.best_top1 == 0.7
        assert history.best_epoch == 2

    def test_serialises(self):
        import json

        history = TrainingHistory()
        history.record({"epoch": 1, "val_top1": 0.5})
        json.dumps(history.to_dict())
