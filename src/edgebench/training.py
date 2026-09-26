"""Training and evaluation primitives.

Deliberately framework-free (plain PyTorch loops) so that the training recipe is
fully visible in the report. The same entry points are reused by quantization-aware
training, which is why they live outside the CLI.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edgebench.config import TrainConfig
from edgebench.utils import get_logger

logger = get_logger("training")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class AccuracyMetrics:
    """Top-k accuracy, loss and calibration for one pass over a dataset."""

    top1: float
    top5: float
    loss: float
    num_samples: int
    ece: float | None = None
    per_class_accuracy: list[float] | None = None
    confusion: list[list[int]] | None = None

    def to_dict(self, include_matrices: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "top1": self.top1,
            "top5": self.top5,
            "loss": self.loss,
            "num_samples": self.num_samples,
            "ece": self.ece,
        }
        if include_matrices:
            payload["per_class_accuracy"] = self.per_class_accuracy
            payload["confusion"] = self.confusion
        return payload


def expected_calibration_error(
    confidences: torch.Tensor, correct: torch.Tensor, num_bins: int = 15
) -> float:
    """Top-label expected calibration error (ECE).

    Takes the *top-label* confidence and whether that label was correct, both
    reduced to 1-D tensors of length N. The estimator is the standard one:

    .. math::

        \\mathrm{ECE} = \\sum_{b=1}^{B} \\frac{|S_b|}{N}
                        \\bigl| \\mathrm{acc}(S_b) - \\mathrm{conf}(S_b) \\bigr|

    Edge deployments frequently gate on a confidence threshold, so a quantized
    model can be far less accurate *after thresholding* than its top-1 score
    suggests even when top-1 barely moves. ECE makes that failure mode visible.
    """
    if confidences.numel() != correct.numel():
        raise ValueError("confidences and correct must have the same length")
    if confidences.numel() == 0:
        return float("nan")

    confidences = confidences.double().flatten()
    correct = correct.double().flatten()

    # Bins are half-open (lower, upper] except the first, which includes 0.0, so
    # that a confidence of exactly 1.0 lands in the final bin.
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    error = torch.zeros((), dtype=torch.float64)

    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = confidences.gt(lower) & confidences.le(upper)
        if index == 0:
            in_bin = in_bin | confidences.eq(0.0)

        fraction = in_bin.double().mean()
        if fraction.item() == 0.0:
            continue
        bin_accuracy = correct[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        error += fraction * (bin_accuracy - bin_confidence).abs()

    return float(error.item())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate(
    model: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader[Any],
    device: torch.device,
    num_classes: int,
    with_details: bool = False,
    criterion: nn.Module | None = None,
) -> AccuracyMetrics:
    """Evaluate ``model`` over ``loader``.

    ``model`` is any callable mapping a batch to logits, which is what lets the
    same evaluation run for eager PyTorch, compiled, quantized and ONNX Runtime
    backends without a parallel implementation per backend.

    ``with_details`` additionally computes per-class accuracy and a confusion
    matrix, which the report uses to show *where* quantization hurts rather than
    only how much.
    """
    # Only nn.Module instances carry a mode to save and restore; the Predictor
    # wrappers manage their own evaluation state.
    previous_mode: bool | None = None
    if isinstance(model, nn.Module):
        previous_mode = model.training
        model.eval()

    criterion = criterion or nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64) if with_details else None
    all_confidences: list[torch.Tensor] = []
    all_correct: list[torch.Tensor] = []

    try:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=False)
            targets = targets.to(device, non_blocking=False)

            logits = model(inputs)
            if isinstance(logits, tuple):  # some exported graphs return (logits, aux)
                logits = logits[0]

            loss = criterion(logits, targets)
            batch = targets.size(0)
            total_loss += float(loss.item()) * batch
            total_samples += batch

            probabilities = logits.softmax(dim=1)
            confidences, predictions = probabilities.max(dim=1)
            correct = predictions.eq(targets)

            top1_correct += int(correct.sum().item())
            k = min(5, num_classes)
            topk_predictions = logits.topk(k, dim=1).indices
            top5_correct += int(topk_predictions.eq(targets.view(-1, 1)).any(dim=1).sum().item())

            all_confidences.append(confidences.detach().cpu())
            all_correct.append(correct.detach().cpu())

            if confusion is not None:
                flat_index = targets * num_classes + predictions
                confusion += torch.bincount(
                    flat_index, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)
    finally:
        if previous_mode is not None:
            model.train(previous_mode)

    if total_samples == 0:
        raise ValueError("evaluation loader produced no samples")

    per_class: list[float] | None = None
    if confusion is not None:
        per_class = []
        for class_index in range(num_classes):
            support = int(confusion[class_index].sum().item())
            correct_count = int(confusion[class_index, class_index].item())
            per_class.append(correct_count / support if support else float("nan"))

    ece = expected_calibration_error(torch.cat(all_confidences), torch.cat(all_correct))

    return AccuracyMetrics(
        top1=top1_correct / total_samples,
        top5=top5_correct / total_samples,
        loss=total_loss / total_samples,
        num_samples=total_samples,
        ece=ece,
        per_class_accuracy=per_class,
        confusion=confusion.tolist() if confusion is not None else None,
    )


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Construct the optimizer, excluding biases and norms from weight decay.

    Applying weight decay to normalization parameters and biases measurably hurts
    small models on small datasets, and it is the single most common deviation
    between published CIFAR recipes.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            groups,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    raise ValueError(f"unsupported optimizer {cfg.optimizer!r}")


class LearningRateSchedule(ABC):
    """Per-step learning-rate schedule.

    All schedules in this module advance once per *optimizer step* and return the
    learning rate they applied. That uniformity matters: ``torch.optim.lr_scheduler``
    classes return ``None`` from ``step()``, and ``StepLR`` is meant to advance once
    per epoch, so mixing the two conventions silently produces a schedule that
    decays at the wrong rate. Wrapping keeps the loop unaware of which is in use.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_index = 0

    @abstractmethod
    def lr_at(self, step: int) -> float:
        """Learning rate for the given global step index."""

    def step(self) -> float:
        """Apply the learning rate for the current step and advance."""
        learning_rate = self.lr_at(self.step_index)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.step_index += 1
        return learning_rate

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"step_index": self.step_index, "base_lr": self.base_lr}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_index = int(state.get("step_index", 0))
        self.base_lr = float(state.get("base_lr", self.base_lr))


class ConstantSchedule(LearningRateSchedule):
    """No decay. Used for smoke runs where a schedule would be noise."""

    def lr_at(self, step: int) -> float:
        return self.base_lr


class WarmupCosineSchedule(LearningRateSchedule):
    """Linear warmup followed by cosine decay to ``min_lr``.

    Written by hand rather than using ``LambdaLR`` so the effective learning rate
    can be recorded per step and plotted in the report.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        base_lr: float = 0.1,
        min_lr: float | None = None,
    ) -> None:
        super().__init__(optimizer, base_lr)
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(0, warmup_steps)
        self.min_lr = min_lr if min_lr is not None else base_lr * 0.01

    def lr_at(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.total_steps = int(state.get("total_steps", self.total_steps))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.min_lr = float(state.get("min_lr", self.min_lr))


class StepDecaySchedule(LearningRateSchedule):
    """Multiply the learning rate by ``gamma`` every ``step_every_epochs`` epochs.

    Implemented per-step (rather than delegating to ``StepLR``) so it obeys the
    same contract as the other schedules in this module.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int,
        base_lr: float,
        step_every_epochs: int = 10,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(optimizer, base_lr)
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.step_every_epochs = max(1, step_every_epochs)
        self.gamma = gamma

    def lr_at(self, step: int) -> float:
        epoch = step // self.steps_per_epoch
        return self.base_lr * (self.gamma ** (epoch // self.step_every_epochs))

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "steps_per_epoch": self.steps_per_epoch,
            "step_every_epochs": self.step_every_epochs,
            "gamma": self.gamma,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.steps_per_epoch = int(state.get("steps_per_epoch", self.steps_per_epoch))
        self.step_every_epochs = int(state.get("step_every_epochs", self.step_every_epochs))
        self.gamma = float(state.get("gamma", self.gamma))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
    epochs: int | None = None,
) -> LearningRateSchedule:
    """Build the LR schedule named in the config.

    ``epochs`` defaults to ``cfg.epochs`` but must be passed explicitly by QAT and
    any other shortened run, otherwise a cosine decay is compressed into the wrong
    number of steps and never reaches its minimum.
    """
    total_epochs = epochs or cfg.epochs
    total_steps = max(1, steps_per_epoch * total_epochs)

    if cfg.scheduler == "cosine":
        # Warmup is capped at the run length so a 1-epoch smoke run still works.
        warmup_epochs = min(cfg.warmup_epochs, total_epochs)
        return WarmupCosineSchedule(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_epochs * steps_per_epoch,
            base_lr=cfg.lr,
        )
    if cfg.scheduler == "step":
        return StepDecaySchedule(
            optimizer,
            steps_per_epoch=steps_per_epoch,
            base_lr=cfg.lr,
        )
    if cfg.scheduler == "none":
        return ConstantSchedule(optimizer, base_lr=cfg.lr)
    raise ValueError(f"unsupported scheduler {cfg.scheduler!r}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Per-epoch trace of a training run."""

    epochs: list[dict[str, float]] = field(default_factory=list)
    best_top1: float = 0.0
    best_epoch: int = -1
    stopped_early: bool = False

    def record(self, entry: dict[str, float]) -> None:
        self.epochs.append(entry)
        if entry["val_top1"] > self.best_top1:
            self.best_top1 = entry["val_top1"]
            self.best_epoch = int(entry["epoch"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "best_top1": self.best_top1,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
        }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: LearningRateSchedule | None = None,
    grad_clip: float | None = None,
    log_interval: int = 100,
    epoch: int = 0,
) -> dict[str, float]:
    """One pass over the training set. Returns mean loss and top-1 accuracy."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_samples = 0
    learning_rates: list[float] = []

    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=False)
        targets = targets.to(device, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, targets)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            # Every schedule in this module returns the learning rate it applied.
            learning_rates.append(scheduler.step())

        batch = targets.size(0)
        running_loss += float(loss.item()) * batch
        running_correct += int(logits.detach().argmax(dim=1).eq(targets).sum().item())
        running_samples += batch

        if log_interval and step % log_interval == 0 and step > 0:
            logger.info(
                "epoch %d step %d/%d loss=%.4f acc=%.2f%%",
                epoch + 1,
                step,
                len(loader),
                running_loss / running_samples,
                100.0 * running_correct / running_samples,
            )

    mean_lr = sum(learning_rates) / len(learning_rates) if learning_rates else 0.0
    return {
        "loss": running_loss / max(1, running_samples),
        "top1": running_correct / max(1, running_samples),
        "lr": mean_lr,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    cfg: TrainConfig,
    device: torch.device,
    num_classes: int,
    epochs: int | None = None,
    learning_rate: float | None = None,
    tag: str = "train",
) -> TrainingHistory:
    """Full training run with best-on-validation checkpoint selection.

    The test set is never consulted here. Whatever selection pressure exists in
    this study is applied through the validation split only.
    """
    epochs = epochs or cfg.epochs
    effective = cfg
    if learning_rate is not None and learning_rate != cfg.lr:
        effective = TrainConfig(**{**cfg.__dict__, "lr": learning_rate, "epochs": epochs})

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, effective)
    scheduler = build_scheduler(
        optimizer, effective, steps_per_epoch=max(1, len(train_loader)), epochs=epochs
    )

    history = TrainingHistory()
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    started = time.perf_counter()

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
            grad_clip=cfg.grad_clip,
            log_interval=cfg.log_interval,
            epoch=epoch,
        )
        val_metrics = evaluate(model, val_loader, device, num_classes)

        history.record(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "val_loss": val_metrics.loss,
                "val_top1": val_metrics.top1,
                "val_top5": val_metrics.top5,
                "lr": train_metrics["lr"],
            }
        )

        if val_metrics.top1 >= history.best_top1:
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        logger.info(
            "[%s] epoch %d/%d train_loss=%.4f train_acc=%.2f%% val_acc=%.2f%% val_ece=%.4f",
            tag,
            epoch + 1,
            epochs,
            train_metrics["loss"],
            100 * train_metrics["top1"],
            100 * val_metrics.top1,
            val_metrics.ece or float("nan"),
        )

    model.load_state_dict(best_state)
    logger.info(
        "[%s] finished in %.1fs; best val top-1 %.2f%% at epoch %d",
        tag,
        time.perf_counter() - started,
        100 * history.best_top1,
        history.best_epoch,
    )
    return history
