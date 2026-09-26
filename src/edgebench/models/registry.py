"""Model zoo: construct, adapt and checkpoint the networks under study."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision

from edgebench.config import ModelEntry
from edgebench.models.adapt import adapt_stem_to_cifar, replace_classifier_head, verify_head
from edgebench.utils import get_logger

logger = get_logger("models")

CHECKPOINT_FORMAT_VERSION = 2

#: Documented families; kept small on purpose so the study stays comparable.
KNOWN_FAMILIES = {"cnn"}


def _torchvision_factory(name: str) -> Callable[[], nn.Module]:
    """Resolve a torchvision model constructor by name.

    The lookup is resolved *eagerly* so that an unknown name raises ``KeyError``
    at construction time. Deferring it into a closure would surface as whatever
    exception torchvision happens to raise on call -- typically a ``ValueError``
    from deep inside its registry -- which is far harder to act on.
    """
    getter = getattr(torchvision.models, "get_model", None)
    if getter is not None:
        try:
            # Probe once to validate the name, then return a fresh-instance factory.
            getter(name, weights=None)
        except Exception as error:
            raise KeyError(f"torchvision has no model named {name!r}: {error}") from error
        return lambda: getter(name, weights=None)

    factory = getattr(torchvision.models, name, None)  # pragma: no cover - legacy path
    if factory is None:  # pragma: no cover
        raise KeyError(f"torchvision has no model named {name!r}")
    return lambda: factory(weights=None)  # pragma: no cover


def build_model(
    entry: ModelEntry,
    num_classes: int,
    input_size: int = 32,
    verify: bool = True,
) -> nn.Module:
    """Construct a randomly initialised model matching ``entry``.

    Weights are intentionally *not* loaded from ImageNet: every number in this
    study is produced by a model trained from scratch under an identical recipe,
    which is the only way to attribute a difference to the architecture.
    """
    if entry.family not in KNOWN_FAMILIES:
        raise ValueError(f"model {entry.id!r}: unknown family {entry.family!r}")

    model = _torchvision_factory(entry.torchvision_name)()

    replace_classifier_head(model, entry.torchvision_name, num_classes)

    if entry.adapt == "cifar":
        adapt_stem_to_cifar(model)

    if verify:
        verify_head(model, num_classes, input_size=input_size)

    logger.debug(
        "built %s (%s) for %d classes, adapt=%s",
        entry.id,
        entry.torchvision_name,
        num_classes,
        entry.adapt,
    )
    return model


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    model_id: str,
    num_classes: int,
    input_size: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist model weights together with everything needed to reinterpret them."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_id": model_id,
        "num_classes": num_classes,
        "input_size": input_size,
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    torch.save(payload, destination)
    logger.debug("saved checkpoint %s", destination)
    return destination


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load weights into ``model`` and return the surrounding metadata.

    Handles both the structured format written by :func:`save_checkpoint` and a
    bare ``state_dict``, so a user can drop in weights from elsewhere.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"checkpoint not found: {source}")

    payload = torch.load(source, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = dict(payload.get("metadata", {}))
        metadata["model_id"] = payload.get("model_id")
        metadata["num_classes"] = payload.get("num_classes")
        metadata["input_size"] = payload.get("input_size")
        metadata["saved_at"] = payload.get("saved_at")
        version = payload.get("format_version")
        if version is not None and version != CHECKPOINT_FORMAT_VERSION:
            logger.warning(
                "checkpoint %s has format version %s, expected %s",
                source,
                version,
                CHECKPOINT_FORMAT_VERSION,
            )
    else:
        state_dict = payload
        metadata = {"model_id": None, "format_version": None}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.warning("missing keys when loading %s: %s", source, missing)
    if unexpected:
        logger.warning("unexpected keys when loading %s: %s", source, unexpected)

    return metadata


def checkpoint_path(checkpoints_dir: str | Path, model_id: str, variant: str = "fp32") -> Path:
    """Canonical checkpoint filename for a (model, variant) pair."""
    return Path(checkpoints_dir) / f"{model_id}__{variant}.pt"
