"""CIFAR-10 adaptation of ImageNet classification architectures.

Every network in the study was designed for 224x224 ImageNet inputs and opens
with an aggressive downsample (a 7x7 stride-2 convolution, or a 3x3 stride-2
convolution followed by a stride-2 max-pool). Applied to 32x32 CIFAR images that
stem throws away spatial resolution the network never gets back: the 32x32 map
is 8x8 after the stem and 1x1 by the time the deepest stage runs.

The standard remedy is the one used by the well-known CIFAR reimplementations:
replace the stem with a single stride-1 3x3 convolution and drop the max-pool.
This keeps the input at 32x32 through the stem and shifts the first real
downsample into the first stage, which restores accuracy without changing the
macro-architecture being benchmarked.

The adaptation is applied *before* training, so the accuracy, parameter count,
FLOPs and latency reported for a model all describe the same object.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from edgebench.utils import get_logger

logger = get_logger("models.adapt")


def _iter_modules_with_names(model: nn.Module):
    """Yield ``(qualified_name, module)`` in definition order."""
    yield from model.named_modules()


def find_first_conv(model: nn.Module) -> tuple[str, nn.Conv2d]:
    """Return the qualified name and module of the first ``Conv2d``."""
    for name, module in _iter_modules_with_names(model):
        if isinstance(module, nn.Conv2d):
            return name, module
    raise ValueError(f"no Conv2d found in {type(model).__name__}")


def find_first_maxpool(model: nn.Module) -> tuple[str, nn.MaxPool2d] | None:
    """Return the first ``MaxPool2d``, or ``None`` when the network has none."""
    for name, module in _iter_modules_with_names(model):
        if isinstance(module, nn.MaxPool2d):
            return name, module
    return None


def get_submodule(model: nn.Module, qualified_name: str) -> nn.Module:
    """Fetch a submodule by dotted name; ``""`` returns ``model`` itself."""
    if not qualified_name:
        return model
    return model.get_submodule(qualified_name)


def set_submodule(model: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    """Replace a submodule addressed by dotted name, in place."""
    if not qualified_name:
        raise ValueError("cannot replace the root module")
    parent_name, _, attribute = qualified_name.rpartition(".")
    parent = get_submodule(model, parent_name) if parent_name else model
    if isinstance(parent, nn.Sequential):
        parent[int(attribute)] = replacement
    else:
        setattr(parent, attribute, replacement)


def adapt_stem_to_cifar(model: nn.Module, verbose: bool = True) -> nn.Module:
    """Rewrite the classification stem for 32x32 inputs, in place.

    Two edits are made:

    1. The first convolution becomes a stride-1, padding-1 3x3 convolution.
       A 7x7 kernel is *replaced* (its weights could not be reused anyway);
       a 3x3 kernel only has its stride changed.
    2. The first max-pool is replaced by identity, if one exists.

    Returns the same model object for chaining.
    """
    conv_name, conv = find_first_conv(model)

    if conv.kernel_size == (3, 3):
        if conv.stride != (1, 1):
            conv.stride = (1, 1)
            conv.padding = (1, 1)
            if verbose:
                logger.debug("%s: stem stride 2 -> 1", conv_name)
    else:
        replacement = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=conv.bias is not None,
        )
        # Preserve any device/dtype already chosen for the module.
        replacement = replacement.to(device=conv.weight.device, dtype=conv.weight.dtype)
        set_submodule(model, conv_name, replacement)
        if verbose:
            logger.debug(
                "%s: stem %dx%d stride 2 -> 3x3 stride 1",
                conv_name,
                *conv.kernel_size,
            )

    pool = find_first_maxpool(model)
    if pool is not None:
        pool_name, _pool_module = pool
        set_submodule(model, pool_name, nn.Identity())
        if verbose:
            logger.debug("%s: max-pool -> Identity", pool_name)

    return model


# ---------------------------------------------------------------------------
# Classifier heads
#
# torchvision's container layouts are stable but not uniform across families, so
# each head is addressed explicitly. If a future torchvision release reshuffles
# a classifier we want a loud failure here rather than a silently untrained head.
# ---------------------------------------------------------------------------


def _replace_linear(model: nn.Module, qualified_name: str, num_classes: int) -> None:
    current = get_submodule(model, qualified_name)
    if not isinstance(current, nn.Linear):
        raise TypeError(
            f"expected nn.Linear at {qualified_name!r} for {type(model).__name__}, "
            f"found {type(current).__name__}"
        )
    set_submodule(model, qualified_name, nn.Linear(current.in_features, num_classes))


def _head_resnet(model: nn.Module, num_classes: int) -> None:
    _replace_linear(model, "fc", num_classes)


def _head_mobilenet_v2(model: nn.Module, num_classes: int) -> None:
    _replace_linear(model, "classifier.1", num_classes)


def _head_mobilenet_v3(model: nn.Module, num_classes: int) -> None:
    _replace_linear(model, "classifier.3", num_classes)


def _head_shufflenet(model: nn.Module, num_classes: int) -> None:
    _replace_linear(model, "fc", num_classes)


def _head_efficientnet(model: nn.Module, num_classes: int) -> None:
    _replace_linear(model, "classifier.1", num_classes)


HEAD_ADAPTERS: dict[str, Callable[[nn.Module, int], None]] = {
    "resnet18": _head_resnet,
    "resnet34": _head_resnet,
    "mobilenet_v2": _head_mobilenet_v2,
    "mobilenet_v3_small": _head_mobilenet_v3,
    "mobilenet_v3_large": _head_mobilenet_v3,
    "shufflenet_v2_x1_0": _head_shufflenet,
    "shufflenet_v2_x1_5": _head_shufflenet,
    "shufflenet_v2_x2_0": _head_shufflenet,
    "efficientnet_b0": _head_efficientnet,
    "efficientnet_b1": _head_efficientnet,
}


def replace_classifier_head(model: nn.Module, torchvision_name: str, num_classes: int) -> None:
    """Swap the classifier head for ``num_classes`` outputs."""
    adapter = HEAD_ADAPTERS.get(torchvision_name)
    if adapter is None:
        raise KeyError(f"no classifier head adapter registered for {torchvision_name!r}")
    adapter(model, num_classes)


def verify_head(model: nn.Module, num_classes: int, input_size: int = 32) -> None:
    """Fail fast if the head does not emit ``num_classes`` logits.

    Cheaper to discover here than after a 30-epoch training run.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, input_size, input_size))
    finally:
        model.train(was_training)

    if output.ndim != 2 or output.shape[1] != num_classes:
        raise ValueError(
            f"expected logits of shape (batch, {num_classes}), got {tuple(output.shape)}"
        )
