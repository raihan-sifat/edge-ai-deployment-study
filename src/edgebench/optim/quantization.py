"""Post-training and quantization-aware INT8 quantization.

Three variants are implemented because they answer three different questions:

``dynamic_int8``
    Weights are quantized ahead of time; activations are quantized on the fly per
    batch. Cheap, no calibration data, and for a convolutional network it mostly
    touches the classifier -- the parameter-light part. Included precisely
    because the *disappointing* result is the interesting one: it shows that
    "apply INT8 quantization" is not a single well-defined action.

``static_int8``
    Weight and activation quantization with pre-computed activation ranges
    obtained from a calibration pass. This is what an edge deployment normally
    means, and it is where the real latency win lives -- provided the model is
    FX-traceable and the backend supports the operators used.

``qat_int8``
    Static quantization with the observers learned during a short fine-tune, which
    recovers most of the accuracy that post-training static quantization loses on
    small models.

All three are wrapped so that an unsupported operator or a missing backend
produces an ``unavailable``/``failed`` outcome with a reason, not a crash.
"""

from __future__ import annotations

import platform
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edgebench.inference import TorchPredictor
from edgebench.optim.base import (
    OptimizationContext,
    OptimizationOutcome,
    clone_module,
    failed,
    is_fatal,
    unavailable,
)
from edgebench.training import fit
from edgebench.utils import get_logger

logger = get_logger("optim.quant")

DEFAULT_CALIBRATION_BATCHES = 16

_ARM_MACHINES = {"aarch64", "arm64", "armv7l", "armv8l", "arm64e"}

#: Engine preference on x86. Recent PyTorch builds replace ``fbgemm`` with
#: ``onednn`` as the engine name while still accepting ``x86``/``fbgemm`` as
#: *qconfig* names, so the two namespaces are resolved separately below.
_X86_ENGINE_PREFERENCE = ("onednn", "x86", "fbgemm")
_ARM_ENGINE_PREFERENCE = ("qnnpack", "onednn", "x86", "fbgemm")


def select_quantization_backend() -> str:
    """Pick the kernel backend to assign to ``torch.backends.quantized.engine``.

    Only names present in ``supported_engines`` are valid here; assigning anything
    else raises. ``fbgemm`` is the historical x86 name and ``onednn`` is its
    current successor, so both are probed in order rather than assumed.

    This is also what a Raspberry-Pi-class deployment needs: ``qnnpack`` is the ARM
    path, and using the x86 path there would silently produce unoptimised kernels.
    """
    supported = list(torch.backends.quantized.supported_engines)
    machine = platform.machine().lower()
    preference = _ARM_ENGINE_PREFERENCE if machine in _ARM_MACHINES else _X86_ENGINE_PREFERENCE

    for candidate in preference:
        if candidate in supported:
            return candidate

    logger.warning("none of %s is available; falling back to %s", preference, supported or ["none"])
    return supported[0] if supported else "none"


def select_qconfig_backend(engine: str) -> str:
    """Resolve the name to pass to ``torch.ao.quantization.get_default_qconfig``.

    ``get_default_qconfig`` accepts ``x86`` and ``fbgemm`` even on builds whose only
    engine is ``onednn``, so the engine name cannot simply be reused. Each candidate
    is probed because the accepted set varies between releases.
    """
    candidates = ["qnnpack"] if engine == "qnnpack" else ["x86", "onednn", "fbgemm", engine]
    for candidate in candidates:
        try:
            torch.ao.quantization.get_default_qconfig(candidate)
        except Exception:
            continue
        return candidate
    return engine


def _dynamic_conv_supported() -> tuple[bool, str]:
    """Whether dynamic quantization can convert ``Conv2d`` on this PyTorch.

    PyTorch only added dynamic-quantizable convolutions recently. On older builds
    ``quantize_dynamic`` silently leaves every convolution in FP32, which turns
    the result into a statement about classifiers rather than about models.
    """
    try:
        from torch.ao.quantization.quantization_mappings import (
            get_default_dynamic_quant_module_mappings,
        )
    except ImportError:  # pragma: no cover - very old torch
        return False, "dynamic quantization mappings unavailable"

    mapping = get_default_dynamic_quant_module_mappings()
    if nn.Conv2d in mapping:
        return True, "Conv2d and Linear weights are dynamically quantized"
    return False, "only Linear (and recurrent) weights are dynamically quantized on this build"


# ---------------------------------------------------------------------------
# Dynamic quantization
# ---------------------------------------------------------------------------


def apply_dynamic_quantization(
    ctx: OptimizationContext, params: dict[str, Any]
) -> OptimizationOutcome:
    """Apply dynamic INT8 quantization to eligible submodules.

    Supported params:
        ``include_conv``: force-include ``Conv2d`` even when the build's dynamic
            mappings omit it (off by default -- it would be a silent no-op).
        ``dtype``: ``qint8`` (default) or ``quint8``.
    """
    optimization_id = str(params.get("optimization_id", "dynamic_int8"))
    dtype = torch.qint8 if str(params.get("dtype", "qint8")) == "qint8" else torch.quint8

    conv_supported, conv_reason = _dynamic_conv_supported()
    include_conv = bool(params.get("include_conv", conv_supported))

    spec: set[type[nn.Module]] = {nn.Linear}
    if include_conv:
        spec.add(nn.Conv2d)

    model = clone_module(ctx.model).eval()

    try:
        quantize_dynamic = torch.ao.quantization.quantize_dynamic
        started = time.perf_counter()
        quantized = quantize_dynamic(model, qconfig_spec=spec, dtype=dtype, inplace=False)
        elapsed = time.perf_counter() - started
    except BaseException as error:
        if is_fatal(error):
            raise
        return failed(optimization_id, error, params=params)

    converted = _count_quantized_modules(quantized)
    if converted == 0:
        return unavailable(
            optimization_id,
            "dynamic quantization converted no modules on this model/build",
            params=params,
        )

    notes = [
        "activations are quantized per batch at run time; there is no calibration step",
        conv_reason,
    ]
    if not include_conv:
        notes.append(
            "convolutions stay in FP32, so the weight-size reduction is limited to "
            "the classifier head -- most parameters are unaffected"
        )

    logger.info(
        "%s: dynamic %s quantization converted %d modules in %.2fs",
        ctx.model_id,
        dtype,
        converted,
        elapsed,
    )

    return OptimizationOutcome(
        optimization_id=optimization_id,
        status="applied",
        params={**params, "dtype": str(dtype), "include_conv": include_conv},
        predictor=TorchPredictor(quantized, label="dynamic-int8"),
        metadata={
            "quantized_modules": converted,
            "conv_dynamically_quantized": include_conv,
            "quantization_seconds": round(elapsed, 3),
            "weight_bits": 8,
            "activation_bits": 8,
        },
        notes=notes,
    )


def _count_quantized_modules(module: nn.Module) -> int:
    count = 0
    for submodule in module.modules():
        path = type(submodule).__module__ or ""
        if ".quantized." in path or path.endswith("quantized"):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Static (post-training) quantization via FX
# ---------------------------------------------------------------------------


def _fx_quantization_api() -> tuple[Any, Any, Any] | None:
    try:
        from torch.ao.quantization.quantize_fx import (
            convert_fx,
            prepare_fx,
            prepare_qat_fx,
        )
    except ImportError:  # pragma: no cover
        return None
    return prepare_fx, convert_fx, prepare_qat_fx


def _build_qconfig_mapping(engine: str, params: dict[str, Any]) -> Any:
    """Build a ``QConfigMapping`` for the selected engine.

    A mapping is used rather than a bare qconfig dict because ``prepare_fx``
    special-cases fixed-qparams operators (sigmoid, softmax, and the hard-swish
    path that MobileNetV3 relies on) only when given a mapping. Passing a bare
    qconfig triggers a warning about exactly that, and silently ignores the
    correct handling for those ops.

    Two knobs are exposed because they materially change the accuracy/speed
    trade and are otherwise invisible:

    ``qconfig_backend``
        Which default qconfig to start from. On an ``onednn`` engine the ``x86``
        factory (``reduce_range=True``) and the ``onednn`` factory
        (``reduce_range=False``) differ in activation range.
    ``reduce_range``
        Override the activation observer's reduced range. Reduced range protects
        older x86 parts from saturation but costs accuracy on VNNI-capable CPUs.
    """
    from torch.ao.quantization.quantize_fx import QConfigMapping

    qconfig_backend = str(params.get("qconfig_backend") or select_qconfig_backend(engine))

    try:
        mapping = torch.ao.quantization.get_default_qconfig_mapping(qconfig_backend)
    except Exception as error:
        logger.warning(
            "get_default_qconfig_mapping(%r) failed (%s); falling back to an explicit qconfig",
            qconfig_backend,
            error,
        )
        mapping = QConfigMapping().set_global(_explicit_qconfig())

    reduce_range = params.get("reduce_range")
    if reduce_range is None:
        return mapping

    override = _explicit_qconfig(reduce_range=bool(reduce_range))
    return QConfigMapping().set_global(override)


def _explicit_qconfig(reduce_range: bool = False) -> Any:
    """Hand-built qconfig used only when a default factory is unavailable."""
    observer = torch.ao.quantization.observer
    return torch.ao.quantization.QConfig(
        activation=observer.HistogramObserver.with_args(reduce_range=reduce_range),
        weight=observer.PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )


def _calibrate(
    prepared: nn.Module,
    loader: DataLoader[Any] | None,
    max_batches: int,
) -> int:
    """Run the observer calibration pass and return how many batches were used."""
    if loader is None:
        raise ValueError("calibration loader is required for static quantization")

    observed = 0
    with torch.inference_mode():
        for inputs, _targets in loader:
            prepared(inputs)
            observed += 1
            if observed >= max_batches:
                break
    if observed == 0:
        raise ValueError("calibration loader produced no batches")
    return observed


def apply_static_quantization(
    ctx: OptimizationContext, params: dict[str, Any]
) -> OptimizationOutcome:
    """Post-training static INT8 quantization using an FX graph rewrite.

    Supported params:
        ``calibration_batches``: number of batches fed to the observers.
        ``qconfig_backend`` / ``reduce_range``: see :func:`_build_qconfig`.
    """
    optimization_id = str(params.get("optimization_id", "static_int8"))
    max_batches = int(params.get("calibration_batches", DEFAULT_CALIBRATION_BATCHES))

    api = _fx_quantization_api()
    if api is None:
        return unavailable(optimization_id, "torch.ao.quantization.quantize_fx unavailable", params)
    prepare_fx, convert_fx, _ = api

    if ctx.calib_loader is None:
        return unavailable(optimization_id, "no calibration loader supplied", params=params)

    engine = select_quantization_backend()
    supported = set(torch.backends.quantized.supported_engines)
    if engine not in supported:
        return unavailable(
            optimization_id,
            f"no usable INT8 quantization engine in this PyTorch build "
            f"(available: {sorted(supported)})",
            params=params,
        )

    previous_engine = torch.backends.quantized.engine
    torch.backends.quantized.engine = engine

    qconfig_backend = str(params.get("qconfig_backend") or select_qconfig_backend(engine))
    qconfig_mapping = _build_qconfig_mapping(engine, params)
    example_inputs = (ctx.example_inputs(2),)
    model = clone_module(ctx.model).eval()

    try:
        started = time.perf_counter()
        prepared = prepare_fx(model, qconfig_mapping, example_inputs)
        observed = _calibrate(prepared, ctx.calib_loader, max_batches)
        converted = convert_fx(prepared)
        elapsed = time.perf_counter() - started
    except BaseException as error:
        torch.backends.quantized.engine = previous_engine
        if is_fatal(error):
            raise
        return failed(optimization_id, error, params=params)

    converted.eval()

    logger.info(
        "%s: static INT8 quantization (engine=%s, qconfig=%s) calibrated on %d batches in %.1fs",
        ctx.model_id,
        engine,
        qconfig_backend,
        observed,
        elapsed,
    )

    return OptimizationOutcome(
        optimization_id=optimization_id,
        status="applied",
        params={
            **params,
            "engine": engine,
            "qconfig_backend": qconfig_backend,
            "calibration_batches": observed,
        },
        predictor=TorchPredictor(converted, label="static-int8"),
        metadata={
            "calibration_batches": observed,
            "quantization_seconds": round(elapsed, 3),
            "engine": engine,
            "qconfig_backend": qconfig_backend,
            "weight_bits": 8,
            "activation_bits": 8,
        },
        notes=[
            "activation ranges come from the validation split; the test split is "
            "used only for the reported accuracy",
            "FX graph mode performs conv+bn+relu fusion automatically",
        ],
    )


# ---------------------------------------------------------------------------
# Quantization-aware training
# ---------------------------------------------------------------------------


def apply_qat_quantization(ctx: OptimizationContext, params: dict[str, Any]) -> OptimizationOutcome:
    """Fine-tune with fake-quant observers in place, then convert to INT8.

    Supported params:
        ``epochs`` / ``learning_rate``: override ``train.qat_epochs`` and
            ``train.qat_lr``.
        ``calibration_batches``: unused (QAT learns activation ranges), accepted
            for symmetry with the other quantizers.
    """
    optimization_id = str(params.get("optimization_id", "qat_int8"))

    api = _fx_quantization_api()
    if api is None:
        return unavailable(optimization_id, "torch.ao.quantization.quantize_fx unavailable", params)
    _, convert_fx, prepare_qat_fx = api

    if ctx.train_loader is None or ctx.val_loader is None:
        return unavailable(
            optimization_id, "QAT requires training and validation loaders", params=params
        )

    engine = select_quantization_backend()
    supported = set(torch.backends.quantized.supported_engines)
    if engine not in supported:
        return unavailable(
            optimization_id,
            f"no usable INT8 quantization engine in this PyTorch build "
            f"(available: {sorted(supported)})",
            params=params,
        )

    previous_engine = torch.backends.quantized.engine
    torch.backends.quantized.engine = engine

    epochs = int(params.get("epochs", ctx.train_cfg.qat_epochs))
    learning_rate = float(params.get("learning_rate", ctx.train_cfg.qat_lr))

    qconfig_backend = str(params.get("qconfig_backend") or select_qconfig_backend(engine))
    qconfig_mapping = _build_qconfig_mapping(engine, params)
    example_inputs = (ctx.example_inputs(2),)
    model = clone_module(ctx.model).train()

    try:
        started = time.perf_counter()
        prepared = prepare_qat_fx(model, qconfig_mapping, example_inputs)
        history = fit(
            prepared,
            ctx.train_loader,
            ctx.val_loader,
            ctx.train_cfg,
            ctx.device,
            ctx.num_classes,
            epochs=epochs,
            learning_rate=learning_rate,
            tag=f"qat:{ctx.model_id}",
        )
        prepared.eval()
        converted = convert_fx(prepared)
        elapsed = time.perf_counter() - started
    except BaseException as error:
        torch.backends.quantized.engine = previous_engine
        if is_fatal(error):
            raise
        return failed(optimization_id, error, params=params)

    converted.eval()

    logger.info(
        "%s: QAT finished %d epoch(s) in %.1fs (best val top-1 %.2f%%)",
        ctx.model_id,
        epochs,
        elapsed,
        100 * history.best_top1,
    )

    return OptimizationOutcome(
        optimization_id=optimization_id,
        status="applied",
        params={
            **params,
            "engine": engine,
            "qconfig_backend": qconfig_backend,
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        predictor=TorchPredictor(converted, label="qat-int8"),
        metadata={
            "qat_epochs": epochs,
            "qat_learning_rate": learning_rate,
            "qat_seconds": round(elapsed, 3),
            "qat_best_val_top1": history.best_top1,
            "qat_history": history.to_dict(),
            "engine": engine,
            "qconfig_backend": qconfig_backend,
            "weight_bits": 8,
            "activation_bits": 8,
        },
        notes=[
            "this is the only entry in the ladder that consumes training compute "
            "at optimization time, not just at training time",
            "fine-tuning cost is excluded from the latency measurement but is a real "
            "deployment cost and is reported separately",
        ],
    )
