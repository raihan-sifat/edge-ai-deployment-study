"""A single inference interface over every backend in the study.

The optimization ladder produces objects of very different kinds -- a compiled
``nn.Module``, a dynamically quantized module, an ONNX Runtime session. The
benchmark stage must not care. :class:`Predictor` is the seam: everything that
can answer "logits for this batch, and how big are you on disk" implements it.

Implementations are intentionally thin. Any real work (thread pinning, warmup,
timing) belongs to the benchmark protocol, not here.
"""

from __future__ import annotations

import contextlib
import gc
import io
import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Predictor(ABC):
    """Backend-agnostic inference callable."""

    #: Human-readable name shown in tables, e.g. ``"torch.compile"``.
    label: str = "unknown"
    #: Coarse backend family, e.g. ``"pytorch"`` or ``"onnxruntime"``.
    backend: str = "unknown"

    @abstractmethod
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, num_classes)`` for ``inputs``."""

    @abstractmethod
    def state_bytes(self) -> int:
        """Size in bytes of the artifact that would ship to the device."""

    def configure_threads(self, num_threads: int | None) -> None:
        """Pin the backend to ``num_threads`` intra-op threads.

        Thread count dominates CPU latency -- often more than the optimization
        being measured. Backends that cannot be reconfigured without rebuilding
        state override this; the default is a no-op so a missing override is a
        documented limitation rather than a crash.
        """
        return None

    def close(self) -> None:  # noqa: B027 - optional hook, not every backend has state
        """Release backend resources. Safe to call more than once.

        Deliberately not abstract: most predictors hold nothing that needs
        releasing, and forcing each one to implement a no-op would be noise.
        """

    def describe(self) -> dict[str, Any]:
        return {"label": self.label, "backend": self.backend}


class TorchPredictor(Predictor):
    """Wrap any ``nn.Module`` -- eager, compiled, quantized or pruned.

    ``torch.inference_mode()`` is used rather than ``torch.no_grad()``: it skips
    version-counter bookkeeping entirely, which is a measurable fraction of the
    runtime for the smallest models at batch size 1.
    """

    backend = "pytorch"

    def __init__(self, module: nn.Module, label: str = "pytorch") -> None:
        self.module = module
        self.label = label
        self.module.eval()

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            output = self.module(inputs)
        if isinstance(output, tuple):
            output = output[0]
        return output

    def state_bytes(self) -> int:
        return measure_state_dict_bytes(self.module)

    def configure_threads(self, num_threads: int | None) -> None:
        """Set the PyTorch intra-op thread pool.

        ``None`` restores everything the OS offers. Inter-op threads stay at 1
        because a single forward pass is inherently sequential and extra inter-op
        threads only add scheduling noise.
        """
        if num_threads is None:
            torch.set_num_threads(os.cpu_count() or 1)
        else:
            torch.set_num_threads(max(1, num_threads))
        # Inter-op threads stay at 1: a single forward pass is sequential, so extra
        # inter-op threads only add scheduling noise. Setting it twice in one
        # process raises, which is harmless here and must not abort the run.
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "module_class": type(self.module).__name__,
            "is_quantized": _is_quantized(self.module),
        }

    def close(self) -> None:
        self.module = self.module.cpu()
        gc.collect()


def measure_state_dict_bytes(module: nn.Module) -> int:
    """Serialise ``module.state_dict()`` to a memory buffer and report its size.

    This is the "what ships" number. It deliberately excludes the Python module
    graph, optimizer state and training metadata, because none of that travels to
    a device.

    Caveat worth knowing when reading the results: ``torch.save`` writes a zip
    container with a few kilobytes of framing, and that overhead is *constant*
    rather than proportional. For the architectures in this study (tens of MB) it
    is far below the measurement's precision, but for a model with only a few
    kilobytes of parameters it dominates the figure entirely. This is why a model
    that is 1.2 kB of weights can serialise to 3.7 kB.
    """
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def _is_quantized(module: nn.Module) -> bool:
    """Detect quantized modules across the eager and dynamic quantization APIs.

    There is no single reliable marker: dynamic quantization swaps in modules
    from ``torch.ao.nn.quantized.dynamic``, eager PTQ statically replaces modules
    and inserts observers turned into quantize/dequantize stubs, and FX-based
    flows produce a third layout. Checking the module path plus a couple of
    attribute markers covers all three.
    """
    for submodule in module.modules():
        submodule_path = type(submodule).__module__ or ""
        if ".quantized." in submodule_path or submodule_path.endswith("quantized"):
            return True
        if any(hasattr(submodule, attribute) for attribute in ("_packed_params", "scale")):
            return True
    return False
