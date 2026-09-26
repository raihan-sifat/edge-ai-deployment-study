"""``torch.compile`` as a deployment optimization.

Two caveats drive the implementation, both of which matter far more on an edge
device than on a server:

1. **Compilation is a one-time cost paid at cold start.** A model that is 30%
   faster in steady state but takes 20 seconds to compile is often worse for an
   appliance that boots frequently. The compile wall time is recorded.
2. **It is not always available.** On Windows, Triton -- the backend Inductor
   needs for meaningful GPU/CPU code generation -- is frequently absent, and
   ``torch.compile`` degrades to a no-op graph or raises outright. That outcome
   is recorded as ``unavailable`` rather than hidden.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from edgebench.inference import TorchPredictor
from edgebench.optim.base import (
    OptimizationContext,
    OptimizationOutcome,
    failed,
    unavailable,
)
from edgebench.utils import get_logger

logger = get_logger("optim.compile")

#: Substrings that indicate a missing or unusable toolchain rather than a bug in
#: the model. Observed examples on Windows without the MSVC build tools:
#: ``InvalidCxxCompiler: Compiler: cl is not found`` and, without Triton,
#: ``Cannot find a working triton installation``.
_UNAVAILABLE_MARKERS = (
    "triton",
    "compiler",
    "is not found",
    "could not find",
    "cannot find",
    "no backend",
    "not supported",
    "does not support",
    "unsupported",
    "not implemented",
    "not available",
)


def apply_compile(ctx: OptimizationContext, params: dict[str, Any]) -> OptimizationOutcome:
    """Wrap the model in ``torch.compile`` and force compilation with a warmup.

    Supported params:
        ``mode``: Inductor mode -- ``default``, ``reduce-overhead`` or
            ``max-autotune``. ``reduce-overhead`` uses CUDA graphs and is
            therefore meaningless on CPU; CPU runs use ``default``.
        ``fullgraph``: require the whole forward pass to be captured in one graph.
        ``dynamic``: enable dynamic shapes (off -- fixed input shape is the
            realistic edge case and gives Inductor more to work with).
    """
    mode = str(params.get("mode", "default"))
    fullgraph = bool(params.get("fullgraph", False))
    dynamic = bool(params.get("dynamic", False))

    if not hasattr(torch, "compile"):  # pragma: no cover - requires torch < 2.0
        return unavailable("compile", "torch.compile requires PyTorch >= 2.0", params=params)

    module = ctx.model

    compile_started = time.perf_counter()
    try:
        compiled = torch.compile(module, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

        # Inductor is lazy: the first call is where compilation actually happens,
        # so timing must happen here rather than around the `torch.compile` call.
        warmup_inputs = ctx.example_inputs(1)
        with torch.inference_mode():
            compiled(warmup_inputs)
        compile_seconds = time.perf_counter() - compile_started
    except BaseException as error:
        message = str(error).lower()
        if any(marker in message for marker in _UNAVAILABLE_MARKERS):
            return unavailable(
                "compile",
                f"no working torch.compile backend on this machine: {error}",
                params=params,
                notes=["common cause: Triton is unavailable on Windows or on this CPU"],
            )
        if isinstance(error, KeyboardInterrupt):
            raise
        return failed("compile", error, params=params)

    logger.info("%s: torch.compile(mode=%s) ready in %.2fs", ctx.model_id, mode, compile_seconds)

    return OptimizationOutcome(
        optimization_id="compile",
        status="applied",
        params={"mode": mode, "fullgraph": fullgraph, "dynamic": dynamic},
        predictor=TorchPredictor(compiled, label=f"torch.compile[{mode}]"),
        metadata={
            "compile_seconds": round(compile_seconds, 3),
            "compile_lazy": True,
        },
        notes=[
            "compilation happens on first call; the recorded time is that first call",
            "steady-state latency is measured after the standard warmup, so the "
            "cold-start cost is excluded from the latency figures but reported separately",
        ],
    )
