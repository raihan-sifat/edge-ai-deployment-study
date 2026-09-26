"""ONNX Runtime as a deployment backend.

This entry in the ladder is different in kind from the others: it changes the
*runtime*, not just the model. That is exactly why it is worth measuring, because
in production an INT8 PyTorch graph and an INT8 ONNX graph are not the same
artifact and do not have the same latency.

It is also the entry most likely to be unavailable, so the checks are explicit:
``onnx`` and ``onnxruntime`` are optional dependencies, and the import failure is
reported as a reason rather than swallowed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from edgebench.inference import Predictor
from edgebench.optim.base import (
    OptimizationContext,
    OptimizationOutcome,
    failed,
    is_fatal,
    unavailable,
)
from edgebench.utils import get_logger

logger = get_logger("optim.onnx")

CPU_PROVIDER = "CPUExecutionProvider"


def onnx_available() -> tuple[bool, str]:
    """Whether the ONNX toolchain can be used, with a reason when it cannot."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        return False, "the 'onnx' package is not installed (pip install 'edgebench[onnx]')"

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return (
            False,
            "the 'onnxruntime' package is not installed (pip install 'edgebench[onnx]')",
        )

    return True, "available"


def export_to_onnx(
    module: nn.Module,
    destination: str | Path,
    input_size: int,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    batch_size_for_export: int = 1,
) -> Path:
    """Export ``module`` to ONNX with an optional dynamic batch axis.

    The batch axis is dynamic by default because edge serving is usually
    batch-of-one with occasional bursts; baking in a fixed batch size would make
    the exported graph useless for the batch sweep.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    was_training = module.training
    module.eval()

    example = torch.randn(batch_size_for_export, 3, input_size, input_size)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None

    try:
        torch.onnx.export(
            module,
            (example,),
            str(target),
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            # Legacy exporter (dynamo=False) is used because the FX quantized and
            # dynamically quantized modules are not supported by the dynamo exporter.
            dynamo=False,
        )
    finally:
        module.train(was_training)

    return target


def quantize_onnx_graph(
    source: Path,
    destination: Path,
    per_channel: bool = True,
) -> Path:
    """Post-training dynamic INT8 quantization of an ONNX graph.

    This uses ONNX Runtime's own quantizer, which is a different implementation
    from PyTorch's. Comparing the two is one of the more useful cross-checks the
    study provides: the same nominal "INT8" can produce a different accuracy and a
    different size depending on who did the quantization.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    return destination


class OnnxRuntimePredictor(Predictor):
    """An ``onnxruntime.InferenceSession`` presented through the ``Predictor`` API."""

    backend = "onnxruntime"

    def __init__(
        self,
        model_path: str | Path,
        label: str = "onnxruntime",
        num_threads: int | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.label = label

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self._num_threads = num_threads
        self._providers = providers or [CPU_PROVIDER]
        self._session: Any = None
        self._input_name = "input"
        self._rebuild_session()

    # -- session management -------------------------------------------------

    def _rebuild_session(self) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        # 0 means "let ORT choose", which is its default-spin heuristic.
        options.intra_op_num_threads = self._num_threads or 0
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3  # errors only

        available = ort.get_available_providers()
        providers = [name for name in self._providers if name in available] or available

        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name

    def configure_threads(self, num_threads: int | None) -> None:
        """Rebuild the session with a new intra-op thread count.

        ONNX Runtime fixes the thread pool when the session is created, so unlike
        the PyTorch backend this cannot be a cheap setter.
        """
        if num_threads == self._num_threads:
            return
        self._num_threads = num_threads
        self._rebuild_session()

    # -- inference ---------------------------------------------------------

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        # ORT has no zero-copy path from a torch tensor, but on CPU the tensor
        # already lives in host memory, so `.numpy()` is a view rather than a copy.
        array = inputs.detach().cpu().numpy()
        outputs = self._session.run(None, {self._input_name: array})
        logits = outputs[0]
        if isinstance(logits, np.ndarray):
            return torch.from_numpy(logits)
        return torch.as_tensor(logits)

    def state_bytes(self) -> int:
        return self.model_path.stat().st_size

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "model_path": str(self.model_path),
            "providers": list(self._session.get_providers()),
            "intra_op_num_threads": self._num_threads,
            "model_bytes": self.state_bytes(),
        }

    def close(self) -> None:
        self._session = None


def apply_onnxruntime(ctx: OptimizationContext, params: dict[str, Any]) -> OptimizationOutcome:
    """Export to ONNX, optionally quantize the graph, and run it via ONNX Runtime.

    Supported params:
        ``quantize``: apply ONNX Runtime dynamic INT8 quantization to the graph.
        ``opset``: target ONNX opset (default 17).
        ``dynamic_batch``: keep the batch axis dynamic (default on).
        ``per_channel``: per-channel weight quantization when ``quantize`` is set.
    """
    optimization_id = str(
        params.get(
            "optimization_id", "onnxruntime_int8" if params.get("quantize") else "onnxruntime"
        )
    )

    ok, reason = onnx_available()
    if not ok:
        return unavailable(optimization_id, reason, params=params)

    opset = int(params.get("opset", 17))
    dynamic_batch = bool(params.get("dynamic_batch", True))
    quantize = bool(params.get("quantize", False))
    per_channel = bool(params.get("per_channel", True))

    fp32_path = ctx.artifacts_dir / f"{ctx.model_id}__fp32.onnx"
    final_path = fp32_path
    if quantize:
        final_path = ctx.artifacts_dir / f"{ctx.model_id}__int8.onnx"

    try:
        export_to_onnx(
            ctx.model,
            fp32_path,
            input_size=ctx.input_size,
            opset_version=opset,
            dynamic_batch=dynamic_batch,
        )
        if quantize:
            quantize_onnx_graph(fp32_path, final_path, per_channel=per_channel)
    except BaseException as error:
        if is_fatal(error):
            raise
        return failed(optimization_id, error, params=params)

    try:
        predictor = OnnxRuntimePredictor(
            final_path,
            label="onnxruntime-int8" if quantize else "onnxruntime-fp32",
        )
    except BaseException as error:
        if is_fatal(error):
            raise
        return failed(optimization_id, error, params=params)

    artifacts = {"onnx_model": str(final_path)}
    if quantize:
        artifacts["onnx_model_fp32"] = str(fp32_path)

    fp32_bytes = fp32_path.stat().st_size
    final_bytes = final_path.stat().st_size

    logger.info(
        "%s: ONNX %s export complete (%s -> %s)",
        ctx.model_id,
        "INT8" if quantize else "FP32",
        _fmt(fp32_bytes),
        _fmt(final_bytes),
    )

    return OptimizationOutcome(
        optimization_id=optimization_id,
        status="applied",
        params={
            **params,
            "opset": opset,
            "dynamic_batch": dynamic_batch,
            "quantize": quantize,
            "per_channel": per_channel,
        },
        predictor=predictor,
        artifacts=artifacts,
        metadata={
            "opset": opset,
            "dynamic_batch": dynamic_batch,
            "onnx_fp32_bytes": fp32_bytes,
            "onnx_final_bytes": final_bytes,
            "graph_quantized": quantize,
            "per_channel": per_channel,
            "weight_bits": 8 if quantize else 32,
        },
        notes=[
            "ONNX Runtime applies its own graph optimizations (ORT_ENABLE_ALL) at "
            "session creation time",
            "the ONNX size includes graph metadata, so it is larger than the raw "
            "weight bytes of the equivalent PyTorch model",
        ],
    )


def _fmt(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB"
