"""Check whether the pruned-model slowdown is real or a measurement-order artifact.

Motivation: in the first offline run, `prune_unstructured_50` measured 4x slower
than FP32 and `onnxruntime_int8` measured 9x slower. Both are implausible --
magnitude pruning keeps dense tensors, and ONNX Runtime's quantized kernels are
usually faster than eager PyTorch, not slower.

The hypothesis is *measurement order*: configurations were benchmarked
sequentially over several minutes, so the later ones ran on a warmer, noisier
machine. This script tests that by measuring the same configurations twice in
opposite orders and comparing.

Run:  python scripts/check_measurement_order.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgebench.bench.latency import measure_latency
from edgebench.config import ModelEntry
from edgebench.inference import TorchPredictor
from edgebench.models import build_model, checkpoint_path, load_checkpoint
from edgebench.optim import apply_identity
from edgebench.optim.base import OptimizationContext
from edgebench.optim.pruning import apply_unstructured_pruning

CHECKPOINT_DIR = Path("checkpoints/offline")
WARMUP = 5
TIMED = 30
REPEATS = 3


def build_predictors() -> dict[str, TorchPredictor]:
    model = build_model(ModelEntry(id="resnet18"), num_classes=10, input_size=32)
    load_checkpoint(model, checkpoint_path(CHECKPOINT_DIR, "resnet18", "fp32"))
    model.eval()

    ctx = OptimizationContext(
        model=model,
        model_id="resnet18",
        num_classes=10,
        input_size=32,
        device=torch.device("cpu"),
        seed=0,
        artifacts_dir=Path("results/offline/raw/onnx"),
        train_cfg=None,  # type: ignore[arg-type] - unused by these two optimizations
        bench_cfg=None,  # type: ignore[arg-type]
    )

    fp32 = apply_identity(ctx, {"optimization_id": "fp32"})
    pruned = apply_unstructured_pruning(
        ctx, {"optimization_id": "prune_unstructured_50", "amount": 0.5}
    )

    assert fp32.predictor and pruned.predictor
    return {"fp32": fp32.predictor, "prune50": pruned.predictor}


def measure(label: str, predictor: TorchPredictor) -> float:
    result = measure_latency(
        predictor,
        resolution=32,
        batch_size=1,
        warmup_iters=WARMUP,
        timed_iters=TIMED,
        repeats=REPEATS,
        num_threads=1,
        measure_memory=False,
    )
    assert result.per_iteration, f"{label}: measurement failed ({result.reason})"
    cv = result.per_iteration["cv"]
    print(
        f"  {label:<10} p50={result.latency_ms:7.2f} ms  "
        f"mean={result.per_iteration['mean_ms']:7.2f}  cv={100 * cv:5.1f}%"
    )
    return float(result.latency_ms)


def main() -> int:
    torch.set_num_threads(1)
    predictors = build_predictors()

    print("\nWarm-up pass (discarded) ...")
    for predictor in predictors.values():
        measure_latency(
            predictor,
            resolution=32,
            batch_size=1,
            warmup_iters=2,
            timed_iters=5,
            repeats=1,
            num_threads=1,
            measure_memory=False,
        )

    orders = [
        ("A: fp32 then prune50", ["fp32", "prune50"]),
        ("B: prune50 then fp32", ["prune50", "fp32"]),
        ("C: fp32 then prune50 (repeat)", ["fp32", "prune50"]),
    ]

    results: dict[str, list[float]] = {"fp32": [], "prune50": []}
    for name, order in orders:
        print(f"\nOrder {name}")
        for key in order:
            results[key].append(measure(key, predictors[key]))

    print("\n" + "=" * 62)
    for key, values in results.items():
        spread = max(values) - min(values)
        print(
            f"{key:<10} median={statistics.median(values):7.2f} ms  "
            f"min={min(values):7.2f}  max={max(values):7.2f}  spread={spread:6.2f} ms"
        )

    ratio = statistics.median(results["prune50"]) / statistics.median(results["fp32"])
    print(f"\nprune50 / fp32 median ratio: {ratio:.2f}x")

    between = max(statistics.median(results[k]) for k in results) / min(
        statistics.median(results[k]) for k in results
    )
    within = max(max(v) - min(v) for v in results.values() if len(v) > 1) / min(
        statistics.median(results[k]) for k in results
    )

    print(f"between-configuration effect: {between:.2f}x")
    print(f"within-configuration spread : {within * 100:.1f}% of median")

    if between < 1.15:
        print(
            "\nVERDICT: the effect is within measurement noise. The large gap in the "
            "full run was a measurement-order artifact, not a property of pruning."
        )
    else:
        print("\nVERDICT: the effect exceeds run-to-run spread and is likely real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
