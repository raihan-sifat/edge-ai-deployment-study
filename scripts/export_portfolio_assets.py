#!/usr/bin/env python
"""Export figures and metrics for the portfolio case study.

Produces two things from a results directory:

``portfolio/images/``
    Web-optimised copies of every figure, in WebP (primary) and PNG (fallback),
    so the case study can ship without a build-time image pipeline.

``portfolio/data/results.json``
    A compact, typed snapshot of the headline numbers, the environment, and which
    optimizations were unavailable. This is what the case study renders from, so
    the page is never hand-edited with numbers that can drift from the results.

Crucially, the export is **honest about placeholder data**. If the results
directory came from the synthetic offline config, ``provisional`` is set to true
and the case study renders a visible warning. Publishing synthetic accuracy as if
it were a real measurement is exactly the failure mode this project exists to
avoid.

Usage::

    python scripts/export_portfolio_assets.py                  # from results/
    python scripts/export_portfolio_assets.py --results results/offline
    python scripts/export_portfolio_assets.py --width 1400 --quality 88
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Figures that belong in the case study, with the alt text each one needs.
#: Alt text lives here rather than in the MDX so it stays next to the thing it
#: describes and is impossible to forget when a figure is added.
FIGURE_SPEC: dict[str, dict[str, str]] = {
    "accuracy_vs_latency": {
        "caption": "Accuracy against single-thread batch-1 latency",
        "alt": (
            "Scatter plot of test accuracy against latency for every model and "
            "optimization, one connected series per architecture, on a logarithmic "
            "latency axis. Static INT8 points sit far to the left of their FP32 "
            "counterparts."
        ),
    },
    "optimization_ladder__resnet18": {
        "caption": "Per-optimization speedup and accuracy change for ResNet-18",
        "alt": (
            "Two stacked bar charts for ResNet-18: speedup relative to FP32 above, "
            "and change in accuracy in percentage points below. Static and QAT INT8 "
            "show roughly three times speedup; dynamic INT8 sits below 1.0."
        ),
    },
    "latency_distribution__resnet18": {
        "caption": "Latency distribution across optimizations",
        "alt": (
            "Box plots of individual per-iteration latency samples for each "
            "optimization, with the raw samples overlaid as jittered points, showing "
            "the spread that a single median would hide."
        ),
    },
    "size_vs_accuracy": {
        "caption": "Model size against accuracy, with marker area proportional to MACs",
        "alt": (
            "Scatter plot of serialized weight size against test accuracy on a "
            "logarithmic size axis, with marker area proportional to "
            "multiply-accumulates."
        ),
    },
    "per_class_accuracy": {
        "caption": "Per-class accuracy change relative to each model's FP32 baseline",
        "alt": (
            "Heatmap of per-class accuracy change in percentage points, one row per "
            "model and optimization, showing that aggregate accuracy hides "
            "class-specific degradation."
        ),
    },
    "resolution_sensitivity": {
        "caption": "Latency at 32x32 versus 224x224",
        "alt": (
            "Grouped bar chart of FP32 latency at each measured input resolution on a "
            "logarithmic axis. 224x224 has 49 times the pixels but does not cost 49 "
            "times the latency."
        ),
    },
    "batch_scaling": {
        "caption": "Throughput against batch size",
        "alt": (
            "Line chart of throughput in images per second against batch size on "
            "logarithmic axes, one line per model and optimization."
        ),
    },
}

#: Maps the raw optimizer ids to the vocabulary a reader of the case study expects.
DISPLAY_NAMES: dict[str, str] = {
    "fp32": "FP32 (baseline)",
    "compile": "torch.compile",
    "dynamic_int8": "Dynamic INT8",
    "static_int8": "Static INT8 (PTQ)",
    "qat_int8": "QAT INT8",
    "prune_unstructured_30": "Prune 30%",
    "prune_unstructured_50": "Prune 50%",
    "prune_unstructured_70": "Prune 70%",
    "onnxruntime": "ONNX Runtime FP32",
    "onnxruntime_int8": "ONNX Runtime INT8",
}

MODEL_DISPLAY: dict[str, str] = {
    "resnet18": "ResNet-18",
    "mobilenet_v2": "MobileNetV2",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "shufflenet_v2": "ShuffleNetV2",
    "efficientnet_b0": "EfficientNet-B0",
}

#: Counts that describe the harness itself rather than any run. Kept here so the
#: case study has one place to read them from.
IMPLEMENTATION_FACTS = {
    "architectures": 5,
    "optimizations": 10,
    "resolutions": 2,
    "batch_sizes": 3,
    "thread_counts": 2,
}


def display_optimization(optimization_id: str) -> str:
    return DISPLAY_NAMES.get(optimization_id, optimization_id.replace("_", " "))


def display_model(model_id: str) -> str:
    return MODEL_DISPLAY.get(model_id, model_id)


def optimize_image(source: Path, destination: Path, max_width: int, quality: int) -> bool:
    """Write a WebP copy of ``source``, downscaled to ``max_width``.

    Figures are rendered at 200 dpi for print; on the web that is roughly 3x more
    pixels than any display needs, and the file size difference is significant.
    """
    try:
        from PIL import Image
    except ImportError:
        # Pillow normally arrives with torchvision, but the export must not fail
        # just because the image extra is missing.
        shutil.copy2(source, destination.with_suffix(source.suffix))
        return False

    with Image.open(source) as image:
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, round(image.height * ratio)), Image.Resampling.LANCZOS)
        # Charts have flat colours and sharp edges; a palette would band the
        # gradients, so keep RGBA and let WebP do lossy compression.
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGBA")

        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="WEBP", quality=quality, method=6)
    return True


def build_figure_block(
    figures_dir: Path, images_dir: Path, max_width: int, quality: int
) -> dict[str, Any]:
    """Convert every figure referenced by the case study and describe the results."""
    figures: dict[str, Any] = {}

    for name, spec in FIGURE_SPEC.items():
        source = figures_dir / f"{name}.png"
        if not source.exists():
            # A figure that was never generated (a results directory from a
            # partial run) is reported as missing rather than silently dropped.
            figures[name] = {**spec, "available": False}
            continue

        webp = images_dir / f"{name}.webp"
        png = images_dir / f"{name}.png"
        converted = optimize_image(source, webp, max_width, quality)
        shutil.copy2(source, png)

        figures[name] = {
            **spec,
            "available": True,
            "webp": f"./images/{webp.name}",
            "png": f"./images/{png.name}",
            "converted": converted,
            "original_width": _image_width(source),
        }

    return figures


def _image_width(path: Path) -> int | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.width
    except Exception:
        return None


def read_run_config(results_dir: Path) -> dict[str, Any]:
    """Read the run metadata written by ``pipeline.run_all``.

    ``pipeline`` writes ``raw/_run__<run_id>.json`` containing the effective
    configuration. That is the authoritative record of which dataset a run used,
    which is what decides whether the results are publishable or provisional.
    """
    raw_dir = results_dir / "raw" if (results_dir / "raw").exists() else results_dir
    for path in sorted(raw_dir.glob("_run__*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _round(value: Any, digits: int = 2) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(number, digits)


def build_payload(results_dir: Path, figures: dict[str, Any]) -> dict[str, Any]:
    """Assemble the JSON the case study renders from."""
    from edgebench.results import load_store

    store = load_store(results_dir)
    frame = store.summary_frame()
    environment = store.environment()
    fingerprint = environment.get("fingerprint", {}) or {}

    run_config = read_run_config(results_dir)
    dataset_name = (
        run_config.get("config", {}).get("data", {}).get("name")
        or run_config.get("summary", {}).get("dataset")
        or "unknown"
    )
    provisional = dataset_name == "synthetic" or "offline" in results_dir.as_posix().lower()

    # ---------------------------------------------------------------- models
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in store.records:
        if record.model_id in seen:
            continue
        seen.add(record.model_id)
        card = record.model_card or {}
        models.append(
            {
                "id": record.model_id,
                "name": display_model(record.model_id),
                "params_millions": _round((card.get("parameters") or 0) / 1e6),
                "macs_millions": _round(card.get("macs_millions")),
                "family": card.get("family"),
                "adapt": card.get("adapt"),
            }
        )
    models.sort(key=lambda item: -(item["params_millions"] or 0))

    # --------------------------------------------------------------- headline
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "model_id": row["model_id"],
                "model": display_model(row["model_id"]),
                "optimization_id": row["optimization_id"],
                "optimization": display_optimization(row["optimization_id"]),
                "status": row["status"],
                "top1_pct": _round(row["top1"] * 100 if row.get("top1") is not None else None),
                "top1_delta_pp": _round(row.get("top1_delta_points")),
                "weight_mib": _round(row.get("weight_mib")),
                "size_reduction_pct": _round(row.get("size_reduction_pct")),
                "latency_ms": _round(row.get("latency_ms")),
                "speedup": _round(row.get("speedup_vs_fp32")),
                "ece": _round(row.get("ece"), 4),
                "timing": row.get("measurement_quality"),
                "unstable": bool(row.get("latency_unstable"))
                if row.get("latency_unstable") is not None
                else False,
            }
        )

    # Order: baseline first, then fastest first -- the reading order of the
    # narrative, which is "here is the reference, here is what beats it".
    def sort_key(item: dict[str, Any]) -> tuple[int, float]:
        if item["optimization_id"] == "fp32":
            return (0, 0.0)
        return (1, -(item["speedup"] or 0.0))

    rows.sort(key=sort_key)

    unavailable = [
        {
            "model": display_model(record.model_id),
            "optimization": display_optimization(record.optimization_id),
            "reason": (record.reason or "").split("\n")[0][:200],
        }
        for record in store.records
        if record.status != "applied"
    ]

    # Best measured configuration per model, for the "so what" summary.
    best_per_model: list[dict[str, Any]] = []
    for model_id in store.model_ids():
        candidates = [
            row
            for row in rows
            if row["model_id"] == model_id and row["status"] == "applied" and row["speedup"]
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda item: item["speedup"])
        best_per_model.append(
            {
                "model": best["model"],
                "optimization": best["optimization"],
                "speedup": best["speedup"],
                "size_reduction_pct": best["size_reduction_pct"],
                "top1_delta_pp": best["top1_delta_pp"],
            }
        )

    return {
        "provisional": provisional,
        "provisional_reason": (
            "These numbers were produced on synthetic random tensors to validate the "
            "harness end to end. They are not measurements of model quality and must "
            "not be cited. Run `edgebench run-all` on the real dataset to replace them."
            if provisional
            else None
        ),
        "dataset": dataset_name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "cpu": environment.get("cpu"),
            "platform": environment.get("platform"),
            "logical_cores": fingerprint.get("cpu_count_logical"),
            "torch": fingerprint.get("torch"),
            "python": fingerprint.get("python"),
            "onnxruntime": fingerprint.get("onnxruntime"),
            "git_commit": environment.get("git_commit"),
        },
        "implementation": {
            **IMPLEMENTATION_FACTS,
            "latency_cells_per_config": (
                IMPLEMENTATION_FACTS["resolutions"]
                * IMPLEMENTATION_FACTS["batch_sizes"]
                * IMPLEMENTATION_FACTS["thread_counts"]
            ),
            "tests": _count_tests(),
        },
        "headline_config": {"resolution": 32, "batch_size": 1, "num_threads": 1},
        "models": models,
        "headline": rows,
        "best_per_model": best_per_model,
        "unavailable": unavailable,
        "figures": figures,
        "counts": {
            "records": len(store.records),
            "applied": sum(1 for r in store.records if r.status == "applied"),
            "not_applied": sum(1 for r in store.records if r.status != "applied"),
        },
    }


def _count_tests() -> int | None:
    """Number of collected tests, so the case study never hardcodes it.

    Asks pytest for the real count first, because parametrisation expands a single
    ``def test_`` into several cases -- counting source lines under-reports
    (166 versus 172 at the time of writing). Network-marked tests are excluded,
    matching what CI runs on every push.

    Falls back to a source-line count if pytest is unavailable, which keeps the
    export script usable from an environment that only has the runtime deps.
    """
    tests_dir = REPO_ROOT / "tests"
    if not tests_dir.exists():
        return None

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-m",
                "not network",
                "-p",
                "no:warnings",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )

        # `-q` prints a per-file tally ("tests/test_models.py: 33") rather than a
        # single total, so both shapes are handled. Which one appears depends on
        # the pytest version, and guessing wrong would silently under-report.
        match = re.search(r"(\d+)\s+tests?\s+collected", completed.stdout)
        if match:
            return int(match.group(1))

        per_file = re.findall(
            r"^\S*tests?[/\\]\S+:\s*(\d+)\s*$", completed.stdout, flags=re.MULTILINE
        )
        if per_file:
            return sum(int(count) for count in per_file)
    except (OSError, subprocess.SubprocessError):
        pass

    total = 0
    for path in sorted(tests_dir.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        total += len(re.findall(r"^\s*def test_", text, flags=re.MULTILINE))
    return total or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="Results directory to read.")
    parser.add_argument("--out", default="portfolio", help="Portfolio directory to write into.")
    parser.add_argument("--width", type=int, default=1600, help="Max figure width in pixels.")
    parser.add_argument("--quality", type=int, default=86, help="WebP quality (1-100).")
    args = parser.parse_args()

    results_dir = (REPO_ROOT / args.results).resolve()
    portfolio_dir = (REPO_ROOT / args.out).resolve()

    if not results_dir.exists():
        print(f"error: results directory not found: {results_dir}", file=sys.stderr)
        print("Run `edgebench run-all` first, or point --results at an existing run.")
        return 1

    images_dir = portfolio_dir / "images"
    data_dir = portfolio_dir / "data"
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Figures live next to the records; a partial run may have written none.
    figures_dir = results_dir / "figures"
    if not figures_dir.exists():
        print(f"note: no figures directory at {figures_dir}")
        print("      run `edgebench report --results ...` to generate them first.")
        figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {results_dir.relative_to(REPO_ROOT)}")
    figures = build_figure_block(figures_dir, images_dir, args.width, args.quality)
    payload = build_payload(results_dir, figures)

    output_path = data_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print()
    print("Exported portfolio assets")
    print("-" * 56)
    available = sum(1 for value in figures.values() if value.get("available"))
    print(f"  figures        {available}/{len(FIGURE_SPEC)} -> {images_dir.relative_to(REPO_ROOT)}")
    print(
        f"  records        {payload['counts']['records']} ({payload['counts']['applied']} applied)"
    )
    print(f"  models         {len(payload['models'])}")
    print(f"  dataset        {payload['dataset']}")
    print(f"  data           {output_path.relative_to(REPO_ROOT)}")

    if payload["provisional"]:
        print()
        print("  !! PROVISIONAL DATA ------------------------------------------")
        print("  These results came from the synthetic offline config.")
        print("  The case study will render a visible warning banner.")
        print("  Replace with `edgebench run-all` before publishing.")
        print("  --------------------------------------------------------------")

    missing = [name for name, value in figures.items() if not value.get("available")]
    if missing:
        print()
        print(f"  figures not found: {', '.join(missing)}")
        print("  the case study omits them rather than showing broken images")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
