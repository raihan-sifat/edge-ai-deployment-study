"""Figure generation.

Every figure is produced from stored records, never from a live model. Two
conventions are applied throughout so the set of charts reads as one document:

* **Latency axes are logarithmic.** The models span an order of magnitude, and a
  linear axis renders the fast end as an unreadable clump.
* **Accuracy is plotted in percentage points, not fractions.** "0.94 vs 0.95" hides
  the size of the difference; "94% vs 95%" does not.

The most important figure is :func:`pareto_accuracy_latency`. It is the one that
answers the question the study exists to ask: given an accuracy budget, what is
the cheapest way to spend it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; must be set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edgebench.utils import get_logger, slugify

logger = get_logger("reporting.plots")

FIGURE_DPI = 200
PALETTE = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#059669",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#be185d",  # pink
    "#4b5563",  # slate
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def apply_style() -> None:
    """Install the shared matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 10,
        }
    )


def _model_color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def save_figure(figure: plt.Figure, path: str | Path, also_svg: bool = True) -> list[Path]:
    """Save a figure as PNG (and optionally SVG) and close it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    written = [destination]
    figure.savefig(destination)
    if also_svg and destination.suffix.lower() == ".png":
        svg_path = destination.with_suffix(".svg")
        figure.savefig(svg_path)
        written.append(svg_path)

    plt.close(figure)
    logger.info("wrote %s", destination.name)
    return written


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def pareto_accuracy_latency(frame: pd.DataFrame, title_suffix: str = "") -> plt.Figure:
    """Accuracy against single-thread batch-of-one latency. The headline figure."""
    data = frame.dropna(subset=["top1", "latency_ms"]).copy()
    figure, axes = plt.subplots(figsize=(8.4, 5.6))

    if data.empty:
        axes.set_axis_off()
        axes.text(0.5, 0.5, "no measured data", ha="center", va="center")
        return figure

    models = sorted(data["model_id"].unique())
    for index, model in enumerate(models):
        subset = data[data["model_id"] == model]
        color = _model_color(index)
        axes.scatter(
            subset["latency_ms"],
            subset["top1"] * 100,
            s=70,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=model,
        )
        # Join the ladder for this model so the direction of travel is visible.
        ordered = subset.sort_values("latency_ms")
        axes.plot(
            ordered["latency_ms"],
            ordered["top1"] * 100,
            color=color,
            linewidth=1.0,
            alpha=0.35,
            zorder=2,
        )

    axes.set_xscale("log")
    axes.set_xlabel("Latency, batch 1, 1 thread (ms, log scale)")
    axes.set_ylabel("Test top-1 accuracy (%)")
    axes.set_title(f"Accuracy vs latency trade-off{title_suffix}")

    # Annotate the best point per model to keep the plot readable.
    for model in models:
        subset = data[data["model_id"] == model]
        best = subset.loc[subset["top1"].idxmax()]
        axes.annotate(
            str(best["optimization_id"]),
            (best["latency_ms"], best["top1"] * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
            color="#374151",
        )

    axes.legend(title="Model", loc="lower right", ncols=2)
    figure.tight_layout()
    return figure


def size_vs_accuracy(frame: pd.DataFrame, title_suffix: str = "") -> plt.Figure:
    """Shipped weight size against accuracy, with marker area proportional to MACs."""
    data = frame.dropna(subset=["top1", "weight_mib"]).copy()
    figure, axes = plt.subplots(figsize=(8.4, 5.6))

    if data.empty:
        axes.set_axis_off()
        axes.text(0.5, 0.5, "no measured data", ha="center", va="center")
        return figure

    # Marker area encodes static compute cost. MACs may be missing for older
    # records, in which case every marker gets the same, neutral size rather than
    # the plot failing outright.
    macs = pd.to_numeric(data["macs"], errors="coerce").fillna(0.0) if "macs" in data else None
    largest = float(macs.max()) if macs is not None and len(macs) else 0.0
    if macs is not None and largest > 0:
        scale = 12.0 + 180.0 * (macs / largest)
    else:
        scale = pd.Series(28.0, index=data.index)
        # If cost is unavailable, fall back to encoding parameter count instead.
        fallback = pd.to_numeric(data.get("parameters", pd.Series(dtype=float)), errors="coerce")
        if isinstance(fallback, pd.Series) and len(fallback) and fallback.max() > 0:
            scale = 12.0 + 180.0 * (fallback / float(fallback.max()))

    for index, model in enumerate(sorted(data["model_id"].unique())):
        subset = data[data["model_id"] == model]
        axes.scatter(
            subset["weight_mib"],
            subset["top1"] * 100,
            s=scale.loc[subset.index],
            color=_model_color(index),
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8,
            label=model,
            zorder=3,
        )

    axes.set_xscale("log")
    axes.set_xlabel("Serialized weight size (MiB, log scale)")
    axes.set_ylabel("Test top-1 accuracy (%)")
    axes.set_title(f"Model size vs accuracy{title_suffix}")
    axes.legend(title="Model  (marker area ~ MACs)", loc="lower right", ncols=2)
    figure.tight_layout()
    return figure


def optimization_ladder(
    frame: pd.DataFrame,
    model_id: str,
    title_suffix: str = "",
) -> plt.Figure:
    """Speedup and accuracy change per optimization, for one architecture.

    Two stacked panels rather than a dual-axis chart: a shared x-axis is enough to
    link them, and a second y-axis would invite misreading the crossing point.
    """
    subset = frame[frame["model_id"] == model_id].copy()
    figure, (top_axes, bottom_axes) = plt.subplots(
        2, 1, figsize=(8.4, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    if subset.empty:
        for axes in (top_axes, bottom_axes):
            axes.set_axis_off()
        top_axes.text(0.5, 0.5, "no data for this model", ha="center", va="center")
        return figure

    subset = subset.sort_values("latency_ms", kind="stable")
    labels = subset["optimization_id"].tolist()
    positions = np.arange(len(labels))

    speedups = subset["speedup_vs_fp32"].fillna(1.0).to_numpy(dtype=float)
    colors = ["#9ca3af" if label == "fp32" else "#2563eb" for label in labels]

    top_axes.bar(positions, speedups, color=colors, width=0.62, zorder=3)
    top_axes.axhline(1.0, color="#6b7280", linewidth=1.0, linestyle="--", zorder=2)
    top_axes.set_ylabel("Speedup vs FP32")
    top_axes.set_title(f"{model_id}: optimization ladder{title_suffix}")
    for position, value in zip(positions, speedups, strict=True):
        top_axes.text(position, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=8)

    deltas = subset["top1_delta_points"].fillna(0.0).to_numpy(dtype=float)
    delta_colors = ["#059669" if value >= 0 else "#dc2626" for value in deltas]
    bottom_axes.bar(positions, deltas, color=delta_colors, width=0.62, zorder=3)
    bottom_axes.axhline(0.0, color="#6b7280", linewidth=1.0, zorder=2)
    bottom_axes.set_ylabel("Change in top-1 (pp)")
    bottom_axes.set_xticks(positions)
    bottom_axes.set_xticklabels(labels, rotation=32, ha="right")
    for position, value in zip(positions, deltas, strict=True):
        bottom_axes.text(
            position,
            value,
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    figure.tight_layout()
    return figure


def latency_distribution(
    latency_frame: pd.DataFrame,
    model_id: str,
    resolution: int | None = None,
    batch_size: int = 1,
    num_threads: int | None = 1,
    max_configs: int = 7,
) -> plt.Figure:
    """Per-iteration latency distribution for each optimization of one model.

    Boxes over the pooled samples from every repeat, plus a swarm of the raw
    points. This is where an unstable configuration shows itself: a wide box, or a
    long upper tail, is a bigger practical problem than a slightly higher median.
    """
    data = latency_frame[
        (latency_frame["model_id"] == model_id)
        & (latency_frame["status"] == "ok")
        & (latency_frame["batch_size"] == batch_size)
        & (latency_frame["num_threads"] == num_threads)
    ].copy()

    figure, axes = plt.subplots(figsize=(8.4, 5.2))

    if resolution is not None:
        data = data[data["resolution"] == resolution]
    if data.empty or data["samples_ms"].map(len).sum() == 0:
        axes.set_axis_off()
        axes.text(0.5, 0.5, "no latency samples available", ha="center", va="center")
        return figure

    # Keep the configurations with the most samples, then order by median.
    data = data[data["samples_ms"].map(len) > 0]
    data = data.assign(median=data["latency_ms"]).sort_values("median")
    if len(data) > max_configs:
        data = data.head(max_configs)

    samples = [np.asarray(entry, dtype=float) for entry in data["samples_ms"]]
    labels = data["optimization_id"].tolist()

    box = axes.boxplot(
        samples,
        orientation="vertical",
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.4},
    )
    grey = "#9ca3af"
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(grey if labels[index] == "fp32" else "#93c5fd")
        patch.set_alpha(0.85)
        patch.set_edgecolor("#374151")

    rng = np.random.default_rng(0)
    for index, values in enumerate(samples, start=1):
        jitter = rng.normal(0, 0.055, size=values.size)
        axes.scatter(
            np.full(values.size, index) + jitter,
            values,
            s=3,
            color="#1f2937",
            alpha=0.25,
            zorder=3,
        )

    axes.set_xticks(range(1, len(labels) + 1))
    axes.set_xticklabels(labels, rotation=24, ha="right")
    axes.set_ylabel("Per-iteration latency (ms)")
    units = "ms" if data["resolution"].nunique() == 1 else "ms (mixed resolutions)"
    axes.set_title(
        f"{model_id}: latency distribution ({units}, batch {batch_size}, "
        f"threads {num_threads if num_threads is not None else 'auto'})"
    )
    figure.tight_layout()
    return figure


def latency_vs_batch_size(
    latency_frame: pd.DataFrame,
    resolution: int | None = None,
    num_threads: int | None = 1,
) -> plt.Figure:
    """Throughput against batch size: where each optimization stops scaling."""
    data = latency_frame[
        (latency_frame["status"] == "ok")
        & (latency_frame["num_threads"] == num_threads)
        & latency_frame["throughput_samples_per_s"].notna()
    ].copy()

    if resolution is not None:
        data = data[data["resolution"] == resolution]

    figure, axes = plt.subplots(figsize=(8.4, 5.4))
    if data.empty:
        axes.set_axis_off()
        axes.text(0.5, 0.5, "no data", ha="center", va="center")
        return figure

    models = sorted(data["model_id"].unique())
    for model_index, model in enumerate(models):
        for opt_index, optimization in enumerate(
            sorted(data[data["model_id"] == model]["optimization_id"].unique())
        ):
            subset = data[
                (data["model_id"] == model) & (data["optimization_id"] == optimization)
            ].sort_values("batch_size")
            if subset["batch_size"].nunique() < 2:
                continue
            axes.plot(
                subset["batch_size"],
                subset["throughput_samples_per_s"],
                marker=MARKERS[opt_index % len(MARKERS)],
                markersize=5,
                color=_model_color(model_index),
                alpha=0.75,
                linewidth=1.3,
                label=f"{model} / {optimization}",
            )

    axes.set_xscale("log", base=2)
    axes.set_yscale("log")
    axes.set_xlabel("Batch size")
    axes.set_ylabel("Throughput (images/s, log scale)")
    axes.set_title("Batch scaling")
    # Only draw a legend if at least one line was actually plotted; a single
    # batch size produces no lines, and matplotlib warns on an empty legend.
    if axes.get_legend_handles_labels()[1]:
        axes.legend(fontsize=7, ncols=2, loc="upper left")
    figure.tight_layout()
    return figure


def resolution_sensitivity(
    latency_frame: pd.DataFrame,
    batch_size: int = 1,
    num_threads: int | None = 1,
) -> plt.Figure:
    """Latency at each measured resolution, grouped by model.

    The point of the figure is that 224x224 inputs are 49x the pixels of 32x32 but
    do not cost 49x the time; the ratio is what a deployment actually plans for.
    """
    data = latency_frame[
        (latency_frame["status"] == "ok")
        & (latency_frame["batch_size"] == batch_size)
        & (latency_frame["num_threads"] == num_threads)
        & (latency_frame["optimization_id"] == "fp32")
    ].copy()

    figure, axes = plt.subplots(figsize=(8.4, 5.0))
    if data.empty:
        axes.set_axis_off()
        axes.text(0.5, 0.5, "no data", ha="center", va="center")
        return figure

    resolutions = sorted(data["resolution"].unique())
    models = sorted(data["model_id"].unique())
    width = 0.8 / max(1, len(resolutions))

    for index, resolution in enumerate(resolutions):
        subset = data[data["resolution"] == resolution].set_index("model_id")
        values = [float(subset["latency_ms"].get(model, np.nan)) for model in models]
        positions = np.arange(len(models)) + index * width
        bars = axes.bar(
            positions,
            values,
            width=width,
            color=PALETTE[index % len(PALETTE)],
            label=f"{resolution}x{resolution}",
            zorder=3,
        )
        axes.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    axes.set_xticks(np.arange(len(models)) + width * (len(resolutions) - 1) / 2)
    axes.set_xticklabels(models, rotation=18, ha="right")
    axes.set_yscale("log")
    axes.set_ylabel("FP32 latency (ms, log scale)")
    axes.set_title(f"Resolution sensitivity (batch {batch_size}, threads {num_threads})")
    axes.legend(title="Input size")
    figure.tight_layout()
    return figure


def per_class_accuracy(frame: pd.DataFrame, num_classes: int = 10) -> plt.Figure:
    """Which classes quantization hurts, for the configurations that hurt most.

    Aggregate top-1 hides the failure mode that matters at deployment: a model can
    hold its overall accuracy while collapsing on one or two classes, which is
    usually unacceptable for a real application. This figure ranks every
    non-baseline configuration by how much accuracy it destroys on its worst
    classes, then shows the full per-class delta profile.
    """
    figure, axes = plt.subplots(figsize=(9.0, 5.4))

    if "per_class_accuracy" not in frame.columns:
        axes.set_axis_off()
        axes.text(
            0.5,
            0.5,
            "per-class accuracy requires accuracy details in the records",
            ha="center",
            va="center",
        )
        return figure

    rows: list[tuple[str, list[float], float]] = []
    for model_id, group in frame.groupby("model_id", sort=True):
        baseline = group[group["optimization_id"] == "fp32"]
        if baseline.empty:
            continue
        baseline_profile = _as_float_list(baseline.iloc[0].get("per_class_accuracy"), num_classes)
        if baseline_profile is None:
            continue

        for _, candidate in group[group["optimization_id"] != "fp32"].iterrows():
            profile = _as_float_list(candidate.get("per_class_accuracy"), num_classes)
            if profile is None:
                continue
            delta = [
                (profile[index] - baseline_profile[index]) * 100.0 for index in range(num_classes)
            ]
            # Rank by total accuracy destroyed, counting only classes that got worse.
            severity = abs(sum(value for value in delta if value < 0))
            rows.append((f"{model_id}\n/ {candidate['optimization_id']}", delta, severity))

    if not rows:
        axes.set_axis_off()
        axes.text(0.5, 0.5, "no per-class data available", ha="center", va="center")
        return figure

    rows.sort(key=lambda item: -item[2])
    rows = rows[:10]
    labels = [item[0] for item in rows]
    matrix = np.asarray([item[1] for item in rows], dtype=float)

    limit = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
    image = axes.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axes.set_xticks(range(num_classes))
    axes.set_xticklabels([str(index) for index in range(num_classes)])
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels, fontsize=7.5)
    axes.set_xlabel("CIFAR-10 class index")
    axes.set_title("Per-class accuracy change vs FP32 baseline (percentage points)")
    axes.grid(False)
    figure.colorbar(image, ax=axes, label="pp change", fraction=0.03, pad=0.02)
    figure.tight_layout()
    return figure


def _as_float_list(value: Any, expected_length: int) -> list[float] | None:
    """Coerce a stored per-class profile to a fixed-length float list."""
    if not isinstance(value, list) or len(value) != expected_length:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def figure_name(prefix: str, *parts: Any) -> str:
    """Deterministic filename for a figure."""
    suffix = "-".join(slugify(str(part)) for part in parts if part)
    return f"{prefix}__{suffix}.png" if suffix else f"{prefix}.png"
