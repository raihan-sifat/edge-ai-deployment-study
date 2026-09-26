"""Table generation.

Everything is derived from the flattened summary frame, so a table cannot
disagree with a chart: both read the same numbers. Tables are written twice --
as CSV for reuse and as Markdown for pasting straight into the README and the
technical report.

``tabulate`` is deliberately not a dependency; the Markdown writer below is 20
lines and removes a build-time failure mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from edgebench.results import ResultStore
from edgebench.utils import ensure_dir, get_logger

logger = get_logger("reporting.tables")

#: Canonical column order and headers for the headline table.
MAIN_COLUMNS: dict[str, str] = {
    "model_id": "Model",
    "optimization_id": "Optimization",
    "parameters": "Params (M)",
    "weight_mib": "Weights (MiB)",
    "size_reduction_pct": "Size -%",
    "macs_millions": "MACs (M)",
    "top1_pct": "Top-1 (%)",
    "top1_delta_points": "d Top-1 (pp)",
    "latency_ms": "Latency p50 (ms)",
    "speedup_vs_fp32": "Speedup",
    "measurement_quality": "Timing",
    "ece": "ECE",
    "throughput_samples_per_s": "Throughput (img/s)",
}


def to_markdown(frame: pd.DataFrame, floatfmt: str = ".2f", na_rep: str = "-") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    if frame.empty:
        return "_no data_\n"

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in frame.iterrows():
        cells: list[str] = []
        for value in row:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append(na_rep)
            elif isinstance(value, float):
                cells.append(format(value, floatfmt))
            elif isinstance(value, int | bool):
                cells.append(str(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived display columns and truncate oversized values."""
    prepared = frame.copy()
    if "top1" in prepared:
        prepared["top1_pct"] = prepared["top1"] * 100.0
    if "parameters" in prepared:
        prepared["parameters"] = prepared["parameters"] / 1e6
    if "macs" in prepared and "macs_millions" not in prepared:
        prepared["macs_millions"] = prepared["macs"] / 1e6
    return prepared


def main_table(frame: pd.DataFrame, sort_by: list[str] | None = None) -> pd.DataFrame:
    """The headline comparison: one row per (model, optimization)."""
    prepared = _prepare(frame)
    if prepared.empty:
        return pd.DataFrame(columns=list(MAIN_COLUMNS.values()))

    available = [column for column in MAIN_COLUMNS if column in prepared.columns]
    subset = prepared[available].copy()

    if sort_by:
        subset = subset.sort_values(sort_by, kind="stable")
    elif "latency_ms" in subset:
        subset = subset.sort_values(["model_id", "latency_ms"], kind="stable", na_position="last")

    return subset.rename(columns={k: v for k, v in MAIN_COLUMNS.items() if k in subset.columns})


def speed_accuracy_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    """For each model, the configurations that are not dominated.

    A configuration is dominated if another one is at least as accurate *and* at
    least as fast. The survivors are the only rows worth putting in a summary:
    everything else is a strictly worse trade.
    """
    columns = [
        "Model",
        "Optimization",
        "Top-1 (%)",
        "Latency p50 (ms)",
        "Speedup",
        "Weights (MiB)",
        "Timing",
    ]
    prepared = _prepare(frame)
    valid = prepared.dropna(subset=["top1", "latency_ms"])
    if valid.empty:
        return pd.DataFrame(columns=columns)

    keepers: list[pd.Series] = []
    for _, group in valid.groupby("model_id", sort=True):
        ordered = group.sort_values("latency_ms", kind="stable")
        best_accuracy = float("-inf")
        for _, row in ordered.iterrows():
            if row["top1"] > best_accuracy:
                keepers.append(row)
                best_accuracy = float(row["top1"])

    frontier = pd.DataFrame(keepers)
    frontier = _prepare(frontier)
    columns_present = [column for column in columns if column in frontier.columns]
    return frontier[columns_present].rename_axis(None, axis=1).reset_index(drop=True)


def status_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Which optimizations ran, and the reason for any that did not."""
    if frame.empty:
        return pd.DataFrame(columns=["Model", "Optimization", "Status", "Reason"])

    subset = frame[["model_id", "optimization_id", "status", "reason"]].copy()
    subset = subset.rename(
        columns={
            "model_id": "Model",
            "optimization_id": "Optimization",
            "status": "Status",
            "reason": "Reason",
        }
    )
    if "Reason" in subset:
        subset["Reason"] = subset["Reason"].fillna("").astype(str).str.slice(0, 160)
    return subset.reset_index(drop=True)


def resolution_effect_table(latency_frame: pd.DataFrame) -> pd.DataFrame:
    """Median latency at each measured resolution, batch 1, single thread."""
    if latency_frame.empty:
        return pd.DataFrame()

    subset = latency_frame[
        (latency_frame["status"] == "ok")
        & (latency_frame["batch_size"] == 1)
        & (latency_frame["num_threads"] == 1)
    ]
    if subset.empty:
        return pd.DataFrame()

    pivot = subset.pivot_table(
        index=["model_id", "optimization_id"],
        columns="resolution",
        values="latency_ms",
        aggfunc="median",
    ).reset_index()

    # The 32 -> 224 ratio is the headline: it is 49x the pixels.
    if 32 in pivot.columns and 224 in pivot.columns:
        pivot["ratio_224_over_32"] = pivot[224] / pivot[32]

    pivot.columns = [
        " ".join(str(part) for part in column if str(part) != "")
        if isinstance(column, tuple)
        else str(column)
        for column in pivot.columns
    ]
    return pivot


def batch_scaling_table(latency_frame: pd.DataFrame) -> pd.DataFrame:
    """Throughput vs batch size, single thread, at the smallest measured resolution."""
    if latency_frame.empty:
        return pd.DataFrame()

    smallest = int(latency_frame["resolution"].min())
    subset = latency_frame[
        (latency_frame["status"] == "ok")
        & (latency_frame["resolution"] == smallest)
        & (latency_frame["num_threads"] == 1)
    ]
    if subset.empty:
        return pd.DataFrame()

    pivot = subset.pivot_table(
        index=["model_id", "optimization_id"],
        columns="batch_size",
        values="throughput_samples_per_s",
        aggfunc="median",
    ).reset_index()
    pivot.columns = [
        " ".join(str(part) for part in column if str(part) != "")
        if isinstance(column, tuple)
        else str(column)
        for column in pivot.columns
    ]
    return pivot


def write_tables(
    frame: pd.DataFrame,
    latency_frame: pd.DataFrame,
    tables_dir: str | Path,
) -> dict[str, Path]:
    """Write every table as CSV and Markdown. Returns a name -> path map."""
    destination = ensure_dir(tables_dir)
    outputs: dict[str, Path] = {}

    tables: dict[str, pd.DataFrame] = {
        "main_comparison": main_table(frame),
        "frontier": speed_accuracy_frontier(frame),
        "optimization_status": status_table(frame),
        "resolution_effect": resolution_effect_table(latency_frame),
        "batch_scaling": batch_scaling_table(latency_frame),
    }

    for name, table in tables.items():
        csv_path = destination / f"{name}.csv"
        md_path = destination / f"{name}.md"
        table.to_csv(csv_path, index=False)
        md_path.write_text(
            f"<!-- generated by edgebench; do not edit by hand -->\n\n{to_markdown(table)}\n",
            encoding="utf-8",
        )
        outputs[name] = csv_path
        logger.info("wrote %s and %s", csv_path.name, md_path.name)

    return outputs


def environment_markdown(store: ResultStore) -> str:
    """A short Markdown block describing the machine the numbers came from."""
    environment = store.environment()
    fingerprint: dict[str, Any] = environment.get("fingerprint", {}) or {}

    lines = [
        "| Property | Value |",
        "| --- | --- |",
        f"| CPU | {environment.get('cpu', 'unknown')} |",
        f"| Logical cores | {fingerprint.get('cpu_count_logical', 'unknown')} |",
        f"| PyTorch | {fingerprint.get('torch', 'unknown')} |",
        f"| Python | {fingerprint.get('python', 'unknown')} |",
        f"| Platform | {environment.get('platform', 'unknown')} |",
        f"| ONNX Runtime | {fingerprint.get('onnxruntime') or 'not installed'} |",
        f"| RAM | {_format_bytes(fingerprint.get('ram_total_bytes'))} |",
        f"| Git commit | {environment.get('git_commit') or 'unknown'} |",
    ]
    return "\n".join(lines) + "\n"


def _format_bytes(value: Any) -> str:
    if not isinstance(value, int | float) or value <= 0:
        return "unknown"
    return f"{value / (1024**3):.2f} GiB"
