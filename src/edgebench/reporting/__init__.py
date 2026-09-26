"""Reporting: turn stored records into figures, tables and a report document."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edgebench.reporting import plots, tables
from edgebench.results import ResultStore, load_store
from edgebench.utils import ensure_dir, get_logger, json_dump

logger = get_logger("reporting")

__all__ = [
    "ReportArtifacts",
    "build_report",
    "environment_markdown",
    "load_store",
    "plots",
    "tables",
]


@dataclass
class ReportArtifacts:
    """Paths produced by one report build."""

    figures: list[Path] = field(default_factory=list)
    tables: dict[str, Path] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "figures": [str(path) for path in self.figures],
            "tables": {name: str(path) for name, path in self.tables.items()},
            "skipped": self.skipped,
        }


def build_report(
    results_dir: str | Path,
    figures_dir: str | Path | None = None,
    tables_dir: str | Path | None = None,
) -> ReportArtifacts:
    """Regenerate every figure and table from the records in ``results_dir``.

    Figure failures are collected rather than raised. A results file that predates
    a new column should cost one missing chart, not a failed build.
    """
    directory = Path(results_dir)
    store = load_store(directory)

    figures_path = ensure_dir(figures_dir or directory / "figures")
    tables_path = ensure_dir(tables_dir or directory / "tables")

    artifacts = ReportArtifacts()

    if not store.records:
        logger.warning("no records under %s: nothing to report", directory)
        artifacts.skipped.append("no records found")
        return artifacts

    plots.apply_style()

    summary = store.summary_frame()
    latency = store.latency_frame()

    artifacts.tables = tables.write_tables(summary, latency, tables_path)
    (tables_path / "environment.md").write_text(
        tables.environment_markdown(store), encoding="utf-8"
    )

    artifacts.figures.extend(
        _build_figure(
            "accuracy_vs_latency",
            lambda: plots.pareto_accuracy_latency(summary),
            figures_path,
            artifacts,
        )
    )
    artifacts.figures.extend(
        _build_figure(
            "size_vs_accuracy",
            lambda: plots.size_vs_accuracy(summary),
            figures_path,
            artifacts,
        )
    )
    artifacts.figures.extend(
        _build_figure(
            "resolution_sensitivity",
            lambda: plots.resolution_sensitivity(latency),
            figures_path,
            artifacts,
        )
    )
    artifacts.figures.extend(
        _build_figure(
            "batch_scaling",
            lambda: plots.latency_vs_batch_size(latency),
            figures_path,
            artifacts,
        )
    )
    artifacts.figures.extend(
        _build_figure(
            "per_class_accuracy",
            lambda: plots.per_class_accuracy(summary),
            figures_path,
            artifacts,
        )
    )

    # Per-model figures are the most useful for the portfolio and the report, but
    # they need at least one measured configuration to say anything.
    for model_id in store.model_ids():
        artifacts.figures.extend(
            _build_figure(
                plots.figure_name("optimization_ladder", model_id),
                lambda model_id=model_id: plots.optimization_ladder(summary, model_id),
                figures_path,
                artifacts,
            )
        )
        artifacts.figures.extend(
            _build_figure(
                plots.figure_name("latency_distribution", model_id),
                lambda model_id=model_id: plots.latency_distribution(latency, model_id),
                figures_path,
                artifacts,
            )
        )

    json_dump(
        {
            "results_dir": str(directory),
            "run_ids": store.run_ids,
            "models": store.model_ids(),
            "optimizations": store.optimization_ids(),
            "environment": store.environment(),
            "figures": [path.name for path in artifacts.figures],
            "tables": {name: path.name for name, path in artifacts.tables.items()},
            "skipped": artifacts.skipped,
        },
        tables_path / "manifest.json",
    )

    logger.info(
        "report built: %d figures, %d tables, %d skipped",
        len(artifacts.figures),
        len(artifacts.tables),
        len(artifacts.skipped),
    )
    return artifacts


def _build_figure(
    name: str,
    factory: Any,
    figures_path: Path,
    artifacts: ReportArtifacts,
) -> list[Path]:
    """Build and save one figure, recording failures instead of raising."""
    try:
        figure = factory()
    except Exception as error:
        artifacts.skipped.append(f"{name}: {type(error).__name__}: {error}")
        logger.warning("skipping figure %s: %s", name, error)
        return []

    try:
        return plots.save_figure(figure, figures_path / name)
    except Exception as error:
        artifacts.skipped.append(f"{name}: save failed: {error}")
        logger.warning("could not save figure %s: %s", name, error)
        return []


def environment_markdown(store: ResultStore) -> str:
    """Re-exported for the report builder."""
    return tables.environment_markdown(store)
