"""Command-line interface.

    edgebench info                          what this machine can measure
    edgebench train  --config ...           train and checkpoint the models
    edgebench run-all --config ...          train (or load), optimize, measure
    edgebench report --results results      regenerate figures and tables

Every command accepts ``--set section.key=value`` overrides so that a single
config file can be reused without editing::

    edgebench run-all --set train.epochs=5 --set benchmark.replays=1
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from edgebench import __version__
from edgebench.config import ConfigError, load_config, parse_set_arguments
from edgebench.utils import configure_logging, describe_cpu, ensure_dir, get_logger

app = typer.Typer(
    name="edgebench",
    help="Benchmark efficient deep learning models for edge AI deployment.",
    no_args_is_help=True,
    add_completion=False,
)

logger = get_logger("cli")

CONFIG_OPTION = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to the run configuration YAML."),
]
MODELS_FILE_OPTION = Annotated[
    Path | None,
    typer.Option(
        "--models-file", help="Path to the model zoo YAML (defaults to a sibling models.yaml)."
    ),
]
SET_OPTION = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        help="Override a config value, e.g. --set train.epochs=5. Repeatable.",
    ),
]
MODEL_FILTER_OPTION = Annotated[
    str | None,
    typer.Option("--models", help="Comma-separated model ids to run (default: all enabled)."),
]
OPTIMIZATION_FILTER_OPTION = Annotated[
    str | None,
    typer.Option(
        "--optimizations",
        help="Comma-separated optimization ids to run (default: all enabled).",
    ),
]
VERBOSE_OPTION = Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")]


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _load(
    config: Path,
    models_file: Path | None,
    set_values: list[str] | None,
) -> Any:
    overrides = parse_set_arguments(set_values)
    return load_config(config, models_path=models_file, overrides=overrides)


def _fail(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"edgebench {__version__}")


@app.command()
def info(
    config: Annotated[Path, typer.Option("--config", "-c", help="Config file to inspect.")] = Path(
        "configs/default.yaml"
    ),
    models_file: MODELS_FILE_OPTION = None,
    set_values: SET_OPTION = None,
) -> None:
    """Show the environment and what this machine can actually measure.

    Run this first on a new machine. It reports which optimizations will apply and
    which will be recorded as unavailable, so a surprising results table is
    explained before it is produced rather than after.
    """
    configure_logging()
    from edgebench.bench.energy import create_energy_meter
    from edgebench.optim import LADDER_IDS
    from edgebench.optim.onnx_runtime import onnx_available
    from edgebench.optim.quantization import select_quantization_backend
    from edgebench.utils import environment_fingerprint

    try:
        loaded = _load(config, models_file, set_values)
    except ConfigError as error:
        _fail(str(error))
        return

    fingerprint = environment_fingerprint()
    meter = create_energy_meter()
    onnx_ok, onnx_reason = onnx_available()

    typer.echo("")
    typer.secho("  Edge AI Deployment Study", bold=True)
    typer.echo("  " + "-" * 58)
    typer.echo(f"  CPU                 {describe_cpu()}")
    typer.echo(f"  Logical cores       {fingerprint.get('cpu_count_logical')}")
    typer.echo(f"  PyTorch             {fingerprint.get('torch')}")
    typer.echo(f"  Python              {fingerprint.get('python')}")
    typer.echo(f"  Platform            {fingerprint.get('platform')}")
    typer.echo(f"  Quantization engine {select_quantization_backend()}")
    typer.echo(
        "  ONNX Runtime        "
        + (str(fingerprint.get("onnxruntime")) if onnx_ok else f"unavailable ({onnx_reason})")
    )
    typer.echo(
        "  Energy metering     "
        + (meter.source if meter.available else f"unavailable ({meter.reason})")
    )
    typer.echo("")
    typer.echo(f"  Dataset             {loaded.data.name} ({loaded.data.num_classes} classes)")
    typer.echo(f"  Input size          {loaded.data.input_size}x{loaded.data.input_size}")
    typer.echo(f"  Resolutions         {list(loaded.benchmark.resolutions)}")
    typer.echo(f"  Batch sizes         {list(loaded.benchmark.batch_sizes)}")
    typer.echo(f"  Thread counts       {list(loaded.benchmark.thread_counts)}")
    typer.echo("")
    typer.secho("  Models", bold=True)
    for entry_model in loaded.models:
        marker = "x" if entry_model.enabled else " "
        typer.echo(f"    [{marker}] {entry_model.id}")
    typer.echo("")
    typer.secho("  Optimization ladder", bold=True)
    for entry_optimization in loaded.optimizations:
        marker = "x" if entry_optimization.enabled else " "
        typer.echo(f"    [{marker}] {entry_optimization.id}")
    typer.echo("")
    typer.echo(f"  Registered ids: {', '.join(LADDER_IDS)}")
    typer.echo("")


@app.command()
def train(
    config: CONFIG_OPTION = Path("configs/default.yaml"),
    models_file: MODELS_FILE_OPTION = None,
    set_values: SET_OPTION = None,
    models: MODEL_FILTER_OPTION = None,
    download: Annotated[
        bool, typer.Option("--download/--no-download", help="Allow downloading the dataset.")
    ] = True,
    retrain: Annotated[
        bool, typer.Option("--retrain", help="Ignore cached checkpoints and train again.")
    ] = False,
    verbose: VERBOSE_OPTION = False,
) -> None:
    """Train and checkpoint the selected models."""
    configure_logging(verbose)

    from edgebench.pipeline import train_only

    try:
        loaded = _load(config, models_file, set_values)
    except ConfigError as error:
        _fail(str(error))
        return

    ensure_dir(loaded.checkpoints_dir)
    try:
        results = train_only(
            loaded,
            model_ids=_split(models),
            download=download,
            retrain=retrain,
        )
    except Exception as error:
        _fail(f"{type(error).__name__}: {error}")
        return

    typer.echo("")
    typer.secho("  Training summary", bold=True)
    for model_id, info in results.items():
        if "error" in info:
            typer.secho(f"    {model_id:<22} FAILED  {info['error']}", fg=typer.colors.RED)
            continue
        top1 = info.get("test_top1") or 0.0
        source = info.get("source", "?")
        typer.echo(f"    {model_id:<22} test top-1 {100 * top1:6.2f}%  ({source})")
    typer.echo("")


@app.command("run-all")
def run_all_command(
    config: CONFIG_OPTION = Path("configs/default.yaml"),
    models_file: MODELS_FILE_OPTION = None,
    set_values: SET_OPTION = None,
    models: MODEL_FILTER_OPTION = None,
    optimizations: OPTIMIZATION_FILTER_OPTION = None,
    download: Annotated[
        bool, typer.Option("--download/--no-download", help="Allow downloading the dataset.")
    ] = True,
    retrain: Annotated[
        bool, typer.Option("--retrain", help="Ignore cached checkpoints and train again.")
    ] = False,
    memory: Annotated[
        bool,
        typer.Option(
            "--memory/--no-memory",
            help="Sample peak RSS during inference (slower, more information).",
        ),
    ] = True,
    report: Annotated[
        bool,
        typer.Option(
            "--report/--no-report", help="Build figures and tables when the run finishes."
        ),
    ] = True,
    verbose: VERBOSE_OPTION = False,
) -> None:
    """Train or load each model, apply the optimization ladder, and measure everything."""
    configure_logging(verbose)

    from edgebench.pipeline import run_all
    from edgebench.reporting import build_report

    try:
        loaded = _load(config, models_file, set_values)
    except ConfigError as error:
        _fail(str(error))
        return

    try:
        summary = run_all(
            loaded,
            model_ids=_split(models),
            optimization_ids=_split(optimizations),
            download=download,
            retrain=retrain,
            measure_memory=memory,
        )
    except Exception as error:
        _fail(f"{type(error).__name__}: {error}")
        return

    typer.echo("")
    typer.secho(f"  Run {summary.run_id} complete in {summary.elapsed_s:.1f}s", bold=True)
    typer.echo(f"    measured      {summary.applied}")
    typer.echo(f"    not measured  {summary.skipped}")
    typer.echo(f"    results       {loaded.results_dir}")

    if report:
        artifacts = build_report(loaded.results_dir)
        typer.echo(f"    figures       {len(artifacts.figures)}")
        typer.echo(f"    tables        {len(artifacts.tables)}")
        if artifacts.skipped:
            typer.secho(f"    skipped: {', '.join(artifacts.skipped)}", fg=typer.colors.YELLOW)
    typer.echo("")


@app.command()
def report(
    results: Annotated[
        Path, typer.Option("--results", "-r", help="Results directory to read records from.")
    ] = Path("results"),
    figures: Annotated[
        Path | None, typer.Option("--figures", help="Where to write figures.")
    ] = None,
    tables: Annotated[Path | None, typer.Option("--tables", help="Where to write tables.")] = None,
    verbose: VERBOSE_OPTION = False,
) -> None:
    """Regenerate figures and tables from stored records."""
    configure_logging(verbose)

    from edgebench.reporting import build_report

    if not results.exists():
        _fail(f"results directory not found: {results}")
        return

    artifacts = build_report(results, figures_dir=figures, tables_dir=tables)

    typer.echo("")
    typer.secho(f"  Report written from {results}", bold=True)
    for path in artifacts.figures:
        typer.echo(f"    figure  {path.name}")
    for name, path in artifacts.tables.items():
        typer.echo(f"    table   {path.name}  ({name})")
    if artifacts.skipped:
        typer.secho("  Skipped:", fg=typer.colors.YELLOW)
        for entry in artifacts.skipped:
            typer.echo(f"    - {entry}")
    typer.echo("")


if __name__ == "__main__":  # pragma: no cover
    app()
