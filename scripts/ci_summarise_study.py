"""Write a Markdown summary of a study run to stdout, for GitHub's job summary.

The workflow pipes this into ``$GITHUB_STEP_SUMMARY``, which means the headline
table and the list of anything that did not apply are readable directly on the
Actions run page -- without downloading the artifacts first. That matters because
the first question after a two-hour run is "did it work and what did it find",
and a raw artifact zip answers neither.

Output is plain Markdown. Nothing here recomputes a measurement; it formats what
the results files already contain.

Usage::

    python scripts/ci_summarise_study.py                     # reads results/study
    python scripts/ci_summarise_study.py --results results   # or any other run
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

#: Written as an escape rather than a literal: Ruff flags the literal as ambiguous
#: with the letter x (RUF001), and this is unambiguously a multiplication sign.
MULTIPLICATION_SIGN = "\u00d7"


def force_utf8_output() -> None:
    """Emit UTF-8 regardless of the platform's default console encoding.

    This is a correctness requirement, not a nicety: ``$GITHUB_STEP_SUMMARY`` must
    be UTF-8, and the table uses characters outside the Windows console's default
    codepage (``Δ``, notably, is absent from cp1252). Without this the script
    raises ``UnicodeEncodeError`` when run locally on Windows, even though it
    would have worked in CI.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):  # pragma: no cover
                reconfigure(encoding="utf-8")


force_utf8_output()


def load_records(raw_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.startswith("_"):  # run metadata, not a measurement
            continue
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records


def fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/study")
    args = parser.parse_args()

    results_dir = Path(args.results)
    raw_dir = results_dir / "raw"

    print("## Study summary")
    print()

    if not raw_dir.exists():
        print(f"> No results found at `{raw_dir}`. The run did not get far enough to write any.")
        return 0

    records = load_records(raw_dir)
    if not records:
        print(f"> `{raw_dir}` exists but contains no measurement records.")
        return 0

    applied = [r for r in records if r.get("status") == "applied"]
    not_applied = [r for r in records if r.get("status") != "applied"]

    # Environment, so the numbers are interpretable without the artifacts.
    environment = (records[0].get("environment") or {}).get("fingerprint", {}) or {}
    cpu = (records[0].get("environment") or {}).get("cpu", "unknown")
    print(f"**Machine:** {cpu}  ")
    print(
        f"**Stack:** PyTorch {environment.get('torch', '?')}, "
        f"Python {environment.get('python', '?')}, "
        f"ONNX Runtime {environment.get('onnxruntime') or 'not installed'}  "
    )
    print(f"**Records:** {len(applied)} applied, {len(not_applied)} not applied")
    print()

    if not applied:
        print("> Nothing applied. The ladder did not run.")
        return 0

    # ---- headline table ---------------------------------------------------
    # Latency is single-thread batch-1 at the smallest measured resolution,
    # matching the definition used in the README and the report.
    def headline_cell(record: dict) -> tuple[float | None, dict | None]:
        cells = [c for c in record.get("latency", []) if c.get("status") == "ok"]
        if not cells:
            return None, None
        smallest = min(c["resolution"] for c in cells)
        preferred = [
            c
            for c in cells
            if c["resolution"] == smallest and c["batch_size"] == 1 and c["num_threads"] == 1
        ]
        cell = preferred[0] if preferred else min(cells, key=lambda c: c.get("latency_ms") or 1e9)
        return cell.get("latency_ms"), cell

    rows: list[dict] = []
    for record in applied:
        accuracy = record.get("accuracy") or {}
        latency_ms, cell = headline_cell(record)
        footprint = record.get("footprint") or {}
        per_iteration = (cell or {}).get("per_iteration") or {}

        rows.append(
            {
                "model": record.get("model_id", "?"),
                "optimization": record.get("optimization_id", "?"),
                "top1": accuracy.get("top1"),
                "weight_mib": (footprint.get("weight_bytes") or 0) / (1024 * 1024) or None,
                "latency_ms": latency_ms,
                "cv": per_iteration.get("cv"),
            }
        )

    print("| Model | Optimization | Top-1 | Weights | Latency p50 | Timing |")
    print("| --- | --- | ---: | ---: | ---: | --- |")

    for row in rows:
        cv = row["cv"]
        timing = "—" if cv is None else ("unstable" if cv > 0.15 else "stable")
        print(
            f"| {row['model']} | `{row['optimization']}` "
            f"| {fmt(row['top1'] * 100 if row['top1'] is not None else None, 2, '%')} "
            f"| {fmt(row['weight_mib'], 1, ' MiB')} "
            f"| {fmt(row['latency_ms'], 2, ' ms')} "
            f"| {timing} |"
        )

    print()

    # ---- per-model best ---------------------------------------------------
    # "Highest accuracy" is not useful here: the baseline almost always wins it,
    # which says nothing a reader did not already know. What is actionable is the
    # fastest configuration that gives up at most one accuracy point, so that is
    # what this reports, with the loss stated explicitly.
    ACCURACY_TOLERANCE_PP = 1.0
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    print(f"### Fastest configuration within {ACCURACY_TOLERANCE_PP:.0f} pp of the model's best")
    print()
    print("| Model | Optimization | Top-1 | Δ vs best | Latency | Speedup |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")

    for model, model_rows in sorted(by_model.items()):
        usable = [row for row in model_rows if row["top1"] is not None and row["latency_ms"]]
        if not usable:
            continue

        best_accuracy = max(row["top1"] for row in usable)
        within = [
            row for row in usable if (best_accuracy - row["top1"]) * 100.0 <= ACCURACY_TOLERANCE_PP
        ]
        fastest = min(within, key=lambda row: row["latency_ms"])

        baseline = next((row for row in usable if row["optimization"] == "fp32"), None)
        speedup = (
            baseline["latency_ms"] / fastest["latency_ms"]
            if baseline and fastest["latency_ms"]
            else None
        )

        loss_pp = (fastest["top1"] - best_accuracy) * 100.0
        print(
            f"| {model} | `{fastest['optimization']}` "
            f"| {fmt(fastest['top1'] * 100, 2, '%')} "
            f"| {fmt(loss_pp, 2, ' pp')} "
            f"| {fmt(fastest['latency_ms'], 2, ' ms')} "
            f"| {fmt(speedup, 2, MULTIPLICATION_SIGN)} |"
        )
    print()

    # ---- anything that did not run ----------------------------------------
    if not_applied:
        print("### Not applied")
        print()
        print("Recorded with a reason rather than omitted, so the results matrix is complete.")
        print()
        print("| Model | Optimization | Status | Reason |")
        print("| --- | --- | --- | --- |")
        for record in sorted(
            not_applied, key=lambda r: (r.get("model_id", ""), r.get("optimization_id", ""))
        ):
            reason = (record.get("reason") or "").split("\n")[0][:160].replace("|", "\\|")
            print(
                f"| {record.get('model_id', '?')} | `{record.get('optimization_id', '?')}` "
                f"| {record.get('status')} | {reason} |"
            )
        print()

    unstable_count = sum(1 for row in rows if (row["cv"] or 0) > 0.15)
    if unstable_count:
        print(
            f"> **{unstable_count} of {len(rows)} rows carry unstable latency "
            f"(cv > 15%).** Expected on a shared runner: accuracy is unaffected, "
            "but treat these latency values as indicative. A quiet local run is "
            "the reference."
        )
        print()

    print(
        "<sub>Generated by `scripts/ci_summarise_study.py` from the stored records. "
        "No measurement is recomputed here.</sub>"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
