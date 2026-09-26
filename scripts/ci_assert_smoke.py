"""CI assertion: the synthetic smoke run produced real, complete measurements.

This is the guard that makes the smoke job meaningful. Without it, the job would
pass as long as the commands exit zero -- which they do even when a run produces
no records at all (``edgebench report`` on an empty directory logs a warning and
returns 0, by design, so a partial sweep still yields partial output).

So the assertions are about *content*, not exit codes:

- records exist for the model
- a minimum number of optimizations actually applied
- every applied record has a latency grid, accuracy, and a positive footprint
- report generation produced figures and tables
- the portfolio export produced its JSON

Run from the repository root:

    python scripts/ci_assert_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("results/offline")
RAW_DIR = RESULTS_DIR / "raw"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PORTFOLIO_JSON = Path("portfolio/data/results.json")

#: The ladder is intentionally fault tolerant, so some rungs may legitimately be
#: unavailable on a CI runner (torch.compile needs a working C++ toolchain, which
#: a bare ubuntu-latest image may or may not provide). Requiring every rung to
#: apply would make the job flaky for reasons unrelated to the harness. Requiring
#: most of them still catches a genuinely broken ladder.
MIN_RECORDS = 4
MIN_APPLIED = 3


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> int:
    if not RAW_DIR.exists():
        fail(f"no raw results directory at {RAW_DIR}")

    records = sorted(RAW_DIR.glob("resnet18__*.json"))
    print(f"records written: {len(records)}")
    if len(records) < MIN_RECORDS:
        fail(f"expected at least {MIN_RECORDS} records, found {len(records)}")

    applied = 0
    skipped: list[str] = []

    for path in records:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            fail(f"{path.name} is not valid JSON: {error}")

        status = payload.get("status")
        optimization = payload.get("optimization_id", path.stem)

        if status != "applied":
            # Not an error by itself, but it must carry a reason so the results
            # table can explain the gap rather than showing a blank row.
            reason = payload.get("reason")
            if not reason:
                fail(f"{optimization}: status={status} but no reason was recorded")
            skipped.append(f"{optimization} ({status})")
            continue

        applied += 1

        if not payload.get("latency"):
            fail(f"{optimization}: applied but the latency grid is empty")

        cells = [cell for cell in payload["latency"] if cell.get("status") == "ok"]
        if not cells:
            fail(f"{optimization}: no latency cell completed successfully")

        for cell in cells:
            if cell.get("latency_ms") is None:
                fail(f"{optimization}: a cell is marked ok but has no latency")
            if not cell.get("samples_ms"):
                fail(f"{optimization}: no raw latency samples were retained")

        if payload.get("accuracy") is None:
            fail(f"{optimization}: applied but has no accuracy block")
        if payload["accuracy"].get("num_samples") != 512:
            fail(
                f"{optimization}: expected 512 synthetic test samples, "
                f"got {payload['accuracy'].get('num_samples')}"
            )

        weight_bytes = payload.get("footprint", {}).get("weight_bytes")
        if not weight_bytes or weight_bytes <= 0:
            fail(f"{optimization}: missing or non-positive weight_bytes")

    print(f"applied: {applied}")

    if skipped:
        print(f"not applied: {', '.join(skipped)}")

    if applied < MIN_APPLIED:
        fail(f"only {applied} optimizations applied; the ladder is broken")

    figures = sorted(FIGURES_DIR.glob("*.png"))
    tables = sorted(TABLES_DIR.glob("*.csv"))
    print(f"figures: {len(figures)}")
    print(f"tables: {len(tables)}")

    if not figures:
        fail("report generation produced no figures")
    if not tables:
        fail("report generation produced no tables")

    manifest = TABLES_DIR / "manifest.json"
    if not manifest.exists():
        fail("report manifest was not written")

    if not PORTFOLIO_JSON.exists():
        fail(f"portfolio export did not write {PORTFOLIO_JSON}")

    exported = json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))
    print(f"portfolio export: {PORTFOLIO_JSON.stat().st_size} bytes")
    print(f"portfolio provisional: {exported.get('provisional')}")

    # The smoke run uses synthetic data, so the export MUST flag it as
    # provisional. If it does not, the guard that stops placeholder accuracy from
    # being presented as a measurement is broken -- which is worse than a failing
    # build.
    if exported.get("provisional") is not True:
        fail(
            "portfolio export ran on synthetic data but did not set provisional=true; "
            "the placeholder-data guard is not working"
        )

    if exported.get("dataset") != "synthetic":
        fail(f"expected dataset='synthetic', got {exported.get('dataset')!r}")

    if not exported.get("headline"):
        fail("portfolio export produced an empty headline table")

    print("smoke assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
