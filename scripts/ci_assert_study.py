"""CI assertion: the real-data study produced genuine, publishable results.

The distinguishing job of this script is checking that the **provisional guard did
not fire**. For a synthetic run the guard firing is the correct behaviour, and
``ci_assert_smoke.py`` asserts exactly that. For a real run the opposite is true:
if the export still marks these results provisional, either the run silently fell
back to synthetic data or the detection logic is broken, and in both cases the
artifacts must not be published.

That makes this the one check that stands between a synthetic run and a public
claim about real accuracy, which is the specific failure this project exists to
prevent.

Run from the repository root:

    python scripts/ci_assert_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("results/study")
RAW_DIR = RESULTS_DIR / "raw"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
PORTFOLIO_JSON = Path("portfolio/data/results.json")

CIFAR10_TEST_SAMPLES = 10_000

#: Chance on CIFAR-10 is 10%. Any real training run must clear it comfortably.
MIN_PLAUSIBLE_TOP1 = 0.15
#: Above this, suspect a test-set leak rather than a good model. Even a 30-epoch
#: from-scratch ResNet-18 does not exceed this on CIFAR-10 without tricks that
#: this harness does not implement.
MAX_PLAUSIBLE_TOP1 = 0.97


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> int:
    if not RAW_DIR.exists():
        fail(f"no raw results at {RAW_DIR}; did the training step run?")

    # `_run__<id>.json` holds run metadata and matches `*__*.json` too, so files
    # beginning with an underscore are filtered out. Treating that file as a
    # measurement makes every field look missing -- the first version of this
    # script failed with "None: not applied but no reason recorded" for exactly
    # that reason.
    records = [path for path in sorted(RAW_DIR.glob("*__*.json")) if not path.name.startswith("_")]
    if not records:
        fail("no benchmark records were written")

    print(f"records: {len(records)}")

    applied: list[dict] = []
    for path in records:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            fail(f"{path.name} is not valid JSON: {error}")

        if payload.get("status") == "applied":
            applied.append(payload)
        else:
            # Unavailable rungs must carry a reason, or the results table shows an
            # unexplained gap.
            if not payload.get("reason"):
                fail(f"{payload.get('optimization_id')}: not applied but no reason recorded")

    print(f"applied: {len(applied)}")
    if not applied:
        fail("no optimization applied; the ladder is broken")

    # ---- the dataset really is the real one ------------------------------
    sample_counts: set[int] = set()
    accuracies: list[tuple[str, str, float]] = []

    for payload in applied:
        accuracy = payload.get("accuracy")
        if accuracy is None:
            fail(f"{payload['optimization_id']}: applied but has no accuracy block")

        samples = accuracy.get("num_samples")
        sample_counts.add(samples)

        # CIFAR-10's test split is exactly 10,000 images. The synthetic path uses
        # 512, so a wrong count means the wrong dataset loaded.
        if samples != CIFAR10_TEST_SAMPLES:
            fail(
                f"{payload['optimization_id']}: expected {CIFAR10_TEST_SAMPLES} test "
                f"samples from the real CIFAR-10 split, got {samples}. This is the "
                "synthetic dataset's signature, or the split logic changed."
            )

        top1 = accuracy.get("top1")
        if top1 is None:
            fail(f"{payload['optimization_id']}: no top1 accuracy")
        accuracies.append((payload.get("model_id", "?"), payload.get("optimization_id", "?"), top1))

        if not payload.get("latency"):
            fail(f"{payload['optimization_id']}: applied but the latency grid is empty")

        cells = [c for c in payload["latency"] if c.get("status") == "ok"]
        if not cells:
            fail(f"{payload['optimization_id']}: no latency cell completed")

    print(f"test-sample counts seen: {sorted(sample_counts)}")

    # ---- accuracies are plausible ----------------------------------------
    best = max(accuracies, key=lambda item: item[2])
    worst = min(accuracies, key=lambda item: item[2])

    print(f"best  top-1: {best[2]:.2%}  ({best[0]} / {best[1]})")
    print(f"worst top-1: {worst[2]:.2%}  ({worst[0]} / {worst[1]})")

    if worst[2] <= MIN_PLAUSIBLE_TOP1:
        fail(
            f"{worst[0]}/{worst[1]} scored {worst[2]:.2%}, at or below chance "
            f"({MIN_PLAUSIBLE_TOP1:.0%}); the training loop is not learning"
        )
    if best[2] >= MAX_PLAUSIBLE_TOP1:
        fail(
            f"{best[0]}/{best[1]} scored {best[2]:.2%}, implausibly high for a "
            "from-scratch run; suspect a test-set leak"
        )

    # ---- artifacts exist -------------------------------------------------
    figures = sorted(FIGURES_DIR.glob("*.png"))
    tables = sorted(TABLES_DIR.glob("*.csv"))
    print(f"figures: {len(figures)}, tables: {len(tables)}")
    if not figures:
        fail("report generation produced no figures")
    if not tables:
        fail("report generation produced no tables")

    # ---- the provisional guard is the critical check ---------------------
    if not PORTFOLIO_JSON.exists():
        fail(f"portfolio export did not write {PORTFOLIO_JSON}")

    exported = json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))
    dataset = exported.get("dataset")

    print(f"export dataset: {dataset}")
    print(f"export provisional: {exported.get('provisional')}")

    if dataset != "cifar10":
        fail(
            f"portfolio export reports dataset={dataset!r}. Publishing figures "
            "labelled with the wrong dataset would be worse than publishing nothing."
        )

    if exported.get("provisional"):
        fail(
            "the export marked these results provisional, but this workflow ran on "
            "real data. Either the run fell back to synthetic data, or the detection "
            f"logic in export_portfolio_assets.py is misfiring (reason given: "
            f"{exported.get('provisional_reason')!r})"
        )

    if exported.get("provisional_reason") is not None:
        fail("provisional_reason is set even though provisional is false; inconsistent export")

    rows = [row for row in exported.get("headline", []) if row.get("status") == "applied"]
    if not rows:
        fail("the exported headline table has no applied configurations")

    unstable = [row for row in rows if row.get("unstable")]
    print(f"latency cells flagged unstable: {len(unstable)}/{len(rows)}")
    if unstable:
        # Not a failure: on a shared runner some dispersion is expected, and the
        # flag exists to surface it. Reported so the CI log states it plainly.
        print("  (expected on a shared runner; accuracy is unaffected)")

    print()
    print("study assertions passed: real CIFAR-10 results, provisional guard correctly silent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
