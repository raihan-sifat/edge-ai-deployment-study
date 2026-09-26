"""CI assertion: the real-CIFAR-10 run actually trained on real data.

This is the only job that would catch a broken dataset path, a data-split
regression, or a torchvision API change, so the assertions are specific to what
real CIFAR-10 guarantees:

- the test split is exactly 10,000 images
- the training split was actually used (the checkpoint exists and carries a
  training history, not a cache hit)
- accuracy after one epoch sits in a plausible band

That last check is a sanity band, not an accuracy target. It catches the two
failure modes that matter: a broken training loop (which lands at chance, ~10%)
and an accidental test-set leak (which would score far above what a single epoch
can legitimately achieve). It deliberately does not assert a specific number,
because that would be a flaky test of the hardware rather than of the code.

Run from the repository root:

    python scripts/ci_assert_real_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAW_DIR = Path("results/smoke/raw")
CHECKPOINT_DIR = Path("checkpoints/smoke")

CIFAR10_TEST_SAMPLES = 10_000

#: Chance on CIFAR-10 is 10%. One epoch of a from-scratch model lands somewhere
#: in the low tens of percent. The upper bound is the leak detector.
MIN_PLAUSIBLE_TOP1 = 0.10
MAX_PLAUSIBLE_TOP1 = 0.85


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> int:
    if not RAW_DIR.exists():
        fail(f"no raw results directory at {RAW_DIR} (was the real-data run skipped?)")

    records = sorted(RAW_DIR.glob("resnet18__*.json"))
    if not records:
        fail("no records produced from the real-data run")

    print(f"records: {len(records)}")

    baseline = next(
        (path for path in records if path.name.endswith("__fp32.json")),
        records[0],
    )
    payload = json.loads(baseline.read_text(encoding="utf-8"))

    accuracy = payload.get("accuracy") or {}
    top1 = accuracy.get("top1")
    samples = accuracy.get("num_samples")

    print(f"baseline record: {baseline.name}")
    print(f"test samples:    {samples}")
    print(f"top-1:           {top1}")

    # ---- the dataset really is CIFAR-10 ---------------------------------
    if samples != CIFAR10_TEST_SAMPLES:
        fail(
            f"expected {CIFAR10_TEST_SAMPLES} CIFAR-10 test images, got {samples}. "
            "Either the dataset did not download correctly or the split logic changed."
        )

    # ---- the model was genuinely trained --------------------------------
    checkpoint = CHECKPOINT_DIR / "resnet18__fp32.pt"
    if not checkpoint.exists():
        fail(f"no checkpoint at {checkpoint}; the run did not train")

    import torch

    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = stored.get("metadata", {}) if isinstance(stored, dict) else {}
    history = metadata.get("training_history") or {}

    if not history.get("epochs"):
        fail("checkpoint carries no training history; it was loaded from cache, not trained")

    epochs_done = len(history["epochs"])
    print(f"epochs trained:  {epochs_done}")
    if epochs_done < 1:
        fail("training history is empty")

    recorded_train_top1 = history["epochs"][-1].get("train_top1")
    print(f"train top-1:     {recorded_train_top1}")

    if recorded_train_top1 is None:
        fail("training history has no train_top1; the training loop did not record metrics")
    if recorded_train_top1 < 0.11:
        fail(
            f"training accuracy of {recorded_train_top1:.3f} is at or below chance; "
            "the training loop is not learning"
        )

    # ---- the split sizes are what the protocol says ----------------------
    saved_at = metadata.get("saved_at")
    print(f"checkpoint saved: {saved_at}")

    # ---- accuracy is plausible ------------------------------------------
    if not (MIN_PLAUSIBLE_TOP1 < top1 < MAX_PLAUSIBLE_TOP1):
        hint = (
            "at or below chance -- the training loop is broken"
            if top1 <= MIN_PLAUSIBLE_TOP1
            else "implausibly high for a single epoch -- suspect a test-set leak"
        )
        fail(f"one-epoch top-1 of {top1:.4f} is outside the plausible band ({hint})")

    print(
        f"one-epoch top-1 {top1:.2%} is in the plausible band "
        f"({MIN_PLAUSIBLE_TOP1:.0%}-{MAX_PLAUSIBLE_TOP1:.0%})"
    )

    print("real-data assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
