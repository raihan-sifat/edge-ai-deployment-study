# Methodology

This document states exactly what the benchmark does, why each choice was made, and
where the results can mislead. It is written to be read before trusting any number in
`results/`.

- [1. Research question](#1-research-question)
- [2. Scope](#2-scope)
- [3. Data protocol](#3-data-protocol)
- [4. Training protocol](#4-training-protocol)
- [5. The optimization ladder](#5-the-optimization-ladder)
- [6. Measurement protocol](#6-measurement-protocol)
- [7. Metrics and their definitions](#7-metrics-and-their-definitions)
- [8. Threats to validity](#8-threats-to-validity)
- [9. Reproducibility](#9-reproducibility)
- [10. What this study does not claim](#10-what-this-study-does-not-claim)

---

## 1. Research question

> Given a fixed accuracy budget on a CPU-only edge target, which deployment
> optimization actually delivers, by how much, and what does it cost in accuracy?

The question is deliberately comparative rather than absolute. Published speedups are
typically quoted against a different baseline, at a different batch size, on different
hardware, making them incomparable. Here every optimization is measured against the
same architecture's FP32 model, on the same machine, under one protocol, in one run.

A secondary question motivates the negative results:

> Which widely repeated optimization recommendations fail to hold, and why?

That question turns out to be more informative than the first, and it is why the
results tables carry an explicit `status` column.

## 2. Scope

**In scope**

| Dimension     | Choice                                                                                |
| ------------- | ------------------------------------------------------------------------------------- |
| Task          | CIFAR-10 image classification (10 classes, 32×32 RGB)                                 |
| Device        | CPU only, x86-64 and ARM-64                                                           |
| Architectures | ResNet-18, MobileNetV2, MobileNetV3-Small, ShuffleNetV2-x1.0, EfficientNet-B0         |
| Optimizations | Compilation, dynamic/static/QAT INT8 quantization, unstructured pruning, ONNX Runtime |
| Resolution    | 32×32 and 224×224                                                                     |
| Precision     | FP32 and INT8                                                                         |

**Deliberately out of scope**

| Excluded                     | Why                                                                                                                                                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pretrained weights           | Every model trains from scratch under one recipe, so a difference is attributable to the architecture, not to ImageNet transfer. Comparing a pretrained model against a from-scratch one measures the pretraining, not the optimization. |
| GPUs and NPUs                | CUDA and TensorRT change the ranking entirely (kernels, memory bandwidth, precision support). Mixing them into a CPU study produces numbers that describe neither.                                                                       |
| Structured pruning           | Correct channel pruning requires dependency-aware surgery across residual adds and concatenations. An incorrect implementation would produce a wrong result, which is worse than an absent one.                                          |
| CIFAR-100 / ImageNet         | Runtime cost. The harness supports the schema; the study targets CIFAR-10.                                                                                                                                                               |
| Wall-plug energy measurement | Requires external hardware. Reported as `not measured` where no counter exists.                                                                                                                                                          |

## 3. Data protocol

CIFAR-10 provides 50,000 training images and 10,000 test images.

### The split

The 10,000-image test set is **held out**. It is never used for model selection, never
used to tune a hyperparameter, and never used to calibrate a quantizer.

```
50,000 train images
├── 45,000  training
└──  5,000  validation   <- model selection, early stopping, calibration
                            (carved out with a seeded permutation)

10,000 test images        <- used exactly once per configuration,
                            after all selection is finished
```

The 5,000-image validation carve-out is drawn from the _train_ split with a fixed
permutation (`numpy.random.default_rng(seed)`), so it is identical across processes
and machines.

This matters more than it might appear. A common pattern in optimization benchmarks is
to calibrate a quantizer on the test set, then report test accuracy. That leaks test
information into the model and inflates the score — and it inflates _quantized_ scores
more than FP32 ones, because quantization is where the calibration influence is
strongest. The protocol here removes that path entirely: **quantizer calibration draws
from the validation split**, and the test split is measured once.

### Preprocessing

```
Train:  RandomCrop(32, padding=4) → RandomHorizontalFlip(p=0.5) → ToTensor → Normalize
Eval:   ToTensor → Normalize
```

Normalization constants are the standard CIFAR-10 channel statistics:
`mean = (0.4914, 0.4822, 0.4465)`, `std = (0.2470, 0.2435, 0.2616)`.

Augmentation is intentionally mild. Stronger recipes (CutMix, RandAugment, AutoAugment)
raise absolute accuracy but also add run-to-run variance, which makes it harder to
attribute a difference to the optimization under test. Since the study's claims are
_relative_ — this optimization versus that one — variance is more costly than a lower
absolute ceiling would be.

## 4. Training protocol

### Architecture adaptation

Every network here was designed for 224×224 ImageNet inputs and opens with an
aggressive downsample: a 7×7 stride-2 convolution, or a 3×3 stride-2 convolution
followed by a stride-2 max-pool. Applied to 32×32 CIFAR images, that stem destroys
spatial resolution the network never recovers — the 32×32 map is 8×8 after the stem and
1×1 by the deepest stage.

The standard fix is applied: replace the stem with a single **stride-1 3×3
convolution**, and replace the first **max-pool with identity**. The input stays 32×32
through the stem, and the first real downsample moves into the first stage.

Two properties of this choice matter:

1. It is applied **before training**, so the accuracy, parameter count, MACs and latency
   reported for a model all describe the same object. Adapting a model after measuring
   it would produce a table whose rows describe different networks.
2. It changes the macro-architecture not at all. The residual blocks, inverted
   bottlenecks, squeeze-and-excitation, and channel shuffles — the things being
   compared — are untouched.

Verified by test: an adapted ResNet-18 emits a 32×32 feature map after `conv1`, and a
model built with `adapt: none` retains its stride-2 7×7 stem.

### Recipe

One recipe, identical for all five architectures.

| Setting           | Value                                                   |
| ----------------- | ------------------------------------------------------- |
| Optimizer         | SGD, momentum 0.9, Nesterov                             |
| Learning rate     | 0.1                                                     |
| Weight decay      | 5e-4, **excluding** biases and normalization parameters |
| Schedule          | 1-epoch linear warmup, then cosine decay to 1% of peak  |
| Batch size        | 128                                                     |
| Label smoothing   | 0.1                                                     |
| Gradient clipping | max-norm 1.0                                            |
| Epochs            | 30                                                      |
| Seed              | 1234                                                    |

Excluding biases and normalization parameters from weight decay is the single most
common deviation between published CIFAR recipes. Applying decay to them measurably
hurts small models on small datasets.

Model selection is best-on-validation. The checkpoint restored at the end of training
is the one with the highest validation top-1, not the last epoch.

### A note on the optimizer comparison

Holding the recipe fixed across architectures is a deliberate trade. EfficientNet-B0
would very likely prefer a different learning rate or optimizer than ResNet-18;
MobileNetV3's hard-swish activations interact with weight decay differently than
ResNet's ReLU. Tuning each architecture separately would raise every absolute accuracy
in the tables.

It would also destroy the study's comparability. If each model has its own recipe, a
difference between two rows has two possible explanations, and the experiment can no
longer distinguish them. Since the research question is about _optimizations_ rather
than _architectures_, recipe uniformity is worth more than peak accuracy. This is a
stated limitation, not an oversight.

## 5. The optimization ladder

Each rung is applied to an independent deep copy of the trained FP32 model. No
optimization can observe or contaminate another, which makes the ladder
**order-independent** — verified by test.

### `fp32` — reference

Eager PyTorch, no graph rewrite, no compression. Produced by the same code path as
every other rung rather than special-cased in the reporting layer, so the baseline
cannot drift from the thing it is baselining.

### `compile` — `torch.compile`

Inductor backend. Two things are recorded that a server-oriented benchmark would omit:

- **Cold-start cost.** Compilation happens on the first call. A model that is 30%
  faster in steady state but takes 20 seconds to compile is often _worse_ for an
  appliance that boots frequently, so compile wall time is recorded in metadata and
  excluded from the latency figures but reported separately.
- **Unavailability.** On Windows without the MSVC build tools, or without Triton,
  `torch.compile` cannot produce a working backend. This is recorded as `unavailable`
  with the exception text, not as a failure and not silently.

### `dynamic_int8` — dynamic quantization

Weights quantized ahead of time; activations quantized per batch at runtime. No
calibration data required.

Implemented with an explicit check on whether the installed PyTorch's dynamic mappings
actually include `Conv2d`. On builds where they do not, `quantize_dynamic` silently
leaves every convolution in FP32 — turning the result into a statement about classifier
heads rather than about models. The harness detects this and records it in the outcome's
notes, because the resulting measurement is still valid but means something narrower
than its name suggests.

### `static_int8` — post-training static quantization

FX graph-mode quantization with activation ranges computed from a calibration pass.

- **Calibration data:** 512 images, drawn from the validation split, in batches of 32.
- **Engine selection:** probed against `torch.backends.quantized.supported_engines`
  rather than assumed. Recent PyTorch builds replace `fbgemm` with `onednn` as the
  engine _name_ while still accepting `x86`/`fbgemm` as _qconfig_ names — two different
  namespaces that must be resolved separately. Getting this wrong silently produces a
  result that says "unsupported" when the backend is in fact available.
- **QConfigMapping:** a mapping is used rather than a bare qconfig, because `prepare_fx`
  special-cases fixed-qparams operators (sigmoid, softmax, and the hard-swish path
  MobileNetV3 depends on) only when given a mapping.
- FX graph mode performs conv+BN+ReLU fusion automatically, so the measured model is
  fused as well as quantized. This is stated because it is a real part of the speedup
  and would otherwise be attributed to quantization alone.

### `qat_int8` — quantization-aware training

Observers are inserted, then the model is fine-tuned for 3 epochs at learning rate 1e-3,
then converted to INT8.

This is the only rung that consumes training compute at optimization time. That cost is
excluded from the latency measurement (it is not inference) but is reported in metadata,
because for a deployment it is a real one-time cost that must be weighed against the
inference gain.

### `prune_unstructured_50` — global magnitude pruning

The smallest-magnitude weights, ranked globally across all prunable layers, are zeroed
to 50%; masks are then made permanent with `prune.remove`.

Four claims are tested rather than assumed:

1. **Sparsity does not shrink the shipped file.** `torch.save` writes dense tensors;
   zeroing half the weights changes values, not shape. The measured serialized size is
   therefore ~unchanged.
2. **The sparse estimate does not beat dense storage until past 50% sparsity.** Storing a
   float32 value with an int32 column index costs 8 bytes per surviving weight against 4
   bytes dense, so the break-even is _exactly_ 50%, and row pointers tip it marginally
   the wrong way there. This is the arithmetic reason unstructured pruning is not a
   compression technique as usually applied.
3. **Dense CPU kernels do not exploit sparsity.** MKL-DNN and oneDNN dense convolution
   paths have no sparsity-aware fast path for arbitrary masks. No speedup is claimed.
   Measured empirically at **1.01x** once measurement-order effects are controlled for
   (see §8).
4. **Zeroing weights still changes accuracy.** Sparsity is not free even when it is not
   cheap.

The theoretical sparse footprint is reported as a **CSR lower bound** and explicitly
labelled an estimate, never a measurement.

### `onnxruntime` / `onnxruntime_int8` — a different runtime

This rung changes the runtime rather than the model, which is why it is worth measuring
separately: in production, an INT8 PyTorch graph and an INT8 ONNX graph are not the same
artifact and do not have the same latency.

- Export uses the TorchScript-based exporter (`dynamo=False`) because the FX-quantized
  and dynamically-quantized modules are not supported by the `torch.export`-based
  exporter.
- The batch axis is dynamic, because edge serving is batch-of-one with occasional
  bursts; a fixed batch size would make the exported graph useless for the batch sweep.
- `onnxruntime_int8` applies **ONNX Runtime's own** dynamic quantization, a different
  implementation from PyTorch's. Comparing the two is one of the more useful cross-checks
  the study provides: the same nominal "INT8" produces different accuracy and different
  size depending on who did the quantizing.
- Equivalence is verified by test: exported logits match the source model within
  `atol=1e-4`.

## 6. Measurement protocol

The protocol is fixed once and applied to every cell.

### Step 1 — pin threads before anything else

Thread count is set before measurement begins. On a CPU it moves latency by more than
most of the optimizations in the ladder, so it is an explicit axis rather than an
ambient condition. Inter-op threads stay at 1: a single forward pass is inherently
sequential, and extra inter-op threads only add scheduling noise.

### Step 2 — warm up and discard

Warmup iterations are discarded. PyTorch's allocators, oneDNN primitive caches, and
`torch.compile` all do one-time work on the first calls.

### Step 3 — time every iteration individually

Each iteration is timed individually and **all samples are retained**. This is what makes
the latency-distribution figure possible, and what allows a skewed distribution to be
seen rather than averaged into a misleading mean.

### Step 4 — time a block as well

A separate pass runs `timed_iters` forwards inside a single timer and divides. This
matters at the fast end: per-iteration timing pays timer overhead and prevents any
pipelining, so the block figure is the throughput bound. Below roughly 0.1 ms per
inference the per-iteration timer itself becomes a significant fraction of the
measurement — which the 32×32 models in this study reach. Both figures are therefore
reported, and neither alone would be honest.

### Step 5 — interleave repeats

Independent repeats are performed, and the distribution across repeats is reported.

### Inputs

Benchmark inputs are uniform in `[-1, 1]` rather than Gaussian. Random normal inputs
occasionally produce denormal activations, and denormal handling on x86 can add tens of
percent to latency for reasons that have nothing to do with the model. Inputs are seeded
so two runs see byte-identical data.

### Memory

Peak RSS is sampled on a background thread every 0.5 ms during inference and reported as
a **delta** against a baseline taken immediately before the run. Limitations, stated
because they affect interpretation:

- RSS is process-wide, so the figure includes Python, PyTorch and allocator overhead.
  The delta removes most of the constant.
- It is _sampled_, not exact. CPU PyTorch offers no peak-allocation hook. The sampling
  interval is recorded in every measurement so the precision is auditable.

Weight bytes are measured by serializing `state_dict()` to a memory buffer. A caveat:
`torch.save` writes a zip container with a constant ~2-3 KB of framing. For these
architectures (tens of MB) that is below the measurement's precision; for a model with
only a few hundred parameters it dominates entirely.

### Energy

Measured only via a real hardware counter — Intel/AMD RAPL through
`/sys/class/powercap` on Linux. RAPL reports the whole CPU package, not this process, so
it is meaningful only on a quiet machine.

On Windows and macOS no unprivileged per-package counter exists. The harness reports
`energy_available: false` with a reason and **refuses to estimate**. Deriving a number
from CPU utilisation or TDP would produce a figure that looks like a measurement and
is not one.

## 7. Metrics and their definitions

| Metric                     | Definition                                                                                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `top1`                     | Fraction of test images whose highest logit is correct                                                                                                            |
| `top5`                     | Fraction whose correct label is among the top 5 logits                                                                                                            |
| `ece`                      | Expected calibration error, 15 bins, top-label confidence                                                                                                         |
| `latency_ms`               | **Median (p50)** of per-iteration samples. Median, not mean, because CPU latency is right-skewed and the mean is dragged by the tail a deployment will never see. |
| `p90/p95/p99_ms`           | Tail latencies — what an interactive application actually experiences                                                                                             |
| `cv`                       | Coefficient of variation across repeat medians. Above 0.15 the cell is flagged `unstable` in the tables.                                                          |
| `throughput_samples_per_s` | `1000 x batch_size / p50_ms`                                                                                                                                      |
| `weight_bytes`             | Serialized `state_dict()` size — what ships to the device                                                                                                         |
| `macs`                     | Multiply-accumulates for one forward pass at batch 1                                                                                                              |
| `speedup_vs_fp32`          | `baseline_p50 / candidate_p50`, **per architecture**                                                                                                              |
| `top1_delta_points`        | Accuracy change in percentage points, per architecture                                                                                                            |
| `rss_peak_delta_bytes`     | Peak RSS growth during inference                                                                                                                                  |

### Two conventions that are fixed globally

**MACs, not FLOPs.** One MAC is one multiply-accumulate, and **FLOPs = 2 × MACs**.
Published papers mix these freely, with the difference frequently being 2x. Both are
written to the results; the convention is stated in the output itself via
`macs_note`.

**Deltas are per-architecture.** `speedup_vs_fp32` and `top1_delta_points` are computed
against the _same model's_ FP32 row, not against a global baseline. The interesting
question is "what does this optimization cost this architecture", not "which
architecture is fastest" — the latter is already visible in the `fp32` rows.

## 8. Threats to validity

### 8.1 Measurement order — the dominant error source

**This was discovered empirically during development and is the single largest source of
error in the harness.**

A full sequential run reported 50%-pruned ResNet-18 at **74.4 ms** against FP32 at
**17.9 ms** — 4.2x slower. This is implausible: magnitude pruning leaves dense tensors,
so dense kernels should be unaffected.

`scripts/check_measurement_order.py` re-measures the same two models, interleaved, in
opposite orders:

```
order A: fp32 (18.14 ms) then prune50 (18.17 ms)
order B: prune50 (17.95 ms) then fp32 (17.24 ms)
order C: fp32 (18.07 ms) then prune50 (19.00 ms)

prune50 / fp32 median ratio: 1.01x
between-configuration effect: 1.01x
within-configuration spread : 5.8% of median
```

The true effect is **1.01x**. The 4.2x gap was drift in machine state over the ~5
minutes of sequential measurement — thermal throttling on a laptop CPU, plus
background load.

**Consequences:**

- The harness records `cv` for every cell, and tables flag `cv > 0.15` as `unstable`.
- Configurations should be **interleaved**, not measured as sequential blocks, when
  precise rank ordering matters.
- Absolute latency claims from a single sequential run on a shared machine should be
  treated as unreliable. This is stated rather than hidden because a benchmark that
  reports a 4.2x effect that does not exist is worse than one that reports nothing.
- The current implementation measures each `(model, optimization)` record as a block.
  Fully interleaving across the whole ladder is the correct next step and is tracked as
  future work.

### 8.2 Thermal and background load

Laptop CPUs throttle. The study targets batch-1 single-thread execution partly because
it is the least thermally demanding configuration and therefore the most stable. Results
from a machine running other work are not comparable to results from a quiet one.

### 8.3 Absolute accuracy

The published suite trains for 10 epochs at learning rate 0.1 rather than the 30 in
`configs/default.yaml`. `configs/default.yaml` remains the reference recipe; 10 is what a
hosted CPU runner can afford, since a single ResNet-18 epoch measures at roughly 16
minutes and the full five-architecture suite at 30 epochs is about 20 hours of CPU time.
The consequence is stated plainly: 10 epochs is a reasonable budget for CIFAR-10 but is
not state-of-the-art, and longer schedules and stronger augmentation would raise every
number. This affects absolute accuracy, not the relative differences the study is about —
with the caveat that quantization _sensitivity_ does depend somewhat on the accuracy of
the starting model.

### 8.4 Architecture-specific quantization sensitivity

ResNet-18 quantizes gracefully; MobileNetV3's hard-swish and EfficientNet's swish
activations are harder to quantize, because unbounded activations have long tails that
INT8 ranges represent poorly. Expect architecture-dependent differences in the
accuracy cost. This is a finding, not a defect.

### 8.5 Operator coverage varies by PyTorch build

`torch.ao.quantization` support differs across versions and platforms. Which
optimizations apply is a property of the installed stack. `edgebench info` reports this
before a run; every record stores the exact versions used.

### 8.6 MAC counting is approximate

`torch.utils.flop_counter` reports the operations it recognizes. Unusual operators may
be missed, and the count excludes some memory-bound work entirely. MACs are reported as
`None` rather than guessed when counting fails.

## 9. Reproducibility

### What a run records

Every JSON record embeds:

- `run_id`, ISO-8601 `created_at`, `schema_version`, `edgebench_version`
- CPU model string, logical core count, platform string
- Full `environment_fingerprint`: PyTorch, NumPy, ONNX Runtime and its providers,
  Python version, total RAM, default thread counts
- Git commit hash (absent outside a checkout)
- `model_card`: parameter breakdown, MACs, adaptation mode, architecture notes
- The full effective configuration

A record is therefore self-describing: given a `results/` directory, the machine,
software stack and configuration that produced it can be reconstructed without the
original environment.

### Determinism

- Seeds are set for Python, NumPy and PyTorch from a single `seed` value.
- Validation and calibration subsets use seeded permutations and are identical across
  machines.
- Benchmark inputs are seeded.
- `train.deterministic: true` enables `torch.use_deterministic_algorithms`. It is off by
  default because it forces some CPU convolutions onto slow paths, which would corrupt
  the latency measurements it was meant to stabilize.

Latency is not bit-reproducible on a multitasking OS and is not claimed to be. The
protocol restricts variance; it cannot eliminate it.

### Regenerating a report

```bash
edgebench report --results results
```

Reporting reads only stored JSON. `results/raw/` is sufficient to reproduce every figure
and table on any machine, with no model, no dataset and no GPU.

## 10. What this study does not claim

- **Not a survey.** Five architectures and six optimization families on one dataset
  cannot establish general laws. Results are claims about these models on this task
  under this protocol.
- **Not a ranking of architectures.** The recipe is uniform by design, so a given
  architecture may be under-tuned.
- **Not hardware-portable.** All measured numbers are CPU-specific. The _methodology_ is
  portable; the numbers are not. An ARM device will rank things differently, particularly
  `dynamic_int8` and `qnnpack`-backed quantization.
- **Not a compression study.** Model size is measured as serialized bytes, deliberately,
  because that is what ships. Serialization format overhead is a real effect on file size
  and is treated as such rather than factored out.
- **Not a publication.** This is a working project whose value is the implementation,
  the measurements and the negative results. Results here have not been peer reviewed.

---

## Appendix A: file-by-file rationale

| Module             | Non-obvious responsibility                                                                                                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`        | Rejects unknown keys. A typo in a config file surfaces immediately instead of silently running with a default.                                                                                  |
| `data.py`          | Enforces the validation/test separation; provides a synthetic dataset so CI can exercise the pipeline offline.                                                                                  |
| `training.py`      | Custom `LearningRateSchedule` ABC. Wrapping `torch.optim.lr_scheduler` is a trap: `StepLR.step()` returns `None`, and mixing per-step and per-epoch stepping silently decays at the wrong rate. |
| `inference.py`     | One interface over PyTorch, compiled, quantized and ONNX Runtime models, so the benchmark never branches on backend.                                                                            |
| `models/adapt.py`  | CIFAR stem adaptation; fails loudly rather than silently leaving an untrained head.                                                                                                             |
| `optim/base.py`    | `is_fatal()` distinguishes `KeyboardInterrupt`/`SystemExit`/`MemoryError` from ordinary failures, so Ctrl-C is not recorded as a failed optimization.                                           |
| `bench/energy.py`  | Refuses to estimate. The clearest statement of the project's stance on fabricated metrics.                                                                                                      |
| `bench/latency.py` | Both per-iteration and block timing, because neither alone is honest across the 0.06 ms to 200 ms range this study spans.                                                                       |
| `results.py`       | Flags unstable cells; computes per-architecture deltas.                                                                                                                                         |
| `reporting/`       | Fails soft: one malformed column costs one missing chart, not a broken build.                                                                                                                   |

## Appendix B: reproducing the headline negative results

```bash
# 1. Confirm what your machine supports before investing hours
edgebench info

# 2. Validate the whole pipeline offline in minutes
edgebench run-all --config configs/offline.yaml

# 3. Check measurement stability on your own hardware
python scripts/check_measurement_order.py

# 4. Run the real study
edgebench run-all
```
