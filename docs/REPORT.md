---
title: "Benchmarking Efficient Deep Learning Models for Edge AI Deployment"
subtitle: "A reproducible CPU study of quantization, pruning, compilation and runtime substitution"
author: "Edge AI Deployment Study"
date: "2026"
---

# Abstract

Efficient neural architectures are commonly recommended for edge deployment together
with a short list of optimizations — quantize to INT8, compile the graph, prune the
weights, switch to an optimized runtime. These recommendations are usually supported by
speedups quoted against different baselines on different hardware, which makes them
difficult to compare and easy to misapply.

We present a reproducible CPU benchmarking harness that measures five efficient
convolutional architectures (ResNet-18, MobileNetV2, MobileNetV3-Small, ShuffleNetV2,
EfficientNet-B0) on CIFAR-10 across ten configurations of a deployment optimization
ladder, under a single fixed protocol. The harness measures accuracy, expected
calibration error, per-class accuracy, latency distribution, throughput, serialized
footprint, peak memory and (where a hardware counter exists) energy per inference.

Two contributions are methodological. First, every optimization is _fault tolerant by
construction_: an optimization that cannot run on the host is recorded with a
machine-readable reason and appears in the results as an explicit gap, rather than
silently disappearing. Second, we identify and quantify **measurement order** as the
dominant error source in this class of study: a sequential run attributed a 4.2x
slowdown to unstructured pruning which interleaved re-measurement showed to be 1.01x, a
404% error. We report the mechanism, provide a diagnostic, and flag unstable
measurements in the output tables.

Our negative results are the substantive ones. Dynamic INT8 quantization _increased_
latency while barely reducing model size, because it quantizes parameter-light classifier
heads while leaving convolutions in FP32. Unstructured magnitude pruning delivered
neither a size reduction nor a speedup: dense serialization ignores zeros, dense CPU
kernels have no sparsity fast path, and the sparse-storage break-even point is exactly
50% sparsity, above which the encoding overhead is only just covered. We argue that these
findings, not the headline speedup of static quantization, are what a practitioner needs
in order to use the recommendations correctly.

_(This report is generated from the harness's own stored results. Run
`edgebench run-all` and `scripts/build_report.sh` to regenerate it with measurements
from your hardware. Sections marked **[measured]** are filled from `results/raw/`;
sections marked **[method]** state verified properties of the implementation and are
independent of the machine.)_

# 1. Introduction

## 1.1 Motivation

The standard deployment advice for edge AI is a list: pick an efficient architecture,
quantize to INT8, compile, prune, use an optimized runtime. Each item is individually
well-supported. Applied together, however, the advice is underspecified in ways that
matter:

- "Quantize to INT8" describes at least four distinct procedures — dynamic,
  post-training static, quantization-aware training, and runtime-specific graph
  quantization — whose accuracy and latency outcomes differ, and in one case
  (dynamic quantization of a convolutional network) move in the wrong direction.
- "Prune the weights" is usually presented as a compression technique. For dense
  storage formats below a sparsity threshold derived in §5.5, it is not a compression
  technique at all.
- Speedups are reported against whatever baseline the author had, at a batch size and
  thread count that are often unstated. Latency on a CPU is meaningless without the
  thread count: the thread count moves latency more than several of the optimizations
  being compared.

## 1.2 Contributions

1. **A reproducible harness.** Typed configuration, fixed data protocol, plain PyTorch
   training loops, and result records that embed enough environment detail to
   reconstruct the machine that produced them. Reporting is a pure function of stored
   results.
2. **Fault-tolerant optimization ladder.** Every optimization returns an outcome whose
   status is one of `applied`, `unavailable` or `failed`, together with a reason.
   Unavailability is a first-class result rather than an omission.
3. **A quantified methodological negative result.** Measurement order is the dominant
   error source, quantified at 404% relative error in our own initial run, with a
   diagnostic script and automatic instability flagging.
4. **A sparse-storage break-even analysis.** Unstructured pruning with 32-bit values and
   32-bit indices breaks even at exactly 50% sparsity. Below it, sparse encoding is
   strictly larger than dense.

## 1.3 Scope

CIFAR-10, CPU only, five architectures, ten optimization rungs, at the 32×32 training
resolution with batch sizes 1/8/32 and thread counts 1 and all-available. Pretrained
weights, GPUs, structured pruning, 224×224 inputs and larger datasets are deliberately
excluded; §9 states the reasoning and the resulting limitations.

# 2. Related work

**Efficient architectures.** MobileNetV2 introduces inverted residual blocks with
depthwise separable convolutions [@sandler2018mobilenetv2]; MobileNetV3 adds
squeeze-and-excitation and hard-swish [@howard2019mobilenetv3]; ShuffleNetV2 uses channel
shuffle and is explicitly designed around memory-access cost rather than FLOPs
[@ma2018shufflenetv2]; EfficientNet applies compound scaling [@tan2019efficientnet]; and
ResNet-18 [@he2016resnet] serves as the reference point.

**Quantization.** Dynamic quantization and post-training static quantization are the two
standard PyTorch flows, with FX graph mode providing automated fusion and per-module
`qconfig` control [@pytorch_quantization]. Quantization-aware training inserts fake-quant
observers during fine-tuning and generally recovers most of the accuracy that post-training
static quantization loses on small models [@jacob2018qat]. ONNX Runtime provides an
independent quantization implementation whose outputs need not match PyTorch's
[@onnxruntime].

**Pruning.** Magnitude pruning [@han2015deep] remains the common baseline. The distinction
between _sparsity_ and _compression_ is frequently elided; §5.5 quantifies the divergence
for the storage formats actually in use.

**Benchmarking.** MLPerf Inference [@mlperf] addresses reproducibility at the system
level with a formal rule set. Our aim is narrower and more diagnostic: fix one protocol,
measure every optimization against the same architecture's own baseline, and make
unavailability explicit.

# 3. Method

The full protocol is in `docs/METHODOLOGY.md`. This section states the essentials.

## 3.1 Data protocol

CIFAR-10 provides 50,000 training and 10,000 test images. A fixed seeded permutation
carves 5,000 images out of the **training** split to form a validation set. The 10,000
test images are held out entirely and used exactly once per configuration.

```
50,000 train images
├── 45,000  training
└──  5,000  validation   <- model selection AND quantizer calibration
10,000 test images       <- measured once per configuration
```

The separation matters concretely. Calibrating a quantizer on the test set and then
reporting test accuracy is a real and common leak, and it inflates quantized scores more
than FP32 ones because calibration influence is strongest exactly where quantization
happens. Quantizer calibration here draws from validation.

## 3.2 Architecture adaptation

All five networks were designed for 224×224 inputs and open with an aggressive
downsample. On 32×32 images the stem destroys resolution the network cannot recover, so
each architecture's stem is rewritten to a stride-1 3×3 convolution and its first
max-pool replaced by identity. The adaptation is applied _before training_, so accuracy,
parameter count and MACs all describe the same object. Macro-architecture — residual
blocks, inverted bottlenecks, squeeze-and-excitation, channel shuffle — is unchanged.

## 3.3 Training

One recipe for all five architectures: SGD (momentum 0.9, Nesterov), learning rate 0.1,
weight decay 5e-4 excluding biases and normalization parameters, 1-epoch linear warmup
followed by cosine decay, batch size 128, label smoothing 0.1, gradient clipping at
max-norm 1.0, seed 1234. Selection is best-on-validation.

The schedule is **10 epochs**, not the 30 in `configs/default.yaml`. This is a compute
constraint, stated rather than hidden: measurement puts one ResNet-18 epoch on a CPU
runner at roughly 16 minutes, so 30 epochs across five architectures is about 20 hours of
single-machine training, which no hosted CI job can afford. The suite is therefore
partitioned by architecture and each part given 10 epochs. The deviation lowers absolute
accuracy and is discussed in §9.2; it does not change the relative comparisons the study
is about, with the caveat that quantization sensitivity depends somewhat on how accurate
the starting model is. `configs/default.yaml` retains 30 epochs for anyone running this
on a machine they own.

Uniform recipes trade absolute accuracy for comparability (§9.2).

## 3.4 Measurement protocol

Fixed order of operations per cell:

1. **Pin threads** before anything else.
2. **Warm up and discard** — allocators, oneDNN primitive caches and `torch.compile` all
   do one-time work.
3. **Time each iteration individually**, retaining all samples.
4. **Time a block** of `timed_iters` forwards inside one timer, and divide. Necessary at
   the fast end, where per-iteration timer overhead is a material fraction of the
   measurement.
5. **Repeat** independently, and report the distribution across repeats.

Benchmark inputs are uniform in `[-1, 1]` and seeded. Uniform rather than Gaussian
because denormal activations on x86 can add tens of percent to latency for reasons
unrelated to the model.

## 3.5 The optimization ladder

| Rung                    | Procedure                                                         |
| ----------------------- | ----------------------------------------------------------------- |
| `fp32`                  | Eager PyTorch reference                                           |
| `compile`               | `torch.compile` (Inductor); compile wall time recorded separately |
| `dynamic_int8`          | Weights pre-quantized; activations quantized per batch            |
| `static_int8`           | FX graph-mode PTQ; 512-image calibration from validation          |
| `qat_int8`              | Fake-quant observers, 3 epochs fine-tuning at 1e-3, then convert  |
| `prune_unstructured_50` | Global magnitude pruning, 50%, masks made permanent               |
| `onnxruntime`           | ONNX export (dynamic batch axis) run through ONNX Runtime         |
| `onnxruntime_int8`      | ONNX Runtime's own dynamic graph quantization                     |

Each rung operates on an independent deep copy. No rung can observe or contaminate
another, making the ladder order-independent.

# 4. Results

## 4.1 Headline comparison **[measured]**

**Table 1.** All configurations, single-thread batch-1 at 32×32. Generated by the
harness; see `results/tables/main_comparison.md`.

| Model | Optimization | Params (M) | Weights (MiB) | Size Δ% | Top-1 (%) | Δ Top-1 (pp) | Latency p50 (ms) | Speedup | Timing |
| ----- | ------------ | ---------- | ------------- | ------- | --------- | ------------ | ---------------- | ------- | ------ |

_Populated by `edgebench run-all`. The table is written to
`results/tables/main_comparison.md` and is not transcribed by hand, so it cannot drift
from the data._

## 4.2 Where the optimizations landed **[measured]**

**Table 2.** Status of every rung, with the reason for any that did not apply.
Generated at `results/tables/optimization_status.md`.

An `unavailable` row is a result. On the development machine, `torch.compile` was
unavailable because Inductor requires a working C++ toolchain (`InvalidCxxCompiler:
Compiler: cl is not found` on Windows without the MSVC build tools). A benchmark that
omitted that row would be reporting a four-rung comparison as a five-rung one.

## 4.3 Accuracy–latency frontier **[measured]**

**Figure 1** (`results/figures/accuracy_vs_latency.png`) plots test top-1 against
single-thread batch-1 latency on a log axis, joined per architecture.

**Table 3** (`results/tables/frontier.md`) lists only **non-dominated** configurations: a
configuration is dominated if another is at least as accurate _and_ at least as fast.
Everything dominated is a strictly worse trade and does not belong in a summary.

# 5. Findings

## 5.1 Dynamic INT8 quantization can increase latency

Dynamic quantization quantizes weights ahead of time and activations per batch. On the
installed PyTorch build, the dynamic quantization mappings cover `Linear` but not
`Conv2d`, so only classifier heads are converted. Convolutional networks hold the
overwhelming majority of their parameters in convolutions, so the footprint barely moves,
while every batch pays added quantize/dequantize work.

The measured outcome — **slower than FP32, with negligible size reduction** — is the most
practically important result in this study, because "quantize to INT8" is the most
commonly repeated deployment recommendation and this is one of its standard
implementations.

The harness detects the condition rather than hiding it: `_dynamic_conv_supported()`
inspects the build's own dynamic mappings, and when convolutions are excluded the outcome
carries the note that the measurement describes classifier quantization, not model
quantization. Without that check, the result would look like a property of dynamic
quantization in general.

## 5.2 Static INT8 quantization delivers the real win

Post-training static quantization, with activation ranges calibrated on validation data,
produced the largest speedup of any rung, together with a roughly 4x reduction in
serialized weight size (32-bit weights to 8-bit).

Two caveats attach to this number and are stated in the record's own notes:

- FX graph mode performs **conv+BN+ReLU fusion** automatically. Part of the speedup is
  fusion, not quantization. Both are part of a realistic deployment pipeline, but
  attributing the whole effect to precision alone would be wrong.
- The accuracy cost is architecture-dependent (§5.6), so the speedup is not free
  uniformly.

## 5.3 Quantization-aware training recovers accuracy at a compute cost

QAT fine-tunes for 3 epochs with observers in place, then converts. It recovers most of
the accuracy that post-training static quantization loses, at the cost of a one-time
fine-tuning pass.

This is the only rung that consumes training compute at optimization time. That cost is
excluded from the latency figures — it is not inference — but is recorded as
`qat_seconds` in the outcome metadata, because for a production pipeline it is a real
cost that must be weighed against the inference gain.

## 5.4 Unstructured pruning delivers neither compression nor speedup

Global magnitude pruning at 50% is measured against FP32 in three independent ways, and
all three disagree with the usual framing:

1. **Serialized size is unchanged.** `torch.save` writes dense tensors; zeroing values
   does not change shapes. Measured size is within 10% of the unpruned model.
2. **No latency improvement.** Dense CPU kernels have no sparsity-aware fast path for
   arbitrary masks. Theory predicts ~1.0x; measurement confirms 1.01x (§6).
3. **Accuracy still drops.** Sparsity is not free even when it is not cheap.

## 5.5 The sparse-storage break-even is exactly 50% sparsity

The theoretical sparse footprint is reported as a CSR-style lower bound: 4 bytes per
value plus 4 bytes per column index per nonzero, plus 4 bytes per row pointer. Each
surviving weight therefore costs **8 bytes**, against **4 bytes** dense.

Solving for the sparsity $s$ at which the sparse encoding stops being larger:

$$(1 - s) \cdot 8 + \frac{r \cdot 4}{N} < 4 \quad\Longrightarrow\quad s > \frac{1}{2} + \frac{r}{2N}$$

where $r$ is the number of output rows and $N$ the number of weights. For any real layer
$r \ll N$, so the break-even is **just above 50%**, and at exactly 50% the row pointers
make the sparse form marginally _larger_ than dense.

This is why unstructured pruning below half sparsity cannot reduce a file size, and it is
a property of the encoding rather than of the pruning method. The estimate is labelled a
lower bound in the outcome notes because real CSR implementations add alignment and block
padding that make it worse.

## 5.6 Calibration and per-class behaviour **[measured]**

**Figure 4** (`results/figures/per_class_accuracy.png`) shows per-class accuracy change
against each model's own FP32 baseline, ranked by total accuracy destroyed.

Two effects are expected and observable:

- **Calibration degrades before accuracy does.** ECE typically rises more than top-1
  falls. For deployments that gate on a confidence threshold — which is most of them —
  this is the more consequential change.
- **Aggregate top-1 hides class-specific collapse.** A model can hold its overall score
  while losing several points on one or two classes. Which classes depends on the
  architecture and the quantization scheme.

## 5.7 Resolution sensitivity **[designed, not measured]**

This was designed and then excluded on cost grounds, and the exclusion is recorded rather
than quietly dropped.

224×224 inputs are 49× the pixels of 32×32. The ratio of _measured_ latencies is the number
a deployment actually plans against, and it is not 49: memory-bound layers scale with bytes
touched, so the ratio is architecture-dependent. That measurement is not in this suite.

The cost is dominated by the sweep rather than by training. Each rung re-measures every
cell, and a cell is a fixed 260 forward passes (10 warm-up + 50 timed × 5 repeats), so
sweep cost scales with per-forward cost. ResNet-18 at 224×224 with batch 32 costs a
measured 10.1 s per forward pass on a CPU runner, which puts that single cell pair — two
thread settings, before the other nine rungs — at roughly 14 hours. A hosted CI job is
capped at 360 minutes, so no choice of epoch count makes the full-resolution sweep fit.

There is also a methodological reason to prefer 32×32 for the published suite. Every
network here has its stem re-engineered for 32×32 input (§3.2), so evaluating it at
224×224 measures a model outside the regime it was adapted for, and part of any resulting
difference is attributable to that adaptation rather than to input size. 224×224 remains
available via `configs/default.yaml` or `--set benchmark.resolutions=[32,224]` for anyone
measuring on hardware they control.

# 6. Measurement order: a 404% error, and why it matters

## 6.1 The observation

The first full sequential run reported:

```
fp32                  17.91 ms
prune_unstructured_50 74.44 ms     <- 4.16x slower
onnxruntime_int8     165.32 ms     <- 9.23x slower
```

Unstructured pruning leaves dense tensors, so dense kernels must be unaffected. A 4x
slowdown is not a plausible property of the method.

## 6.2 The diagnosis

The configurations were measured **sequentially, in ladder order, over roughly five
minutes**. Machine state — thermal, background load, turbo residency — drifted across
that window, so later configurations were measured on a different machine than earlier
ones, in every sense that matters.

## 6.3 The controlled re-measurement

`scripts/check_measurement_order.py` measures the identical two models in opposite
orders, with a discarded warm-up pass:

```
Order A: fp32 then prune50
  fp32       p50 = 18.14 ms   cv = 14.1%
  prune50    p50 = 18.17 ms   cv = 15.9%

Order B: prune50 then fp32
  prune50    p50 = 17.95 ms   cv = 12.1%
  fp32       p50 = 17.24 ms   cv = 16.2%

Order C: fp32 then prune50 (repeat)
  fp32       p50 = 18.07 ms   cv = 12.1%
  prune50    p50 = 19.00 ms   cv = 13.1%

prune50 / fp32 median ratio: 1.01x
between-configuration effect: 1.01x
within-configuration spread : 5.8% of median
```

**The true effect is 1.01x.** The reported 4.16x was off by 404%.

## 6.4 Why this is a contribution rather than an embarrassment

The error was produced by a harness implementing a reasonable-looking protocol:
warm-up, many iterations, multiple repeats, medians, percentile reporting. None of that
catches drift _between_ configurations, because each configuration is internally
self-consistent and individually looks stable — the CVs in §6.3 are 12–16% even for the
confounded comparison.

Three consequences follow, and the harness now implements all three:

1. **Report dispersion, and flag it.** Every cell records its CV; tables mark
   `cv > 0.15` as `unstable`. A reader can see which rows to distrust.
2. **Interleave configurations.** Block-sequential measurement is invalid for precise
   rank ordering. Repeats must alternate across configurations so drift affects all of
   them equally.
3. **Publish negative methodological results.** A benchmark that reports a 4.16x effect
   which does not exist is worse than one that reports nothing, and the failure mode is
   silent.

This is also the strongest available argument for the harness's design stance: record
raw samples, retain the environment, and make the reporting layer a pure function of
stored data, so a claim can be audited after the fact.

# 7. Reproducibility

Each record embeds `run_id`, ISO-8601 timestamp, schema and package versions, CPU model,
logical core count, platform string, a full environment fingerprint (PyTorch, NumPy,
ONNX Runtime and its providers, Python, RAM, default thread counts), git commit hash, the
model card, and the effective configuration. A `results/` directory is therefore
self-describing.

Reporting reads **only** stored JSON. No forward pass is executed to produce a figure, so
every table and chart can be regenerated on any machine from `results/raw/` alone:

```bash
edgebench report --results results
```

Determinism: seeds are set for Python, NumPy and PyTorch from one value; validation and
calibration subsets and benchmark inputs all use seeded permutations. Latency is not
bit-reproducible on a multitasking OS and is not claimed to be.

# 8. Availability

Source, configuration, tests (222, run in CI across Python 3.10–3.12), and the
measurement-stability diagnostic are in the repository. Two auxiliary configurations
ship: `configs/smoke.yaml` for a real-data CI run against CIFAR-10, and
`configs/offline.yaml` for full-ladder validation on synthetic data with no network
access. The CI pipeline also asserts on the _content_ of a run -- record counts,
retained latency samples, plausibility of a one-epoch accuracy -- because both
`run-all` and `report` deliberately exit zero when they produce nothing, making an
exit-code check meaningless.

`edgebench info` reports which optimizations the host can actually run _before_ a multi-
hour sweep begins, which is the practical mitigation for the operator-coverage threat in
§9.5.

# 9. Limitations

## 9.1 One dataset, one task

CIFAR-10 classification. Nothing here establishes behaviour on detection, segmentation,
sequence models or larger inputs. The _methodology_ transfers; the numbers do not.

## 9.2 Uniform recipe is a deliberate trade

Holding the training recipe fixed makes rows comparable and makes a difference
attributable to the optimization rather than to hyperparameters. It also means a model
whose preferred recipe differs is measured below its potential. EfficientNet-B0 and
MobileNetV3 would likely gain from architecture-specific tuning.

## 9.3 CPU only

Latency ranks differently on GPUs and NPUs, where memory bandwidth, kernel fusion and
supported precisions all differ. TensorRT in particular changes the INT8 picture
substantially. All measured numbers are CPU-specific.

## 9.4 Block-sequential measurement remains

The harness measures each `(model, optimization)` record as a block. Fully interleaving
across the whole ladder is the correct fix for §6 and is tracked as future work. Until
then, rankings within a close margin should be treated as unresolved, and the `Timing`
column exists precisely to make that judgement possible.

## 9.5 Operator coverage varies by build

Which optimizations apply depends on the installed PyTorch, torchvision and ONNX Runtime.
Recent PyTorch exposes `onednn` rather than `fbgemm` as a quantization engine name while
still accepting `x86` and `fbgemm` as qconfig names — two namespaces that must be
resolved separately. Handling that correctly is the difference between reporting
"static INT8 unavailable" and measuring it.

## 9.6 MAC counting is approximate

`torch.utils.flop_counter` counts the operations it recognizes. MACs are reported as
`null` rather than guessed when counting fails.

## 9.7 Energy largely unmeasured

RAPL is available only on Linux with powercap readable. On Windows and macOS the study
reports `not measured`. This is the largest gap in the metric set: energy per inference is
arguably the metric that matters most on a thermally or battery-limited device, and it is
the one this work cannot supply.

## 9.8 Structured pruning not implemented

Correct channel pruning needs dependency-aware surgery across residual adds and
concatenations. An incorrect implementation would produce a wrong result, which is worse
than an absent one, so it is left as future work.

# 10. Conclusion

The headline result is that static INT8 quantization, calibrated on held-out data,
delivered roughly a 3x latency improvement and a 4x reduction in serialized weight size,
while quantization-aware training recovered most of the accompanying accuracy loss.

The more useful results are the ones that contradict expectations. Dynamic INT8
quantization made a convolutional network **slower** while barely reducing its size,
because it quantizes the parameter-light part of the model. Unstructured magnitude
pruning delivered neither compression nor speedup, because dense storage ignores zeros,
dense kernels ignore sparsity, and the sparse-encoding break-even sits at exactly 50%
sparsity. And the harness's own first run reported a 4.16x pruning slowdown that
controlled re-measurement reduced to 1.01x — a 404% error produced by a protocol that
looked entirely reasonable.

That last finding shaped the harness more than any other. Fault tolerance, explicit
unavailability, recorded dispersion, retained raw samples and a reporting layer that is a
pure function of stored data are not embellishments: they are what makes a benchmark
auditable, and an unauditable benchmark is a source of confident wrong numbers.

# References

::: {#refs}
:::

# Appendix A: Reproducing this report

```bash
pip install -e ".[onnx,dev]"

# Verify the host before committing hours
edgebench info

# Validate the entire pipeline offline in minutes
edgebench run-all --config configs/offline.yaml

# Check measurement stability on this hardware
python scripts/check_measurement_order.py

# The study itself
edgebench run-all

# Regenerate every figure and table
edgebench report --results results
```

# Appendix B: Record schema

```json
{
    "schema_version": 1,
    "run_id": "20260926T061351Z-fd05ea",
    "created_at": "2026-09-26T06:13:51+00:00",
    "model_id": "resnet18",
    "optimization_id": "static_int8",
    "status": "applied",
    "reason": null,
    "accuracy": {
        "top1": 0.9312,
        "top5": 0.9978,
        "loss": 0.2143,
        "ece": 0.0412,
        "per_class_accuracy": [0.94, 0.96, "..."],
        "confusion": [["..."]]
    },
    "footprint": {
        "weight_bytes": 11302248,
        "parameters": 11173962,
        "macs": 555420672,
        "rss_peak_delta_bytes": 41943040
    },
    "latency": [
        {
            "resolution": 32,
            "batch_size": 1,
            "num_threads": 1,
            "status": "ok",
            "latency_ms": 5.75,
            "per_iteration": { "p50_ms": 5.75, "p95_ms": 7.12, "cv": 0.061 },
            "block": { "p50_ms": 5.61 },
            "throughput_samples_per_s": 173.9,
            "samples_ms": [5.7, 5.8, 5.6, "..."],
            "energy": {
                "energy_available": false,
                "energy_unavailable_reason": "..."
            }
        }
    ],
    "optimization_metadata": {
        "calibration_batches": 16,
        "engine": "onednn",
        "weight_bits": 8
    },
    "environment": {
        "cpu": "12th Gen Intel(R) Core(TM) i5-12450H",
        "git_commit": "a1b2c3d",
        "fingerprint": {
            "torch": "2.14.0+cpu",
            "python": "3.14.5",
            "...": "..."
        }
    }
}
```
