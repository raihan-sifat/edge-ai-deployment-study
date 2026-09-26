"""Benchmark statistics, footprint accounting and the latency protocol."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from edgebench.bench.energy import (
    RaplEnergyMeter,
    UnavailableEnergyMeter,
    create_energy_meter,
)
from edgebench.bench.footprint import footprint_summary, measure_peak_memory
from edgebench.bench.latency import LatencyMeasurement, make_inputs, measure_latency
from edgebench.bench.stats import TimingStats, summarise_speedup
from edgebench.inference import TorchPredictor, measure_state_dict_bytes


class TinyNet(nn.Module):
    """Small deterministic network for protocol tests."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def _scaled_net(num_classes: int = 10, width: int = 64, layers: int = 4) -> nn.Module:
    """A network with enough compute that batching behaviour is meaningful.

    ``TinyNet`` runs in ~0.06 ms, where per-batch dispatch overhead dominates and
    batching can measure *slower* per image. This builds something large enough
    for the throughput curves to have the shape a real deployment would see.
    """
    blocks: list[nn.Module] = [nn.Conv2d(3, width, 3, padding=1), nn.ReLU()]
    for _ in range(layers):
        blocks += [
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]
    blocks += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, num_classes)]
    return nn.Sequential(*blocks)


class TestTimingStats:
    def test_known_distribution(self):
        stats = TimingStats.from_samples_ms([1.0, 2.0, 3.0, 4.0, 5.0])

        assert stats.count == 5
        assert stats.mean == pytest.approx(3.0)
        assert stats.minimum == 1.0
        assert stats.maximum == 5.0
        assert stats.p50 == pytest.approx(3.0)
        assert stats.iqr == pytest.approx(2.0)

    def test_percentiles_ordered(self):
        samples = list(range(1, 101))
        stats = TimingStats.from_samples_ms([float(value) for value in samples])

        assert stats.minimum <= stats.p50 <= stats.p90 <= stats.p95 <= stats.p99
        assert stats.p99 <= stats.maximum

    def test_single_sample_has_zero_std(self):
        stats = TimingStats.from_samples_ms([4.2])
        assert stats.std == 0.0
        assert stats.cv == 0.0

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty sample set"):
            TimingStats.from_samples_ms([])

    def test_throughput_is_inverse_of_median(self):
        stats = TimingStats.from_samples_ms([10.0, 10.0, 10.0])
        assert stats.throughput_per_second == pytest.approx(100.0)

    def test_throughput_of_zero_latency_is_nan(self):
        stats = TimingStats.from_samples_ms([0.0, 0.0])
        assert math.isnan(stats.throughput_per_second)

    def test_to_dict_prefixing(self):
        stats = TimingStats.from_samples_ms([1.0, 2.0])
        prefixed = stats.to_dict(prefix="cell_")
        assert "cell_p50_ms" in prefixed
        assert "cell_cv" in prefixed

    def test_unstable_distribution_has_high_cv(self):
        stable = TimingStats.from_samples_ms([10.0, 10.1, 9.9, 10.05])
        unstable = TimingStats.from_samples_ms([10.0, 40.0, 5.0, 60.0])
        assert stable.cv < 0.05
        assert unstable.cv > 0.5


class TestSpeedup:
    def test_speedup_and_reduction(self):
        result = summarise_speedup(baseline_ms=20.0, candidate_ms=5.0)
        assert result["speedup"] == pytest.approx(4.0)
        assert result["latency_reduction_pct"] == pytest.approx(75.0)

    def test_slower_candidate(self):
        result = summarise_speedup(baseline_ms=10.0, candidate_ms=20.0)
        assert result["speedup"] == pytest.approx(0.5)
        assert result["latency_reduction_pct"] == pytest.approx(-100.0)

    def test_zero_baseline_is_nan_not_crash(self):
        assert math.isnan(summarise_speedup(0.0, 5.0)["speedup"])


class TestInputs:
    def test_shape_and_dtype(self):
        inputs = make_inputs(batch_size=4, resolution=32)
        assert inputs.shape == (4, 3, 32, 32)
        assert inputs.dtype == torch.float32

    def test_range_is_uniform_minus_one_to_one(self):
        """Uniform inputs avoid the denormal-activation artifact of Gaussian noise."""
        inputs = make_inputs(batch_size=32, resolution=16, seed=0)
        assert float(inputs.min()) >= -1.0
        assert float(inputs.max()) <= 1.0

    def test_deterministic_for_a_given_seed(self):
        first = make_inputs(batch_size=2, resolution=8, seed=7)
        second = make_inputs(batch_size=2, resolution=8, seed=7)
        third = make_inputs(batch_size=2, resolution=8, seed=8)

        assert torch.equal(first, second)
        assert not torch.equal(first, third)


class TestLatencyProtocol:
    @pytest.fixture
    def predictor(self):
        torch.manual_seed(0)
        return TorchPredictor(TinyNet(), label="tiny")

    def test_measurement_is_complete(self, predictor):
        result = measure_latency(
            predictor,
            resolution=32,
            batch_size=1,
            warmup_iters=1,
            timed_iters=3,
            repeats=2,
            num_threads=1,
            measure_memory=False,
        )

        assert result.status == "ok"
        assert result.latency_ms is not None and result.latency_ms > 0
        assert result.per_iteration is not None
        assert result.block is not None

        # timed_iters samples per repeat.
        assert len(result.samples_ms) == 3 * 2
        assert result.per_iteration["count"] == 6

    def test_throughput_is_consistent_with_latency_and_batch(self):
        """Invariant: throughput (samples/s) must equal 1000 * batch / p50_latency.

        Deliberately an arithmetic identity rather than "bigger batches are
        faster". Batching does not always improve throughput on CPU: for a model
        this small (p50 around 0.06 ms) per-batch Python and operator-dispatch
        overhead dominates, and batch 4 can measure *slower* per image than batch
        1. Asserting a speed-up here would encode a hardware assumption as a test.
        """
        for batch_size in (1, 4):
            result = measure_latency(
                TorchPredictor(_scaled_net()),
                32,
                batch_size,
                1,
                3,
                1,
                1,
                measure_memory=False,
            )

            assert result.latency_ms is not None and result.latency_ms > 0
            assert result.throughput_samples_per_s is not None
            assert result.throughput_samples_per_s == pytest.approx(
                1000.0 * batch_size / result.latency_ms, rel=1e-6
            )

    def test_batching_does_not_catastrophically_degrade_throughput(self):
        """Batching must not be pathologically broken. Direction is not asserted.

        An earlier version of this test asserted that batch 8 beats batch 1 in
        throughput, and it failed on the development machine: for a model of this
        size the two are within noise, and on a busy machine the batch-8 figure can
        come out slightly lower. That is a hardware and machine-load property, not
        a property of the harness, so encoding it as an assertion makes the suite
        flaky for no benefit.

        What *is* worth testing is that batching is not catastrophically broken --
        a real bug (activations recomputed per sample, a batch dimension handled
        incorrectly) would show up as throughput collapsing by an order of
        magnitude. The generous bound catches that while tolerating normal
        variation.
        """
        predictor = TorchPredictor(_scaled_net())
        single = measure_latency(predictor, 32, 1, 2, 5, 3, 1, measure_memory=False)
        batched = measure_latency(predictor, 32, 8, 2, 5, 3, 1, measure_memory=False)

        assert single.throughput_samples_per_s is not None
        assert batched.throughput_samples_per_s is not None
        assert batched.throughput_samples_per_s > 0.5 * single.throughput_samples_per_s, (
            f"batch-8 throughput {batched.throughput_samples_per_s:.0f} samples/s "
            f"is less than half of batch-1's {single.throughput_samples_per_s:.0f}; "
            "batching appears to be broken rather than merely unhelpful"
        )

    def test_thread_count_is_recorded(self, predictor):
        result = measure_latency(predictor, 32, 1, 1, 2, 1, num_threads=1, measure_memory=False)
        assert result.num_threads == 1

    def test_failure_is_captured_not_raised(self):
        """A backend that cannot serve a shape must not abort the grid."""

        class BrokenPredictor(TorchPredictor):
            def __call__(self, inputs):
                raise RuntimeError("simulated backend failure")

        broken = BrokenPredictor(TinyNet(), label="broken")
        result = measure_latency(broken, 32, 1, 1, 2, 1, 1, measure_memory=False)

        assert result.status == "failed"
        assert result.reason is not None
        assert "simulated backend failure" in result.reason
        assert result.latency_ms is None

    def test_memory_delta_recorded_when_enabled(self, predictor):
        result = measure_latency(predictor, 32, 1, 1, 2, 1, 1, measure_memory=True)
        assert "rss_peak_delta_bytes" in result.memory

    def test_round_trip_through_dict(self, predictor):
        result = measure_latency(predictor, 32, 1, 1, 2, 1, 1, measure_memory=False)
        restored = LatencyMeasurement.from_dict(result.to_dict())
        assert restored.latency_ms == result.latency_ms
        assert restored.resolution == result.resolution


class TestFootprint:
    def test_state_bytes_matches_dense_parameter_size(self):
        """Bytes of parameters plus a small, constant zip-container overhead.

        The overhead is ~1-2 KB and does not scale with model size, so it is
        negligible for the models in this study (tens of MB) but dominant for a
        1 KB toy model. Asserting a constant slack rather than a ratio captures
        the real property: the measurement counts parameters, plus a fixed shell.
        """
        model = TinyNet()
        expected = sum(p.numel() * p.element_size() for p in model.parameters())
        actual = measure_state_dict_bytes(model)

        assert actual >= expected, "serialized size cannot be smaller than raw weights"
        assert actual - expected < 8192, (
            f"container overhead was {actual - expected} bytes, which is larger than "
            "the expected few kilobytes of zip/metadata framing"
        )

    def test_state_bytes_overhead_is_constant_not_proportional(self):
        """The zip container overhead must not scale with model size.

        This is the property that makes the measurement trustworthy for real
        models: at 40 MB the few-kilobyte shell is below measurement precision,
        whereas for a 1 kB toy model it dominates. Asserting the overhead itself
        is constant is the correct test; asserting a size *ratio* is not, because
        it fails precisely when the model is very small.
        """
        small = TinyNet(num_classes=2)
        large = nn.Sequential(nn.Linear(256, 512), nn.Linear(512, 512))

        small_params = sum(p.numel() * 4 for p in small.parameters())
        large_params = sum(p.numel() * 4 for p in large.parameters())

        small_overhead = measure_state_dict_bytes(small) - small_params
        large_overhead = measure_state_dict_bytes(large) - large_params

        assert large_params > 10 * small_params, "fixture must differ by orders of magnitude"
        assert small_overhead >= 0
        assert large_overhead >= 0
        # Both overheads should be within a kilobyte of each other.
        assert abs(small_overhead - large_overhead) < 2048

    def test_footprint_summary_fields(self):
        model = TinyNet()
        predictor = TorchPredictor(model)
        summary = footprint_summary(
            predictor, model, torch.zeros(1, 3, 32, 32), input_size=32, measure_memory=False
        )

        assert summary["weight_bytes"] > 0
        assert summary["parameters"] == sum(p.numel() for p in model.parameters())
        assert summary["input_pixels"] == 32 * 32
        assert summary["batch_size_for_memory"] == 1
        assert "compression_ratio" in summary

    def test_input_pixels_track_resolution(self):
        model = TinyNet()
        predictor = TorchPredictor(model)
        summary = footprint_summary(
            predictor, model, torch.zeros(1, 3, 224, 224), input_size=224, measure_memory=False
        )
        assert summary["input_pixels"] == 224 * 224

    def test_peak_memory_is_non_negative(self):
        model = TinyNet()
        predictor = TorchPredictor(model)
        reading = measure_peak_memory(predictor, torch.zeros(1, 3, 32, 32), iterations=2)

        assert reading.delta_bytes >= 0
        assert reading.peak_bytes >= reading.baseline_bytes


class TestEnergy:
    def test_unavailable_meter_reports_reason_and_no_fabricated_number(self):
        meter = UnavailableEnergyMeter("no hardware counter on this platform")
        assert meter.available is False

        calls = {"count": 0}

        def work() -> None:
            calls["count"] += 1

        reading = meter.measure(work, iterations=3)

        assert calls["count"] == 3, "work must still run even when energy is unmeasurable"
        assert reading.available is False
        assert reading.joules is None
        assert reading.per_iteration_joules(3) is None
        assert reading.average_watts() is None
        assert reading.reason == "no hardware counter on this platform"

    def test_create_meter_never_raises(self):
        meter = create_energy_meter()
        assert isinstance(meter, UnavailableEnergyMeter | RaplEnergyMeter)

    def test_rapl_meter_without_domains_is_unavailable(self):
        assert RaplEnergyMeter([]).available is False

    def test_energy_reading_dict_is_json_safe(self):
        import json

        reading = UnavailableEnergyMeter("n/a").measure(lambda: None, iterations=1)
        json.dumps(reading.to_dict())


def test_predictor_thread_configuration():
    predictor = TorchPredictor(TinyNet())
    original = torch.get_num_threads()

    try:
        predictor.configure_threads(1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)


def test_predictor_handles_tuple_output():
    class TupleNet(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 10), "auxiliary"

    predictor = TorchPredictor(TupleNet())
    logits = predictor(torch.zeros(2, 3, 32, 32))
    assert logits.shape == (2, 10)
