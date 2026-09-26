"""The optimization ladder: registration, isolation, and honest failure reporting.

These tests do not assert that an optimization speeds anything up. That is what
the experiment is for. They assert the *contract*: every optimization either
applies cleanly, or records a machine-readable reason why it could not, and never
leaves the process in a broken state.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edgebench.config import BenchmarkConfig, ModelEntry, OptimizationEntry, TrainConfig
from edgebench.inference import TorchPredictor
from edgebench.models import build_model
from edgebench.optim import LADDER, LADDER_IDS, OptimizationContext, apply_ladder, run_optimization
from edgebench.optim.base import clone_module, failed, is_fatal, unavailable
from edgebench.optim.pruning import measure_sparsity
from edgebench.optim.quantization import (
    select_qconfig_backend,
    select_quantization_backend,
)


@pytest.fixture
def model() -> nn.Module:
    torch.manual_seed(0)
    entry = ModelEntry(id="mobilenet_v3_small", torchvision_name="mobilenet_v3_small")
    built = build_model(entry, num_classes=10, input_size=32)
    return built.eval()


@pytest.fixture
def loaders() -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(64, 3, 32, 32, generator=generator)
    labels = torch.randint(0, 10, (64,), generator=generator)
    dataset = TensorDataset(images, labels)

    train = DataLoader(dataset, batch_size=16, shuffle=False)
    val = DataLoader(dataset, batch_size=16, shuffle=False)
    return train, val


@pytest.fixture
def context(model: nn.Module, loaders, tmp_path) -> OptimizationContext:
    train_loader, val_loader = loaders
    return OptimizationContext(
        model=model,
        model_id="mobilenet_v3_small",
        num_classes=10,
        input_size=32,
        device=torch.device("cpu"),
        seed=0,
        artifacts_dir=tmp_path / "artifacts",
        train_cfg=TrainConfig(epochs=1, qat_epochs=1, batch_size=16, log_interval=0),
        bench_cfg=BenchmarkConfig(
            resolutions=(32,), batch_sizes=(1,), warmup_iters=1, timed_iters=2, repeats=1
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        calib_loader=val_loader,
    )


class TestRegistry:
    def test_expected_ids_registered(self):
        assert set(LADDER_IDS) >= {
            "fp32",
            "compile",
            "dynamic_int8",
            "static_int8",
            "qat_int8",
            "prune_unstructured_50",
            "onnxruntime",
        }

    def test_every_entry_is_callable(self):
        for name, implementation in LADDER.items():
            assert callable(implementation), name

    def test_unknown_id_is_unavailable_not_crash(self, context):
        outcome = run_optimization(OptimizationEntry(id="does_not_exist"), context)
        assert outcome.status == "unavailable"
        assert outcome.reason is not None
        assert "unknown optimization id" in outcome.reason

    def test_registry_covers_default_config(self, configs_dir):
        from edgebench.config import load_config

        config = load_config(configs_dir / "default.yaml")
        for entry in config.optimizations:
            assert entry.id in LADDER, f"{entry.id} is enabled in config but not registered"


class TestBaseline:
    def test_fp32_applies(self, context):
        outcome = run_optimization(OptimizationEntry(id="fp32"), context)
        assert outcome.status == "applied"
        assert outcome.predictor is not None
        assert outcome.metadata["weight_bits"] == 32

    def test_fp32_output_is_unchanged(self, context):
        """The reference row must not alter the model."""
        outcome = run_optimization(OptimizationEntry(id="fp32"), context)
        assert outcome.predictor is not None

        inputs = torch.rand(2, 3, 32, 32)
        with torch.inference_mode():
            expected = context.model(inputs)
        assert torch.allclose(outcome.predictor(inputs), expected, atol=1e-6)


class TestIsolation:
    def test_optimizations_do_not_contaminate_each_other(self, context):
        """The ladder must be order-independent.

        Sparsity is compared before and after rather than against zero, because a
        freshly initialised network already contains a small fraction of exactly
        zero weights (a few tenths of a percent). Asserting "still zero" would
        conflate that with actual mutation.
        """
        baseline_sparsity = measure_sparsity(context.model)["zero_fraction_all"]
        reference = clone_module(context.model)

        run_optimization(OptimizationEntry(id="prune_unstructured_50"), context)
        assert measure_sparsity(context.model)["zero_fraction_all"] == pytest.approx(
            baseline_sparsity, abs=1e-9
        ), "pruning must not mutate the shared base model"

        run_optimization(OptimizationEntry(id="dynamic_int8"), context)

        inputs = torch.rand(2, 3, 32, 32)
        with torch.inference_mode():
            before = reference(inputs)
            after = context.model(inputs)
        assert torch.allclose(before, after, atol=1e-6)

    def test_pruning_outcome_differs_from_the_base_model(self, context):
        """Isolation must not be achieved by accidentally no-op'ing the pruning."""
        outcome = run_optimization(OptimizationEntry(id="prune_unstructured_50"), context)
        assert outcome.status == "applied"
        assert outcome.metadata["zero_fraction_prunable"] == pytest.approx(0.5, abs=0.02)

    def test_clone_module_is_independent(self, model):
        cloned = clone_module(model)
        assert cloned is not model

        with torch.no_grad():
            for parameter in cloned.parameters():
                parameter.zero_()

        assert any(p.abs().sum().item() > 0 for p in model.parameters())


class TestPruning:
    def test_applies_and_reports_sparsity(self, context):
        entry = OptimizationEntry(id="prune_unstructured_50", params={"amount": 0.5})
        outcome = run_optimization(entry, context)

        assert outcome.status == "applied"
        assert outcome.metadata["zero_fraction_prunable"] == pytest.approx(0.5, abs=0.02)

    def test_sparsity_amount_parsed_from_id(self, context):
        outcome = run_optimization(OptimizationEntry(id="prune_unstructured_70"), context)
        assert outcome.status == "applied"
        assert outcome.params["amount"] == pytest.approx(0.7)
        assert outcome.metadata["zero_fraction_prunable"] == pytest.approx(0.7, abs=0.02)

    def test_dense_size_is_not_reduced(self, context):
        """Magnitude pruning must not claim a file-size win it does not deliver."""
        baseline = TorchPredictor(clone_module(context.model))
        outcome = run_optimization(OptimizationEntry(id="prune_unstructured_50"), context)
        assert outcome.predictor is not None

        ratio = outcome.predictor.state_bytes() / baseline.state_bytes()
        assert 0.9 < ratio < 1.1, "dense serialization size should be ~unchanged"

    def test_sparse_estimate_crossover_is_at_fifty_percent(self, context):
        """Sparse storage with fp32 values and int32 indices only wins past 50%.

        Each surviving weight costs 8 bytes (4 value + 4 index) against 4 bytes
        dense, so at exactly 50% sparsity the two layouts cost the same and the
        row pointers make the sparse form marginally *worse*. This is not a bug in
        the estimate -- it is why unstructured pruning below 50% cannot reduce a
        file size, and why the estimate is reported rather than asserted as a win.
        """
        fifty = run_optimization(OptimizationEntry(id="prune_unstructured_50"), context)
        assert fifty.status == "applied"
        assert (
            fifty.metadata["estimated_sparse_weight_bytes"] >= fifty.metadata["dense_weight_bytes"]
        ), "at 50% sparsity a CSR estimate cannot beat dense storage"

        seventy = run_optimization(OptimizationEntry(id="prune_unstructured_70"), context)
        assert seventy.status == "applied"
        assert (
            seventy.metadata["estimated_sparse_weight_bytes"]
            < seventy.metadata["dense_weight_bytes"]
        ), "past the crossover a CSR estimate must be smaller than dense"

    def test_sparse_estimate_is_labelled_as_an_estimate(self, context):
        outcome = run_optimization(OptimizationEntry(id="prune_unstructured_50"), context)
        assert any("lower bound" in note for note in outcome.notes)
        assert "estimated_sparse_weight_bytes" in outcome.metadata
        assert "dense_weight_bytes" in outcome.metadata

    def test_invalid_amount_is_unavailable(self, context):
        entry = OptimizationEntry(id="prune_unstructured_50", params={"amount": 1.5})
        outcome = run_optimization(entry, context)
        assert outcome.status == "unavailable"
        assert "must lie in" in (outcome.reason or "")

    def test_unknown_scope_is_unavailable(self, context):
        entry = OptimizationEntry(
            id="prune_unstructured_50", params={"amount": 0.5, "scope": "diagonal"}
        )
        outcome = run_optimization(entry, context)
        assert outcome.status == "unavailable"

    def test_local_scope_also_applies(self, context):
        entry = OptimizationEntry(
            id="prune_unstructured_30", params={"amount": 0.3, "scope": "local"}
        )
        outcome = run_optimization(entry, context)
        assert outcome.status == "applied"

    def test_pruned_output_still_well_formed(self, context):
        outcome = run_optimization(OptimizationEntry(id="prune_unstructured_50"), context)
        assert outcome.predictor is not None
        logits = outcome.predictor(torch.rand(2, 3, 32, 32))
        assert logits.shape == (2, 10)
        assert torch.isfinite(logits).all()


class TestQuantization:
    def test_dynamic_quantization_applies_or_explains(self, context):
        outcome = run_optimization(OptimizationEntry(id="dynamic_int8"), context)
        assert outcome.status in {"applied", "unavailable", "failed"}

        if outcome.status == "applied":
            assert outcome.predictor is not None
            assert outcome.metadata["quantized_modules"] > 0
            logits = outcome.predictor(torch.rand(2, 3, 32, 32))
            assert logits.shape == (2, 10)
        else:
            assert outcome.reason, "a non-applied optimization must carry a reason"

    def test_static_quantization_applies_or_explains(self, context):
        outcome = run_optimization(OptimizationEntry(id="static_int8"), context)
        assert outcome.status in {"applied", "unavailable", "failed"}
        if outcome.status == "applied":
            assert outcome.predictor is not None
            assert outcome.metadata["calibration_batches"] > 0
            logits = outcome.predictor(torch.rand(2, 3, 32, 32))
            assert logits.shape == (2, 10)
        else:
            assert outcome.reason

    def test_static_without_calibration_loader_is_unavailable(self, context):
        from dataclasses import replace

        without_calibration = replace(context, calib_loader=None)
        outcome = run_optimization(OptimizationEntry(id="static_int8"), without_calibration)

        assert outcome.status == "unavailable"
        assert "calibration" in (outcome.reason or "")

    def test_qat_applies_and_records_cost(self, context):
        outcome = run_optimization(OptimizationEntry(id="qat_int8"), context)
        assert outcome.status in {"applied", "unavailable", "failed"}

        if outcome.status == "applied":
            assert outcome.metadata["qat_epochs"] == 1
            assert "qat_seconds" in outcome.metadata
            assert any("deployment cost" in note for note in outcome.notes)
        else:
            assert outcome.reason

    def test_engine_is_placed_in_original_state_after_failure(self, context):
        """A failed quantization attempt must not leave the global engine changed."""
        before = torch.backends.quantized.engine

        from dataclasses import replace

        broken = replace(
            context,
            calib_loader=DataLoader(
                TensorDataset(torch.zeros(0, 3, 32, 32), torch.zeros(0, dtype=torch.long)),
                batch_size=1,
            ),
        )
        run_optimization(OptimizationEntry(id="static_int8"), broken)

        assert torch.backends.quantized.engine == before

    def test_engine_selection_returns_a_supported_name(self):
        engine = select_quantization_backend()
        assert engine in set(torch.backends.quantized.supported_engines)

    def test_qconfig_backend_is_accepted_by_torch(self):
        engine = select_quantization_backend()
        qconfig_backend = select_qconfig_backend(engine)
        # Must not raise.
        torch.ao.quantization.get_default_qconfig(qconfig_backend)


class TestOnnxRuntime:
    def test_export_produces_a_usable_predictor(self, context):
        outcome = run_optimization(OptimizationEntry(id="onnxruntime"), context)

        assert outcome.status in {"applied", "unavailable", "failed"}
        if outcome.status != "applied":
            assert outcome.reason
            return

        assert outcome.predictor is not None
        assert outcome.predictor.state_bytes() > 0

        logits = outcome.predictor(torch.rand(2, 3, 32, 32))
        assert logits.shape == (2, 10)

    def test_onnx_matches_pytorch_within_tolerance(self, context):
        """The exported graph must compute the same function as the source model."""
        outcome = run_optimization(OptimizationEntry(id="onnxruntime"), context)
        if outcome.status != "applied" or outcome.predictor is None:
            pytest.skip("onnxruntime unavailable")

        inputs = torch.rand(2, 3, 32, 32)
        with torch.inference_mode():
            expected = context.model(inputs)
        actual = outcome.predictor(inputs)

        assert torch.allclose(expected, actual, atol=1e-4)

    def test_artifacts_are_written(self, context):
        outcome = run_optimization(OptimizationEntry(id="onnxruntime"), context)
        if outcome.status != "applied":
            pytest.skip("onnxruntime unavailable")

        from pathlib import Path

        for path in outcome.artifacts.values():
            assert Path(path).exists()

    def test_int8_variant_smaller_than_fp32(self, context):
        fp32 = run_optimization(OptimizationEntry(id="onnxruntime"), context)
        if fp32.status != "applied" or fp32.predictor is None:
            pytest.skip("onnxruntime unavailable")

        int8 = run_optimization(OptimizationEntry(id="onnxruntime_int8"), context)
        if int8.status != "applied" or int8.predictor is None:
            pytest.skip("onnx int8 quantization unavailable")

        assert int8.predictor.state_bytes() < fp32.predictor.state_bytes()

    def test_dynamic_batch_axis_is_honoured(self, context):
        outcome = run_optimization(OptimizationEntry(id="onnxruntime"), context)
        if outcome.status != "applied" or outcome.predictor is None:
            pytest.skip("onnxruntime unavailable")

        # A dynamic batch axis means both of these must work on one session.
        assert outcome.predictor(torch.rand(1, 3, 32, 32)).shape == (1, 10)
        assert outcome.predictor(torch.rand(5, 3, 32, 32)).shape == (5, 10)


class TestCompile:
    def test_compile_applies_or_records_reason(self, context):
        outcome = run_optimization(OptimizationEntry(id="compile"), context)

        assert outcome.status in {"applied", "unavailable", "failed"}
        if outcome.status == "unavailable":
            # Unavailability must be explained, and must be a toolchain problem.
            assert outcome.reason
            assert any(
                marker in outcome.reason.lower()
                for marker in ("backend", "triton", "compiler", "not found")
            )
        if outcome.status == "applied":
            assert outcome.metadata["compile_seconds"] >= 0.0
            assert outcome.predictor is not None
            assert outcome.predictor(torch.rand(1, 3, 32, 32)).shape == (1, 10)


class TestLadderOrchestration:
    def test_full_ladder_returns_one_outcome_per_entry(self, context):
        entries = [OptimizationEntry(id=name) for name in ("fp32", "prune_unstructured_50")]
        outcomes = apply_ladder(entries, context)

        assert len(outcomes) == len(entries)
        assert [o.optimization_id for o in outcomes] == [e.id for e in entries]

    def test_one_failing_entry_does_not_stop_the_ladder(self, context, monkeypatch):
        """Fault tolerance is the central design property of the ladder."""
        import edgebench.optim as optim_module

        def explode(_ctx, _params):
            raise RuntimeError("simulated implementation crash")

        monkeypatch.setitem(optim_module.LADDER, "prune_unstructured_50", explode)

        entries = [
            OptimizationEntry(id="fp32"),
            OptimizationEntry(id="prune_unstructured_50"),
            OptimizationEntry(id="dynamic_int8"),
        ]
        outcomes = apply_ladder(entries, context)

        assert len(outcomes) == 3
        assert outcomes[0].status == "applied"
        assert outcomes[1].status == "failed"
        assert "simulated implementation crash" in (outcomes[1].reason or "")
        # The ladder continued past the failure.
        assert outcomes[2].status in {"applied", "unavailable", "failed"}

    def test_outcome_serialises_to_json(self, context):
        import json

        for entry in [OptimizationEntry(id=name) for name in ("fp32", "prune_unstructured_50")]:
            outcome = run_optimization(entry, context)
            json.dumps(outcome.to_dict())

    def test_optimization_id_is_normalised_to_the_config_id(self, context):
        """Implementations may pick a generic label; the ladder always rewrites it."""
        outcome = run_optimization(OptimizationEntry(id="prune_unstructured_70"), context)
        assert outcome.optimization_id == "prune_unstructured_70"


class TestOutcomeHelpers:
    def test_unavailable_carries_reason(self):
        outcome = unavailable("x", "because reasons")
        assert outcome.status == "unavailable"
        assert outcome.reason == "because reasons"
        assert outcome.applied is False

    def test_failed_captures_exception_text(self):
        try:
            raise ValueError("boom")
        except ValueError as error:
            outcome = failed("x", error)

        assert outcome.status == "failed"
        assert "boom" in (outcome.reason or "")

    def test_is_fatal_excludes_interrupts_and_memory_errors(self):
        assert is_fatal(KeyboardInterrupt())
        assert is_fatal(SystemExit())
        assert is_fatal(MemoryError())
        assert not is_fatal(ValueError("ordinary"))

    def test_applied_requires_a_predictor(self):
        from edgebench.optim.base import OptimizationOutcome

        assert OptimizationOutcome("x", "applied").applied is False
        assert OptimizationOutcome(
            "x", "applied", predictor=TorchPredictor(nn.Linear(2, 2))
        ).applied
