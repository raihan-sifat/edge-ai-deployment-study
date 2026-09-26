"""Model construction and CIFAR adaptation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from edgebench.config import ModelEntry
from edgebench.models import (
    build_model,
    count_multiply_accumulates,
    parameter_breakdown,
)
from edgebench.models.adapt import (
    adapt_stem_to_cifar,
    find_first_conv,
    find_first_maxpool,
    get_submodule,
    replace_classifier_head,
    set_submodule,
    verify_head,
)

ADAPTED_MODELS = [
    "resnet18",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "shufflenet_v2_x1_0",
    "efficientnet_b0",
]


class TestStemAdaptation:
    def test_resnet_stem_becomes_stride_one_3x3(self):
        model = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        name, conv = find_first_conv(model)

        assert name == "conv1"
        assert conv.kernel_size == (3, 3)
        assert conv.stride == (1, 1)
        assert conv.padding == (1, 1), "padding must be 1 to preserve 32x32 resolution"

    def test_resnet_first_maxpool_removed(self):
        """Built with adapt='none' there IS a max-pool; after adaptation there is not."""
        entry = ModelEntry(id="resnet18", adapt="none")
        model = build_model(entry, num_classes=10, verify=False)

        pool = find_first_maxpool(model)
        assert pool is not None, "the ImageNet stem must contain a max-pool"
        pool_name, pool_module = pool
        assert isinstance(pool_module, nn.MaxPool2d)

        adapt_stem_to_cifar(model, verbose=False)

        assert isinstance(model.get_submodule(pool_name), nn.Identity)
        assert find_first_maxpool(model) is None, "no MaxPool2d may survive adaptation"

    def test_build_model_already_adapts_when_configured(self):
        model = build_model(ModelEntry(id="resnet18", adapt="cifar"), num_classes=10, verify=False)
        assert find_first_maxpool(model) is None

    def test_stem_adaptation_preserves_spatial_resolution(self):
        """The whole point: 32x32 in must not collapse to 1x1 before the last stage."""
        model = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        model.eval()

        captured: dict[str, tuple[int, ...]] = {}

        def hook(_module, _inputs, output):
            captured.setdefault("shape", tuple(output.shape))

        # conv1 is the first stage; its output must still be 32x32.
        model.get_submodule("conv1").register_forward_hook(hook)
        with torch.inference_mode():
            model(torch.zeros(1, 3, 32, 32))

        assert captured["shape"][-2:] == (32, 32)

    def test_adaptation_is_idempotent(self):
        model = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        before_conv = dict(model.conv1.named_parameters())

        adapt_stem_to_cifar(model, verbose=False)
        adapt_stem_to_cifar(model, verbose=False)

        after_conv = dict(model.conv1.named_parameters())
        assert set(before_conv) == set(after_conv)
        assert model.conv1.stride == (1, 1)

    def test_seven_by_seven_kernel_is_replaced_not_padded(self):
        """A 7x7 kernel cannot be reused; the replacement must be a fresh 3x3."""
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3))
        adapt_stem_to_cifar(model, verbose=False)

        conv = model[0]
        assert isinstance(conv, nn.Conv2d)
        assert conv.kernel_size == (3, 3)
        assert conv.stride == (1, 1)
        assert conv.in_channels == 3
        assert conv.out_channels == 8

    def test_adapt_none_leaves_model_untouched(self):
        model = build_model(ModelEntry(id="resnet18", adapt="none"), num_classes=10, verify=False)
        _, conv = find_first_conv(model)
        assert conv.stride == (2, 2), "adapt='none' must not modify the stem"


class TestClassifierHead:
    @pytest.mark.parametrize("name", ADAPTED_MODELS)
    def test_head_replaced_for_requested_classes(self, name):
        entry = ModelEntry(id=name, torchvision_name=name)
        model = build_model(entry, num_classes=7)
        model.eval()

        with torch.inference_mode():
            logits = model(torch.zeros(1, 3, 32, 32))

        assert logits.shape == (1, 7)

    @pytest.mark.parametrize("num_classes", [10, 100])
    def test_various_class_counts(self, num_classes):
        model = build_model(ModelEntry(id="resnet18"), num_classes=num_classes)
        verify_head(model, num_classes)

    def test_verify_head_rejects_wrong_class_count(self):
        model = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        with pytest.raises(ValueError, match="expected logits"):
            verify_head(model, num_classes=42)

    def test_unregistered_model_raises(self):
        with pytest.raises(KeyError, match="no classifier head adapter"):
            replace_classifier_head(nn.Linear(4, 4), "not_a_model", 10)

    def test_verify_head_restores_training_mode(self):
        model = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        model.train()
        verify_head(model, 10)
        assert model.training is True


class TestSubmoduleHelpers:
    def test_get_and_set_round_trip(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())
        replacement = nn.Conv2d(3, 8, 1)
        set_submodule(model, "0", replacement)
        assert get_submodule(model, "0") is replacement

    def test_set_root_rejected(self):
        with pytest.raises(ValueError, match="root module"):
            set_submodule(nn.Linear(2, 2), "", nn.Linear(2, 2))

    def test_get_root_returns_model(self):
        model = nn.Linear(2, 2)
        assert get_submodule(model, "") is model

    def test_no_conv_raises(self):
        with pytest.raises(ValueError, match="no Conv2d"):
            find_first_conv(nn.Sequential(nn.ReLU()))

    def test_find_maxpool_returns_none_when_absent(self):
        assert find_first_maxpool(nn.Sequential(nn.Conv2d(3, 4, 3))) is None


class TestAnalysis:
    def test_parameter_counts_add_up(self):
        model = build_model(ModelEntry(id="resnet18"), num_classes=10)
        breakdown = parameter_breakdown(model)

        assert breakdown.total == sum(p.numel() for p in model.parameters())
        assert breakdown.total == breakdown.trainable + breakdown.frozen
        assert (
            breakdown.conv + breakdown.linear + breakdown.normalization + breakdown.other
            == breakdown.total
        )

    def test_resnet18_parameter_count_is_plausible(self):
        """CIFAR-adapted ResNet-18 has ~11.2M parameters (ImageNet version ~11.7M)."""
        model = build_model(ModelEntry(id="resnet18"), num_classes=10)
        total = parameter_breakdown(model).total
        assert 10_000_000 < total < 12_000_000

    def test_frozen_parameters_detected(self):
        model = build_model(ModelEntry(id="resnet18"), num_classes=10)
        for parameter in model.conv1.parameters():
            parameter.requires_grad = False

        breakdown = parameter_breakdown(model)
        assert breakdown.frozen == sum(p.numel() for p in model.conv1.parameters())
        assert breakdown.trainable < breakdown.total

    def test_mac_count_is_positive_and_scales_with_input(self):
        model = build_model(ModelEntry(id="mobilenet_v3_small"), num_classes=10)

        macs_32 = count_multiply_accumulates(model, input_size=32, batch_size=1)
        macs_64 = count_multiply_accumulates(model, input_size=64, batch_size=1)

        assert macs_32 is not None and macs_32 > 0
        assert macs_64 is not None
        # 4x the pixels must cost noticeably more compute (not necessarily exactly
        # 4x: the classifier head is constant).
        assert macs_64 > macs_32

    def test_mac_count_scales_linearly_with_batch(self):
        model = build_model(ModelEntry(id="mobilenet_v3_small"), num_classes=10)
        one = count_multiply_accumulates(model, input_size=32, batch_size=1)
        four = count_multiply_accumulates(model, input_size=32, batch_size=4)
        assert one is not None and four is not None
        assert four == pytest.approx(4 * one, rel=1e-6)

    def test_mac_count_leaves_model_in_eval_after_call(self):
        model = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        model.train()
        count_multiply_accumulates(model, input_size=32)
        assert model.training is True, "counting must restore the previous mode"


class TestCheckpointing:
    def test_round_trip(self, tmp_path):
        from edgebench.models import checkpoint_path, load_checkpoint, save_checkpoint

        model = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        path = checkpoint_path(tmp_path, "resnet18", "fp32")

        save_checkpoint(
            model,
            path,
            model_id="resnet18",
            num_classes=10,
            input_size=32,
            metadata={"note": "unit test"},
        )
        assert path.exists()

        fresh = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        # Perturb so a successful load is provable.
        with torch.no_grad():
            for parameter in fresh.parameters():
                parameter.zero_()

        metadata = load_checkpoint(fresh, path)
        assert metadata["note"] == "unit test"
        assert metadata["model_id"] == "resnet18"

        reference = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        load_checkpoint(reference, path)
        for (name_a, a), (name_b, b) in zip(
            fresh.state_dict().items(), reference.state_dict().items(), strict=True
        ):
            assert name_a == name_b
            assert torch.equal(a, b)

    def test_missing_checkpoint_raises(self, tmp_path):
        from edgebench.models import load_checkpoint

        model = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            load_checkpoint(model, tmp_path / "absent.pt")

    def test_bare_state_dict_is_accepted(self, tmp_path):
        """A user should be able to drop in weights from elsewhere."""
        from edgebench.models import load_checkpoint

        model = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        path = tmp_path / "raw.pt"
        torch.save(model.state_dict(), path)

        fresh = build_model(ModelEntry(id="resnet18"), num_classes=10, verify=False)
        metadata = load_checkpoint(fresh, path)
        assert metadata["model_id"] is None


def test_unknown_family_rejected():
    with pytest.raises(ValueError, match="unknown family"):
        build_model(ModelEntry(id="x", family="transformer"), num_classes=10)


def test_unknown_torchvision_name_raises():
    with pytest.raises(KeyError):
        build_model(ModelEntry(id="nope", torchvision_name="nope"), num_classes=10)
