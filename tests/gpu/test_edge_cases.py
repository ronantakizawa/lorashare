"""
Edge case and robustness tests for lorashare.

Tests boundary conditions, dtype handling, error recovery,
and unusual but valid configurations.
"""

import warnings
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lorashare import SHAREModel

from .conftest import TASK_NAMES, requires_adapters, requires_gpu


@requires_adapters
@requires_gpu
class TestEdgeCases:
    """Robustness tests for edge cases and unusual configurations."""

    def test_two_adapters_minimum(self, adapter_paths):
        """Library should work with exactly 2 adapters (the minimum)."""
        paths = {k: v for k, v in list(adapter_paths.items())[:2]}

        share = SHAREModel.from_adapters(paths, num_components=32, device="cuda")
        assert len(share.adapter_names) == 2

        # Both should reconstruct
        for name in share.adapter_names:
            weights = share.reconstruct(name)
            assert len(weights) > 0

    def test_num_components_equals_max(self, adapter_paths):
        """Setting k = max available components should work."""
        # With 4 adapters of rank 8, max components = min(feature_dim, 4*8) = 32
        share = SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")
        assert share.config.num_components == 32

        for name in share.adapter_names:
            weights = share.reconstruct(name)
            assert len(weights) > 0

    def test_num_components_exceeds_max(self, adapter_paths):
        """Setting k > max available should clamp with a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            share = SHAREModel.from_adapters(
                adapter_paths, num_components=9999, device="cuda"
            )

            # Should have received a clamping warning
            clamping_warnings = [
                x for x in w if "clamping" in str(x.message).lower()
            ]
            assert len(clamping_warnings) > 0, "Expected a clamping warning"

        # Model should still work
        assert share.config.num_components < 9999
        for name in share.adapter_names:
            weights = share.reconstruct(name)
            assert len(weights) > 0

    def test_variance_threshold_low(self, adapter_paths):
        """Very low variance threshold (0.5) should select fewer components."""
        share_low = SHAREModel.from_adapters(
            adapter_paths, num_components="auto", variance_threshold=0.5, device="cuda"
        )
        share_high = SHAREModel.from_adapters(
            adapter_paths, num_components="auto", variance_threshold=0.99, device="cuda"
        )

        assert share_low.config.num_components <= share_high.config.num_components, (
            f"Lower threshold ({share_low.config.num_components} components) "
            f"selected more than higher threshold ({share_high.config.num_components})"
        )

    def test_variance_threshold_near_one(self, adapter_paths):
        """threshold=0.99 should select many components but still work."""
        share = SHAREModel.from_adapters(
            adapter_paths, num_components="auto", variance_threshold=0.99, device="cuda"
        )
        assert share.config.num_components >= 1

        for name in share.adapter_names:
            err = share.reconstruction_error(name, original_path=adapter_paths[name])
            assert err["mean"] < 0.5, f"High error ({err['mean']:.4f}) despite 99% variance threshold"

    def test_auto_vs_fixed_component_selection(self, adapter_paths):
        """Both 'auto' and fixed k should produce valid compression."""
        share_auto = SHAREModel.from_adapters(
            adapter_paths, num_components="auto", variance_threshold=0.95, device="cuda"
        )
        share_fixed = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cuda"
        )

        assert share_auto.config.component_selection == "auto"
        assert share_fixed.config.component_selection == "fixed"

        # Both should produce valid reconstructions
        for name in ["sst2", "cola"]:
            auto_recon = share_auto.reconstruct(name)
            fixed_recon = share_fixed.reconstruct(name)
            assert len(auto_recon) > 0
            assert len(fixed_recon) > 0

    def test_reconstruction_error_decreases_with_k(self, adapter_paths):
        """Reconstruction error should monotonically decrease with more components."""
        errors = []
        for k in [2, 4, 8, 16, 32]:
            share = SHAREModel.from_adapters(
                adapter_paths, num_components=k, device="cuda"
            )
            err = share.reconstruction_error("sst2", original_path=adapter_paths["sst2"])
            errors.append((k, err["mean"]))

        for i in range(len(errors) - 1):
            k_low, err_low = errors[i]
            k_high, err_high = errors[i + 1]
            assert err_low >= err_high, (
                f"Error increased from k={k_low} ({err_low:.6f}) "
                f"to k={k_high} ({err_high:.6f})"
            )

    def test_dtype_float16_adapters(self, adapter_paths, tmp_path):
        """Library should handle float16 adapters by converting internally."""
        import json

        # Create float16 copies of first two adapters
        fp16_paths = {}
        for i, (name, path) in enumerate(list(adapter_paths.items())[:2]):
            from safetensors.torch import load_file

            weights = load_file(str(Path(path) / "adapter_model.safetensors"))
            fp16_weights = {k: v.half() for k, v in weights.items()}

            fp16_dir = tmp_path / f"fp16_{name}"
            fp16_dir.mkdir()
            save_file(fp16_weights, str(fp16_dir / "adapter_model.safetensors"))

            # Copy config
            config_src = Path(path) / "adapter_config.json"
            config_dst = fp16_dir / "adapter_config.json"
            with open(config_src) as f:
                config = json.load(f)
            with open(config_dst, "w") as f:
                json.dump(config, f)

            fp16_paths[name] = str(fp16_dir)

        # Should handle float16 without errors
        share = SHAREModel.from_adapters(fp16_paths, num_components=8, device="cuda")
        assert len(share.adapter_names) == 2

        for name in share.adapter_names:
            weights = share.reconstruct(name)
            assert len(weights) > 0

    def test_on_error_skip_with_corrupted_adapter(self, adapter_paths, tmp_path):
        """on_error='skip' should skip invalid adapters and compress the rest."""
        # Create a corrupted adapter directory
        corrupt_dir = tmp_path / "corrupt_adapter"
        corrupt_dir.mkdir()
        # No adapter files — just an empty directory

        mixed_paths = dict(adapter_paths)
        mixed_paths["corrupt"] = str(corrupt_dir)

        share = SHAREModel.from_adapters(
            mixed_paths, num_components=32, device="cuda", on_error="skip"
        )

        # Should have skipped the corrupted one
        assert "corrupt" not in share.adapter_names
        assert len(share.adapter_names) == 4  # Original 4 should all be valid

    def test_on_error_raise_with_corrupted_adapter(self, adapter_paths, tmp_path):
        """on_error='raise' should fail on invalid adapters."""
        corrupt_dir = tmp_path / "corrupt_adapter"
        corrupt_dir.mkdir()

        mixed_paths = dict(adapter_paths)
        mixed_paths["corrupt"] = str(corrupt_dir)

        with pytest.raises(Exception):
            SHAREModel.from_adapters(
                mixed_paths, num_components=32, device="cuda", on_error="raise"
            )

    def test_large_classifier_head_preserved(self, adapter_paths, share_model):
        """Adapter with classifier heads should have them fully preserved."""
        for task_name in share_model.adapter_names:
            heads = share_model.all_classifier_heads.get(task_name, {})
            if not heads:
                continue

            reconstructed = share_model.reconstruct(task_name)

            for key, original_tensor in heads.items():
                assert key in reconstructed, f"Classifier key {key} missing in reconstruction"
                assert torch.equal(reconstructed[key], original_tensor), (
                    f"Classifier head {key} modified during compression"
                )

    def test_single_adapter_rejected(self, adapter_paths):
        """Compressing a single adapter should raise ValueError."""
        single = {k: v for k, v in list(adapter_paths.items())[:1]}

        with pytest.raises(ValueError, match="at least 2"):
            SHAREModel.from_adapters(single, num_components=32, device="cuda")

    def test_reconstruction_error_metrics_valid(self, adapter_paths, share_model):
        """Reconstruction error metrics should be well-formed."""
        for task_name in TASK_NAMES:
            err = share_model.reconstruction_error(
                task_name, original_path=adapter_paths[task_name]
            )

            assert "per_layer" in err
            assert "mean" in err
            assert "max" in err
            assert isinstance(err["mean"], float)
            assert isinstance(err["max"], float)
            assert err["mean"] >= 0
            assert err["max"] >= 0
            assert err["max"] >= err["mean"]
            assert len(err["per_layer"]) > 0

    def test_add_then_remove_adapter(self, adapter_paths):
        """Adding and then removing an adapter should leave model valid."""
        first_three = {k: v for k, v in list(adapter_paths.items())[:3]}
        share = SHAREModel.from_adapters(first_three, num_components=32, device="cuda")

        fourth_name = list(adapter_paths.keys())[3]
        fourth_path = adapter_paths[fourth_name]

        # Add
        share.add_adapter(fourth_path, name=fourth_name, device="cuda")
        assert len(share.adapter_names) == 4

        # Remove
        share.remove_adapter(fourth_name)
        assert len(share.adapter_names) == 3
        assert fourth_name not in share.adapter_names

        # Remaining adapters should still reconstruct
        for name in share.adapter_names:
            weights = share.reconstruct(name)
            assert len(weights) > 0

    def test_remove_last_adapter_rejected(self, adapter_paths):
        """Removing the last adapter should raise ValueError."""
        two_paths = {k: v for k, v in list(adapter_paths.items())[:2]}
        share = SHAREModel.from_adapters(two_paths, num_components=32, device="cuda")

        first_name = share.adapter_names[0]
        share.remove_adapter(first_name)

        # Now only 1 adapter remains — removing it should fail
        last_name = share.adapter_names[0]
        with pytest.raises(ValueError, match="last adapter"):
            share.remove_adapter(last_name)

    def test_missing_adapter_raises_keyerror(self, share_model):
        """Reconstructing a non-existent adapter should raise KeyError."""
        with pytest.raises(KeyError):
            share_model.reconstruct("nonexistent_adapter_xyz")

    def test_on_error_invalid_value_raises(self, adapter_paths):
        """Invalid on_error value should raise ValueError."""
        with pytest.raises(ValueError, match="on_error"):
            SHAREModel.from_adapters(
                adapter_paths, num_components=32, on_error="invalid"
            )
