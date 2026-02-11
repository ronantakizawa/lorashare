"""Tests for the I/O module."""

import json

import pytest
import torch

from lorashare.config import SHAREConfig
from lorashare.compression import (
    combine_adapter_weights,
    compute_adapter_loadings,
    compute_shared_components,
)
from lorashare.io import (
    load_peft_adapter,
    load_share_checkpoint,
    save_reconstructed_adapter,
    save_share_checkpoint,
    validate_adapters,
)


# ── Load PEFT Adapter ────────────────────────────────────────────────────────


class TestLoadPeftAdapter:
    def test_load_local(self, saved_adapters, synthetic_lora_config):
        weights, classifier_heads, config, name = load_peft_adapter(saved_adapters["cola"])
        assert name == "cola"
        assert config["peft_type"] == "LORA"
        assert all("lora_" in k for k in weights)

    def test_custom_name(self, saved_adapters):
        _, _, _, name = load_peft_adapter(saved_adapters["cola"], adapter_name="my_adapter")
        assert name == "my_adapter"

    def test_rejects_non_lora(self, tmp_path):
        adapter_dir = tmp_path / "bad"
        adapter_dir.mkdir()
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump({"peft_type": "PREFIX_TUNING"}, f)
        from safetensors.torch import save_file
        save_file({"x": torch.zeros(1)}, str(adapter_dir / "adapter_model.safetensors"))
        with pytest.raises(ValueError, match="PREFIX_TUNING"):
            load_peft_adapter(str(adapter_dir))

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_peft_adapter(str(tmp_path / "nonexistent"))

    def test_separates_lora_and_classifier_keys(self, tmp_path, synthetic_lora_config):
        """Non-LoRA keys like classifier.weight should be separated."""
        from safetensors.torch import save_file
        adapter_dir = tmp_path / "with_classifier"
        adapter_dir.mkdir()
        weights = {
            "base_model.model.encoder.layer.0.attention.self.query.lora_A.weight": torch.randn(4, 64),
            "base_model.model.encoder.layer.0.attention.self.query.lora_B.weight": torch.randn(64, 4),
            "base_model.model.classifier.weight": torch.randn(2, 768),
        }
        save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(synthetic_lora_config, f)

        lora_weights, classifier_heads, _, _ = load_peft_adapter(str(adapter_dir))
        assert len(lora_weights) == 2  # Only LoRA keys
        assert len(classifier_heads) == 1  # Only classifier key
        assert "base_model.model.classifier.weight" in classifier_heads


# ── Validate Adapters ────────────────────────────────────────────────────────


class TestValidateAdapters:
    def test_valid_configs(self, synthetic_lora_config):
        configs = {
            "cola": synthetic_lora_config,
            "mrpc": synthetic_lora_config.copy(),
        }
        validate_adapters(configs)  # Should not raise

    def test_rejects_single_adapter(self, synthetic_lora_config):
        with pytest.raises(ValueError, match="at least 2"):
            validate_adapters({"cola": synthetic_lora_config})

    def test_rejects_rank_mismatch(self, synthetic_lora_config):
        cfg2 = synthetic_lora_config.copy()
        cfg2["r"] = 16
        with pytest.raises(ValueError, match="Rank mismatch"):
            validate_adapters({"cola": synthetic_lora_config, "mrpc": cfg2})

    def test_rejects_base_model_mismatch(self, synthetic_lora_config):
        cfg2 = synthetic_lora_config.copy()
        cfg2["base_model_name_or_path"] = "bert-base"
        with pytest.raises(ValueError, match="Base model mismatch"):
            validate_adapters({"cola": synthetic_lora_config, "mrpc": cfg2})

    def test_rejects_target_modules_mismatch(self, synthetic_lora_config):
        cfg2 = synthetic_lora_config.copy()
        cfg2["target_modules"] = ["query", "key"]
        with pytest.raises(ValueError, match="Target modules mismatch"):
            validate_adapters({"cola": synthetic_lora_config, "mrpc": cfg2})


# ── SHARE Checkpoint Round-Trip ──────────────────────────────────────────────


class TestShareCheckpointRoundTrip:
    def test_save_load(self, tmp_path, synthetic_adapters, synthetic_lora_config):
        # Compress
        combined = combine_adapter_weights(synthetic_adapters)
        components, eigenvalues, k = compute_shared_components(combined, num_components=8)
        all_loadings = {}
        for name, weights in synthetic_adapters.items():
            all_loadings[name] = compute_adapter_loadings(components, weights)

        config = SHAREConfig(
            num_components=k,
            adapter_names=list(synthetic_adapters.keys()),
            num_adapters=len(synthetic_adapters),
            lora_rank=4,
        )
        adapter_configs = {name: synthetic_lora_config for name in synthetic_adapters}

        # Save
        out = tmp_path / "checkpoint"
        save_share_checkpoint(out, config, components, all_loadings, adapter_configs)

        # Load
        loaded_config, loaded_components, loaded_loadings, loaded_adapter_configs, loaded_classifier_heads = (
            load_share_checkpoint(out)
        )

        assert loaded_config.num_components == 8
        assert loaded_config.num_adapters == 3
        assert set(loaded_loadings.keys()) == {"cola", "mrpc", "rte"}
        for gk in components:
            assert torch.allclose(components[gk], loaded_components[gk])
        for name in all_loadings:
            for gk in all_loadings[name]:
                assert torch.allclose(
                    all_loadings[name][gk], loaded_loadings[name][gk]
                )

    def test_directory_structure(self, tmp_path, synthetic_adapters, synthetic_lora_config):
        combined = combine_adapter_weights(synthetic_adapters)
        components, _, k = compute_shared_components(combined, num_components=8)
        all_loadings = {
            name: compute_adapter_loadings(components, weights)
            for name, weights in synthetic_adapters.items()
        }
        config = SHAREConfig(num_components=k)
        adapter_configs = {name: synthetic_lora_config for name in synthetic_adapters}

        out = tmp_path / "checkpoint"
        save_share_checkpoint(out, config, components, all_loadings, adapter_configs)

        assert (out / "share_config.json").exists()
        assert (out / "shared_components.safetensors").exists()
        for name in synthetic_adapters:
            assert (out / "adapters" / name / "loadings.safetensors").exists()
            assert (out / "adapters" / name / "adapter_meta.json").exists()

    def test_classifier_heads_roundtrip(self, tmp_path, synthetic_adapters, synthetic_lora_config):
        """Test that classifier heads are saved and loaded correctly."""
        combined = combine_adapter_weights(synthetic_adapters)
        components, _, k = compute_shared_components(combined, num_components=8)
        all_loadings = {
            name: compute_adapter_loadings(components, weights)
            for name, weights in synthetic_adapters.items()
        }

        # Add classifier heads
        all_classifier_heads = {
            "cola": {"classifier.weight": torch.randn(2, 768), "classifier.bias": torch.randn(2)},
            "mrpc": {"classifier.weight": torch.randn(2, 768), "classifier.bias": torch.randn(2)},
        }

        config = SHAREConfig(num_components=k)
        adapter_configs = {name: synthetic_lora_config for name in synthetic_adapters}

        # Save
        out = tmp_path / "checkpoint"
        save_share_checkpoint(out, config, components, all_loadings, adapter_configs, all_classifier_heads)

        # Load
        _, _, _, _, loaded_classifier_heads = load_share_checkpoint(out)

        # Verify classifier heads are preserved
        assert "cola" in loaded_classifier_heads
        assert "mrpc" in loaded_classifier_heads
        assert "rte" not in loaded_classifier_heads  # rte has no classifier heads
        assert torch.allclose(loaded_classifier_heads["cola"]["classifier.weight"], all_classifier_heads["cola"]["classifier.weight"])
        assert torch.allclose(loaded_classifier_heads["cola"]["classifier.bias"], all_classifier_heads["cola"]["classifier.bias"])


# ── Reconstructed Adapter ────────────────────────────────────────────────────


class TestSaveReconstructedAdapter:
    def test_format(self, tmp_path, synthetic_lora_config):
        weights = {
            "base_model.model.encoder.layer.0.lora_A.weight": torch.randn(4, 64),
            "base_model.model.encoder.layer.0.lora_B.weight": torch.randn(64, 4),
        }
        out = tmp_path / "reconstructed"
        save_reconstructed_adapter(out, weights, synthetic_lora_config)

        assert (out / "adapter_config.json").exists()
        assert (out / "adapter_model.safetensors").exists()

        with open(out / "adapter_config.json") as f:
            config = json.load(f)
        assert config["peft_type"] == "LORA"

    def test_merges_classifier_heads(self, tmp_path, synthetic_lora_config):
        """Test that classifier heads are merged into the saved adapter."""
        from safetensors.torch import load_file
        weights = {
            "base_model.model.encoder.layer.0.lora_A.weight": torch.randn(4, 64),
            "base_model.model.encoder.layer.0.lora_B.weight": torch.randn(64, 4),
        }
        classifier_heads = {
            "classifier.weight": torch.randn(2, 768),
            "classifier.bias": torch.randn(2),
        }
        out = tmp_path / "reconstructed"
        save_reconstructed_adapter(out, weights, synthetic_lora_config, classifier_heads)

        # Load and verify all weights are present
        loaded = load_file(str(out / "adapter_model.safetensors"))
        assert len(loaded) == 4  # 2 LoRA + 2 classifier
        assert "base_model.model.encoder.layer.0.lora_A.weight" in loaded
        assert "classifier.weight" in loaded
        assert "classifier.bias" in loaded


# ── Config Round-Trip ────────────────────────────────────────────────────────


class TestConfigRoundTrip:
    def test_save_load(self, tmp_path):
        config = SHAREConfig(
            num_components=16,
            base_model_name_or_path="roberta-base",
            adapter_names=["cola", "mrpc"],
            num_adapters=2,
            lora_rank=8,
        )
        config.save(tmp_path)
        loaded = SHAREConfig.load(tmp_path)
        assert loaded.num_components == 16
        assert loaded.base_model_name_or_path == "roberta-base"
        assert loaded.adapter_names == ["cola", "mrpc"]
