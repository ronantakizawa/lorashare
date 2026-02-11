"""Integration tests for SHAREModel."""

import json

import pytest
import torch

from lorashare import SHAREModel
from lorashare.compression import parse_lora_key


class TestFromAdapters:
    def test_basic(self, saved_adapters):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )
        assert set(share.adapter_names) == {"cola", "mrpc", "rte"}
        assert share.config.num_components == 8
        assert share.config.num_adapters == 3

    def test_with_dict(self, saved_adapters):
        share = SHAREModel.from_adapters(
            {"a": saved_adapters["cola"], "b": saved_adapters["mrpc"]},
            num_components=4,
        )
        assert set(share.adapter_names) == {"a", "b"}

    def test_auto_components(self, saved_adapters):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components="auto",
            variance_threshold=0.95,
        )
        assert share.config.num_components > 0
        assert share.config.component_selection == "auto"

    def test_incompatible_raises(self, tmp_path, synthetic_lora_config):
        """Adapters with different ranks should raise."""
        from safetensors.torch import save_file

        # Create two adapters with different ranks
        for name, rank in [("a", 4), ("b", 8)]:
            d = tmp_path / name
            d.mkdir()
            weights = {
                "base_model.model.encoder.layer.0.attention.self.query.lora_A.weight": torch.randn(rank, 64),
                "base_model.model.encoder.layer.0.attention.self.query.lora_B.weight": torch.randn(64, rank),
            }
            config = synthetic_lora_config.copy()
            config["r"] = rank
            save_file(weights, str(d / "adapter_model.safetensors"))
            with open(d / "adapter_config.json", "w") as f:
                json.dump(config, f)

        with pytest.raises(ValueError, match="Rank mismatch"):
            SHAREModel.from_adapters(
                {"a": str(tmp_path / "a"), "b": str(tmp_path / "b")}
            )

    def test_config_has_stats(self, saved_adapters):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )
        stats = share.config.compression_stats
        assert "original_total_params" in stats
        assert "compressed_total_params" in stats
        assert "compression_ratio" in stats
        assert stats["compression_ratio"] > 0


class TestSaveLoadRoundTrip:
    def test_roundtrip(self, tmp_path, saved_adapters):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )

        ckpt = tmp_path / "checkpoint"
        share.save_pretrained(ckpt)

        loaded = SHAREModel.from_pretrained(ckpt)
        assert loaded.config.num_components == share.config.num_components
        assert set(loaded.adapter_names) == set(share.adapter_names)

        # Verify reconstruction matches
        for name in share.adapter_names:
            orig_recon = share.reconstruct(name)
            loaded_recon = loaded.reconstruct(name)
            for key in orig_recon:
                assert torch.allclose(orig_recon[key], loaded_recon[key], atol=1e-6)


class TestReconstruct:
    def test_returns_valid_peft_keys(self, saved_adapters):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )
        weights = share.reconstruct("cola")
        # Check that all LoRA keys are valid
        lora_keys = [k for k in weights if "lora_" in k]
        for key in lora_keys:
            layer, side = parse_lora_key(key)
            assert side in ("A", "B")

    def test_saves_to_disk(self, tmp_path, saved_adapters):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )
        out = tmp_path / "reconstructed"
        share.reconstruct("cola", output_dir=out)

        assert (out / "adapter_config.json").exists()
        assert (out / "adapter_model.safetensors").exists()

        with open(out / "adapter_config.json") as f:
            config = json.load(f)
        assert config["peft_type"] == "LORA"

    def test_missing_adapter_raises(self, saved_adapters):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )
        with pytest.raises(KeyError, match="nonexistent"):
            share.reconstruct("nonexistent")


class TestReconstructionError:
    def test_returns_metrics(self, saved_adapters, synthetic_adapters):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )
        error = share.reconstruction_error(
            "cola", original_weights=synthetic_adapters["cola"]
        )
        assert "mean" in error
        assert "max" in error
        assert "per_layer" in error
        assert error["mean"] >= 0

    def test_from_path(self, saved_adapters):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )
        error = share.reconstruction_error(
            "cola", original_path=saved_adapters["cola"]
        )
        assert error["mean"] >= 0

    def test_no_source_raises(self, saved_adapters):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )
        with pytest.raises(ValueError, match="Must provide"):
            share.reconstruction_error("cola")


class TestSummary:
    def test_no_crash(self, saved_adapters, capsys):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )
        share.summary()
        captured = capsys.readouterr()
        assert "SHARE Compression Summary" in captured.out
        assert "roberta-base" in captured.out
