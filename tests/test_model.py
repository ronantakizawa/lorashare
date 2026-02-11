"""Integration tests for SHAREModel."""

import json

import pytest
import torch
from safetensors.torch import save_file

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


class TestAddAdapter:
    def test_projection_path(self, saved_adapters):
        """Adding an adapter that fits the existing subspace uses fast path."""
        share = SHAREModel.from_adapters(
            {"cola": saved_adapters["cola"], "mrpc": saved_adapters["mrpc"]},
            num_components=8,
        )
        # Use a low threshold to force the projection path with random data
        result = share.add_adapter(
            saved_adapters["rte"], name="rte", variance_threshold=0.01,
        )

        assert result["method"] == "projection"
        assert "rte" in share.adapter_names
        assert share.config.num_adapters == 3
        # Reconstruction should produce valid LoRA keys
        recon = share.reconstruct("rte")
        assert len(recon) > 0
        for key in recon:
            layer, side = parse_lora_key(key)
            assert side in ("A", "B")

    def test_recompute_path(self, saved_adapters):
        """force_recompute=True triggers full recomputation."""
        share = SHAREModel.from_adapters(
            {"cola": saved_adapters["cola"], "mrpc": saved_adapters["mrpc"]},
            num_components=8,
        )
        result = share.add_adapter(
            saved_adapters["rte"], name="rte", force_recompute=True,
        )

        assert result["method"] == "recompute"
        assert "rte" in share.adapter_names
        assert share.config.num_adapters == 3
        # All adapters should reconstruct
        for name in share.adapter_names:
            recon = share.reconstruct(name)
            assert len(recon) > 0

    def test_preserves_existing_on_projection(self, saved_adapters):
        """Fast path must not change existing adapter reconstructions."""
        share = SHAREModel.from_adapters(
            {"cola": saved_adapters["cola"], "mrpc": saved_adapters["mrpc"]},
            num_components=8,
        )
        # Snapshot reconstructions before adding
        before = {
            name: share.reconstruct(name) for name in share.adapter_names
        }

        # Use a low threshold to force the projection path with random data
        share.add_adapter(
            saved_adapters["rte"], name="rte", variance_threshold=0.01,
        )

        for name, old_recon in before.items():
            new_recon = share.reconstruct(name)
            for key in old_recon:
                assert torch.allclose(old_recon[key], new_recon[key], atol=1e-6)

    def test_roundtrip_after_add(self, tmp_path, saved_adapters):
        """Save/load should preserve an added adapter."""
        share = SHAREModel.from_adapters(
            {"cola": saved_adapters["cola"], "mrpc": saved_adapters["mrpc"]},
            num_components=8,
        )
        share.add_adapter(saved_adapters["rte"], name="rte")

        ckpt = tmp_path / "checkpoint"
        share.save_pretrained(ckpt)

        loaded = SHAREModel.from_pretrained(ckpt)
        assert set(loaded.adapter_names) == {"cola", "mrpc", "rte"}
        for name in share.adapter_names:
            orig_recon = share.reconstruct(name)
            loaded_recon = loaded.reconstruct(name)
            for key in orig_recon:
                assert torch.allclose(orig_recon[key], loaded_recon[key], atol=1e-6)

    def test_duplicate_name_raises(self, saved_adapters):
        """Adding an adapter with a name that already exists should raise."""
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8,
        )
        with pytest.raises(ValueError, match="already exists"):
            share.add_adapter(saved_adapters["cola"], name="cola")

    def test_incompatible_rank_raises(self, tmp_path, saved_adapters, synthetic_lora_config):
        """Adding an adapter with a different rank should raise."""
        share = SHAREModel.from_adapters(
            {"cola": saved_adapters["cola"], "mrpc": saved_adapters["mrpc"]},
            num_components=8,
        )

        # Create adapter with rank 8 instead of 4
        bad_dir = tmp_path / "bad_rank"
        bad_dir.mkdir()
        weights = {
            "base_model.model.encoder.layer.0.attention.self.query.lora_A.weight": torch.randn(8, 64),
            "base_model.model.encoder.layer.0.attention.self.query.lora_B.weight": torch.randn(64, 8),
        }
        cfg = synthetic_lora_config.copy()
        cfg["r"] = 8
        save_file(weights, str(bad_dir / "adapter_model.safetensors"))
        with open(bad_dir / "adapter_config.json", "w") as f:
            json.dump(cfg, f)

        with pytest.raises(ValueError, match="Rank mismatch"):
            share.add_adapter(str(bad_dir), name="bad")


class TestRemoveAdapter:
    def test_basic_remove(self, saved_adapters):
        """Removing an adapter updates names and count."""
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8,
        )
        share.remove_adapter("cola")

        assert "cola" not in share.adapter_names
        assert share.config.num_adapters == 2
        assert set(share.adapter_names) == {"mrpc", "rte"}

    def test_remaining_adapters_reconstruct(self, saved_adapters):
        """Remaining adapters still reconstruct after removal."""
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8,
        )
        # Snapshot before removal
        before = share.reconstruct("mrpc")

        share.remove_adapter("cola")

        after = share.reconstruct("mrpc")
        for key in before:
            assert torch.allclose(before[key], after[key], atol=1e-6)

    def test_roundtrip_after_remove(self, tmp_path, saved_adapters):
        """Save/load works after removing an adapter."""
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8,
        )
        share.remove_adapter("rte")

        ckpt = tmp_path / "checkpoint"
        share.save_pretrained(ckpt)

        loaded = SHAREModel.from_pretrained(ckpt)
        assert set(loaded.adapter_names) == {"cola", "mrpc"}

    def test_missing_adapter_raises(self, saved_adapters):
        """Removing a nonexistent adapter raises KeyError."""
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8,
        )
        with pytest.raises(KeyError, match="nonexistent"):
            share.remove_adapter("nonexistent")

    def test_remove_last_raises(self, saved_adapters):
        """Cannot remove the last remaining adapter."""
        share = SHAREModel.from_adapters(
            {"cola": saved_adapters["cola"], "mrpc": saved_adapters["mrpc"]},
            num_components=8,
        )
        share.remove_adapter("cola")
        with pytest.raises(ValueError, match="last adapter"):
            share.remove_adapter("mrpc")


class TestOnError:
    def test_skip_corrupted_adapter(self, tmp_path, saved_adapters):
        """on_error='skip' skips a corrupted adapter and compresses the rest."""
        # Create a corrupted adapter directory (missing weights file)
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')
        # No weights file → will raise FileNotFoundError

        paths = [saved_adapters["cola"], str(bad_dir), saved_adapters["mrpc"]]
        share = SHAREModel.from_adapters(paths, num_components=4, on_error="skip")

        assert share.config.num_adapters == 2
        assert set(share.adapter_names) == {"cola", "mrpc"}

    def test_skip_insufficient_raises(self, tmp_path, saved_adapters):
        """on_error='skip' still raises if fewer than 2 adapters remain."""
        bad1 = tmp_path / "bad1"
        bad1.mkdir()
        (bad1 / "adapter_config.json").write_text('{"peft_type": "LORA"}')
        bad2 = tmp_path / "bad2"
        bad2.mkdir()
        (bad2 / "adapter_config.json").write_text('{"peft_type": "LORA"}')

        paths = [saved_adapters["cola"], str(bad1), str(bad2)]
        with pytest.raises(ValueError, match="at least 2"):
            SHAREModel.from_adapters(paths, num_components=4, on_error="skip")

    def test_raise_is_default(self, tmp_path, saved_adapters):
        """Default on_error='raise' aborts on any failure."""
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')

        paths = [saved_adapters["cola"], str(bad_dir), saved_adapters["mrpc"]]
        with pytest.raises(FileNotFoundError):
            SHAREModel.from_adapters(paths, num_components=4)

    def test_skip_with_dict_input(self, tmp_path, saved_adapters):
        """on_error='skip' works with dict-style adapter input too."""
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')

        adapter_dict = {
            "cola": saved_adapters["cola"],
            "bad": str(bad_dir),
            "mrpc": saved_adapters["mrpc"],
        }
        share = SHAREModel.from_adapters(
            adapter_dict, num_components=4, on_error="skip",
        )

        assert share.config.num_adapters == 2
        assert set(share.adapter_names) == {"cola", "mrpc"}


class TestSummary:
    def test_no_crash(self, saved_adapters, capsys):
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()), num_components=8
        )
        share.summary()
        captured = capsys.readouterr()
        assert "SHARE Compression Summary" in captured.out
        assert "roberta-base" in captured.out
