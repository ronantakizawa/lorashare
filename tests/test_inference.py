"""Integration test: Verify reconstructed adapters work for inference."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lorashare import SHAREModel


def create_realistic_adapter(
    path: Path,
    task_name: str,
    hidden_size: int = 768,
    num_labels: int = 2,
    rank: int = 8,
    num_layers: int = 2,
):
    """Create a realistic PEFT LoRA adapter with classifier heads."""
    path.mkdir(parents=True, exist_ok=True)

    weights = {}

    # Create LoRA weights for multiple layers
    for layer_idx in range(num_layers):
        for module in ["query", "value", "key"]:
            # A matrices: (rank, hidden_size)
            weights[f"base_model.model.encoder.layer.{layer_idx}.attention.self.{module}.lora_A.weight"] = (
                torch.randn(rank, hidden_size) * 0.01
            )
            # B matrices: (hidden_size, rank)
            weights[f"base_model.model.encoder.layer.{layer_idx}.attention.self.{module}.lora_B.weight"] = (
                torch.randn(hidden_size, rank) * 0.01
            )

    # Add task-specific classifier head (this is what makes each adapter unique for inference)
    weights["classifier.weight"] = torch.randn(num_labels, hidden_size) * 0.1
    weights["classifier.bias"] = torch.randn(num_labels) * 0.1

    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "roberta-base",
        "r": rank,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["query", "value", "key"],
        "task_type": "SEQ_CLS",
        "inference_mode": False,
    }

    save_file(weights, str(path / "adapter_model.safetensors"))
    with open(path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return weights


class TestInferenceWithMultipleAdapters:
    """Test that compressed adapters work for actual inference."""

    def test_classifier_outputs_preserved(self, tmp_path):
        """Verify that reconstructed adapters produce outputs (classifier heads work)."""
        # Create 3 different adapters for different tasks
        cola_path = tmp_path / "cola"
        mrpc_path = tmp_path / "mrpc"
        rte_path = tmp_path / "rte"

        cola_weights = create_realistic_adapter(cola_path, "cola", num_labels=2)
        mrpc_weights = create_realistic_adapter(mrpc_path, "mrpc", num_labels=2)
        rte_weights = create_realistic_adapter(rte_path, "rte", num_labels=2)

        # Compress all adapters
        share = SHAREModel.from_adapters(
            [str(cola_path), str(mrpc_path), str(rte_path)],
            num_components=16,
        )

        # Test reconstruction for each adapter
        for adapter_name, original_weights in [
            ("cola", cola_weights),
            ("mrpc", mrpc_weights),
            ("rte", rte_weights),
        ]:
            reconstructed = share.reconstruct(adapter_name)

            # Verify classifier heads are present
            assert "classifier.weight" in reconstructed
            assert "classifier.bias" in reconstructed

            # Verify classifier shape matches
            assert reconstructed["classifier.weight"].shape == original_weights["classifier.weight"].shape
            assert reconstructed["classifier.bias"].shape == original_weights["classifier.bias"].shape

    def test_different_classifiers_for_different_tasks(self, tmp_path):
        """Verify each adapter gets its own unique classifier head."""
        # Create adapters with different classifier dimensions
        binary_path = tmp_path / "binary_task"
        multi_path = tmp_path / "multi_task"

        create_realistic_adapter(binary_path, "binary", num_labels=2)
        create_realistic_adapter(multi_path, "multi", num_labels=5)

        share = SHAREModel.from_adapters(
            [str(binary_path), str(multi_path)],
            num_components=8,
        )

        # Reconstruct and verify each has correct classifier
        binary_recon = share.reconstruct("binary_task")
        multi_recon = share.reconstruct("multi_task")

        assert binary_recon["classifier.weight"].shape[0] == 2  # 2 labels
        assert multi_recon["classifier.weight"].shape[0] == 5  # 5 labels

    def test_classifier_values_preserved_exactly(self, tmp_path):
        """Verify classifier head values are not modified during compression."""
        adapter_path = tmp_path / "test_adapter"
        original_weights = create_realistic_adapter(adapter_path, "test")

        # Extract original classifier values
        orig_classifier_weight = original_weights["classifier.weight"].clone()
        orig_classifier_bias = original_weights["classifier.bias"].clone()

        # Compress and reconstruct
        share = SHAREModel.from_adapters(
            [str(adapter_path)] * 3,  # Use same adapter 3 times to test
            num_components=8,
        )

        reconstructed = share.reconstruct("test_adapter")

        # Classifier heads should be EXACTLY the same (no compression applied)
        assert torch.allclose(
            reconstructed["classifier.weight"],
            orig_classifier_weight,
            atol=1e-7,
        ), "Classifier weight should be preserved exactly"

        assert torch.allclose(
            reconstructed["classifier.bias"],
            orig_classifier_bias,
            atol=1e-7,
        ), "Classifier bias should be preserved exactly"

    def test_reconstructed_adapter_can_be_saved_and_loaded(self, tmp_path):
        """Verify reconstructed adapter can be saved in standard PEFT format."""
        # Need at least 2 adapters for compression
        adapter_path1 = tmp_path / "original1"
        adapter_path2 = tmp_path / "original2"
        create_realistic_adapter(adapter_path1, "test")
        create_realistic_adapter(adapter_path2, "test2")

        share = SHAREModel.from_adapters(
            [str(adapter_path1), str(adapter_path2)], num_components=8
        )

        # Reconstruct and save (adapter name is derived from path: "original1")
        output_path = tmp_path / "reconstructed"
        share.reconstruct("original1", output_dir=output_path)

        # Verify standard PEFT files exist
        assert (output_path / "adapter_config.json").exists()
        assert (output_path / "adapter_model.safetensors").exists()

        # Verify config is valid
        with open(output_path / "adapter_config.json") as f:
            config = json.load(f)
        assert config["peft_type"] == "LORA"
        assert config["r"] == 8

        # Verify weights can be loaded
        from safetensors.torch import load_file
        loaded_weights = load_file(str(output_path / "adapter_model.safetensors"))

        # Should have both LoRA and classifier keys
        lora_keys = [k for k in loaded_weights if "lora_" in k]
        classifier_keys = [k for k in loaded_weights if "classifier" in k]

        assert len(lora_keys) > 0, "Should have LoRA weights"
        assert len(classifier_keys) > 0, "Should have classifier heads"

    def test_multiple_adapters_have_independent_classifiers(self, tmp_path):
        """Verify different adapters maintain their unique classifier heads."""
        # Create 3 adapters with distinctive classifier values
        adapters = {}
        for i, name in enumerate(["task_a", "task_b", "task_c"]):
            path = tmp_path / name
            weights = create_realistic_adapter(path, name)

            # Make each classifier distinctive by scaling
            weights["classifier.weight"] = weights["classifier.weight"] * (i + 1)
            weights["classifier.bias"] = weights["classifier.bias"] * (i + 1)

            # Re-save with distinctive values
            save_file(weights, str(path / "adapter_model.safetensors"))
            adapters[name] = weights

        # Compress all together
        share = SHAREModel.from_adapters(
            [str(tmp_path / name) for name in adapters.keys()],
            num_components=8,
        )

        # Verify each reconstructed adapter has its own classifier
        for name, original in adapters.items():
            reconstructed = share.reconstruct(name)

            # Classifier should match original exactly
            assert torch.allclose(
                reconstructed["classifier.weight"],
                original["classifier.weight"],
                atol=1e-6,
            ), f"{name}: classifier weight should be preserved"

            # Verify they're different from each other
            for other_name, other_original in adapters.items():
                if other_name != name:
                    # Should NOT be close to other adapters' classifiers
                    assert not torch.allclose(
                        reconstructed["classifier.weight"],
                        other_original["classifier.weight"],
                        atol=0.1,
                    ), f"{name} classifier should differ from {other_name}"

    def test_lora_compressed_but_classifier_not(self, tmp_path):
        """Verify LoRA weights are compressed but classifier heads are stored verbatim."""
        # Need at least 2 adapters for compression
        adapter_path1 = tmp_path / "test1"
        adapter_path2 = tmp_path / "test2"
        original_weights = create_realistic_adapter(adapter_path1, "test")
        create_realistic_adapter(adapter_path2, "test2")

        share = SHAREModel.from_adapters(
            [str(adapter_path1), str(adapter_path2)], num_components=4
        )

        # Save checkpoint and inspect
        checkpoint_path = tmp_path / "checkpoint"
        share.save_pretrained(checkpoint_path)

        # Classifier heads should be in separate file
        classifier_file = checkpoint_path / "adapters" / "test1" / "classifier_head.safetensors"
        assert classifier_file.exists(), "Classifier heads should be saved separately"

        from safetensors.torch import load_file
        saved_classifier = load_file(str(classifier_file))

        # Should contain exact original classifier values
        assert "classifier.weight" in saved_classifier
        assert "classifier.bias" in saved_classifier
        assert torch.allclose(
            saved_classifier["classifier.weight"],
            original_weights["classifier.weight"],
            atol=1e-7,
        )


class TestInferenceRoundTrip:
    """Test full round-trip: compress -> save -> load -> reconstruct -> inference."""

    def test_full_pipeline(self, tmp_path):
        """Test complete workflow with multiple adapters."""
        # Step 1: Create original adapters
        adapters = {}
        for name in ["cola", "mrpc", "rte"]:
            path = tmp_path / "original" / name
            weights = create_realistic_adapter(path, name, num_labels=2)
            adapters[name] = {
                "path": path,
                "weights": weights,
            }

        # Step 2: Compress
        share = SHAREModel.from_adapters(
            [str(info["path"]) for info in adapters.values()],
            num_components=16,
        )

        # Step 3: Save checkpoint
        checkpoint_path = tmp_path / "checkpoint"
        share.save_pretrained(checkpoint_path)

        # Step 4: Load from checkpoint
        loaded_share = SHAREModel.from_pretrained(checkpoint_path)

        # Step 5: Reconstruct each adapter and verify
        for name, info in adapters.items():
            # Reconstruct
            reconstructed = loaded_share.reconstruct(name)

            # Verify structure
            assert len(reconstructed) > 0
            assert "classifier.weight" in reconstructed
            assert "classifier.bias" in reconstructed

            # Verify classifier preserved exactly
            assert torch.allclose(
                reconstructed["classifier.weight"],
                info["weights"]["classifier.weight"],
                atol=1e-6,
            )

            # Save reconstructed adapter
            output_path = tmp_path / "reconstructed" / name
            loaded_share.reconstruct(name, output_dir=output_path)

            # Verify can be loaded as standard PEFT adapter
            assert (output_path / "adapter_config.json").exists()
            assert (output_path / "adapter_model.safetensors").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
