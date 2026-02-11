"""
Real inference test with actual transformers models.
Tests that reconstructed adapters can generate predictions.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lorashare import SHAREModel


def create_peft_lora_adapter(
    path: Path,
    hidden_size: int = 768,
    num_labels: int = 2,
    rank: int = 8,
    num_layers: int = 12,
):
    """Create a PEFT LoRA adapter compatible with roberta-base."""
    path.mkdir(parents=True, exist_ok=True)

    weights = {}

    # Create LoRA weights for all roberta layers
    for layer_idx in range(num_layers):
        for module in ["query", "value"]:
            # A matrices: (rank, hidden_size)
            weights[f"base_model.model.roberta.encoder.layer.{layer_idx}.attention.self.{module}.lora_A.default.weight"] = (
                torch.randn(rank, hidden_size) * 0.01
            )
            # B matrices: (hidden_size, rank)
            weights[f"base_model.model.roberta.encoder.layer.{layer_idx}.attention.self.{module}.lora_B.default.weight"] = (
                torch.randn(hidden_size, rank) * 0.01
            )

    # Add classifier head
    weights["base_model.model.classifier.dense.weight"] = torch.randn(hidden_size, hidden_size) * 0.1
    weights["base_model.model.classifier.dense.bias"] = torch.randn(hidden_size) * 0.1
    weights["base_model.model.classifier.out_proj.weight"] = torch.randn(num_labels, hidden_size) * 0.1
    weights["base_model.model.classifier.out_proj.bias"] = torch.randn(num_labels) * 0.1

    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "roberta-base",
        "r": rank,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["query", "value"],
        "task_type": "SEQ_CLS",
        "inference_mode": False,
        "modules_to_save": ["classifier"],
    }

    save_file(weights, str(path / "adapter_model.safetensors"))
    with open(path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return weights


@pytest.mark.slow
class TestRealModelInference:
    """Test with actual transformers models."""

    def test_reconstructed_adapter_forward_pass(self, tmp_path):
        """Test that reconstructed adapter can do forward pass on real model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from peft import PeftModel
        except ImportError:
            pytest.skip("transformers and peft not installed")

        # Create synthetic adapters
        adapter1_path = tmp_path / "adapter1"
        adapter2_path = tmp_path / "adapter2"
        adapter3_path = tmp_path / "adapter3"

        create_peft_lora_adapter(adapter1_path, num_labels=2)
        create_peft_lora_adapter(adapter2_path, num_labels=2)
        create_peft_lora_adapter(adapter3_path, num_labels=2)

        # Compress with lorashare
        share = SHAREModel.from_adapters(
            [str(adapter1_path), str(adapter2_path), str(adapter3_path)],
            num_components=16,
        )

        # Reconstruct adapter1
        reconstructed_path = tmp_path / "reconstructed"
        share.reconstruct("adapter1", output_dir=reconstructed_path)

        # Load base model (use tiny model for speed)
        print("\nLoading base model...")
        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=2,
            )
        except Exception as e:
            pytest.skip(f"Could not load roberta-base: {e}")

        # Try to apply reconstructed adapter
        print("Applying reconstructed adapter...")
        try:
            model = PeftModel.from_pretrained(base_model, str(reconstructed_path))
        except Exception as e:
            pytest.fail(f"Failed to load reconstructed adapter: {e}")

        # Run forward pass
        print("Running forward pass...")
        model.eval()
        with torch.no_grad():
            # Create dummy input
            batch_size = 2
            seq_len = 16
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)

            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as e:
                pytest.fail(f"Forward pass failed: {e}")

            # Verify output structure
            assert hasattr(outputs, "logits"), "Output should have logits"
            assert outputs.logits.shape == (batch_size, 2), f"Expected shape (2, 2), got {outputs.logits.shape}"

        print("✓ Forward pass successful!")

    def test_apply_method_works(self, tmp_path):
        """Test that SHAREModel.apply() method works for inference."""
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError:
            pytest.skip("transformers not installed")

        # Create adapters
        adapter1_path = tmp_path / "adapter1"
        adapter2_path = tmp_path / "adapter2"

        create_peft_lora_adapter(adapter1_path, num_labels=2)
        create_peft_lora_adapter(adapter2_path, num_labels=2)

        # Compress
        share = SHAREModel.from_adapters(
            [str(adapter1_path), str(adapter2_path)],
            num_components=8,
        )

        # Load base model
        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=2,
            )
        except Exception:
            pytest.skip("Could not load roberta-base")

        # Use apply() method
        print("\nTesting apply() method...")
        try:
            model = share.apply(base_model, "adapter1")
        except Exception as e:
            pytest.fail(f"apply() method failed: {e}")

        # Run inference
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(0, 1000, (1, 10))
            attention_mask = torch.ones(1, 10)

            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as e:
                pytest.fail(f"Inference with apply() failed: {e}")

            assert outputs.logits.shape == (1, 2)

        print("✓ apply() method works!")

    def test_different_adapters_give_different_outputs(self, tmp_path):
        """Verify different reconstructed adapters produce different predictions."""
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError:
            pytest.skip("transformers not installed")

        # Create 2 adapters with very different classifier weights
        adapter1_path = tmp_path / "adapter1"
        adapter2_path = tmp_path / "adapter2"

        weights1 = create_peft_lora_adapter(adapter1_path, num_labels=2)
        weights2 = create_peft_lora_adapter(adapter2_path, num_labels=2)

        # Make classifier weights very different
        weights2["base_model.model.classifier.out_proj.weight"] *= 10
        save_file(weights2, str(adapter2_path / "adapter_model.safetensors"))

        # Compress
        share = SHAREModel.from_adapters(
            [str(adapter1_path), str(adapter2_path)],
            num_components=8,
        )

        # Load base model
        try:
            base_model1 = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            )
            base_model2 = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            )
        except Exception:
            pytest.skip("Could not load roberta-base")

        # Apply different adapters
        model1 = share.apply(base_model1, "adapter1")
        model2 = share.apply(base_model2, "adapter2")

        # Same input
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            out1 = model1(input_ids=input_ids, attention_mask=attention_mask)
            out2 = model2(input_ids=input_ids, attention_mask=attention_mask)

        # Outputs should be different (since classifiers are different)
        assert not torch.allclose(out1.logits, out2.logits, atol=0.1), (
            "Different adapters should produce different outputs"
        )

        print("✓ Different adapters produce different outputs!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
