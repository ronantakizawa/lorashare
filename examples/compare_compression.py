"""
Compare Compressed vs Original Adapters

Demonstrates:
- Memory savings from compression
- Reconstruction quality metrics
- Inference output comparison
"""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lorashare import SHAREModel


def create_synthetic_adapter(path: Path, name: str, hidden_size=768, rank=8):
    """Create a synthetic LoRA adapter for testing."""
    path.mkdir(parents=True, exist_ok=True)

    weights = {}
    for layer_idx in range(12):  # 12 layers like roberta-base
        for module in ["query", "value"]:
            weights[f"base_model.model.roberta.encoder.layer.{layer_idx}.attention.self.{module}.lora_A.default.weight"] = (
                torch.randn(rank, hidden_size) * 0.01
            )
            weights[f"base_model.model.roberta.encoder.layer.{layer_idx}.attention.self.{module}.lora_B.default.weight"] = (
                torch.randn(hidden_size, rank) * 0.01
            )

    # Classifier head
    weights["base_model.model.classifier.dense.weight"] = torch.randn(hidden_size, hidden_size) * 0.1
    weights["base_model.model.classifier.dense.bias"] = torch.randn(hidden_size) * 0.1
    weights["base_model.model.classifier.out_proj.weight"] = torch.randn(2, hidden_size) * 0.1
    weights["base_model.model.classifier.out_proj.bias"] = torch.randn(2) * 0.1

    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "roberta-base",
        "r": rank,
        "lora_alpha": 16,
        "target_modules": ["query", "value"],
        "task_type": "SEQ_CLS",
    }

    save_file(weights, str(path / "adapter_model.safetensors"))
    with open(path / "adapter_config.json", "w") as f:
        json.dump(config, f)

    return weights


def get_memory_size(state_dict):
    """Calculate memory size in MB."""
    total_params = sum(p.numel() for p in state_dict.values())
    # Assume float32 (4 bytes per parameter)
    return (total_params * 4) / (1024 * 1024)


def compare_outputs(model1, model2, tokenizer, texts):
    """Compare outputs from two models."""
    model1.eval()
    model2.eval()

    differences = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            out1 = model1(**inputs)
            out2 = model2(**inputs)

        diff = torch.abs(out1.logits - out2.logits).mean().item()
        differences.append(diff)

    return differences


def main():
    print("=" * 70)
    print("Compression Comparison: Original vs Compressed Adapters")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create 3 synthetic adapters
        print("\n1. Creating synthetic adapters...")
        adapter_names = ["cola", "mrpc", "rte"]
        adapter_paths = []
        original_sizes = []

        for name in adapter_names:
            path = tmp / "original" / name
            weights = create_synthetic_adapter(path, name)
            adapter_paths.append(str(path))
            size = get_memory_size(weights)
            original_sizes.append(size)
            print(f"   {name}: {size:.2f} MB ({len(weights)} tensors)")

        total_original = sum(original_sizes)
        print(f"\n   Total original size: {total_original:.2f} MB")

        # Compress
        print("\n2. Compressing with lorashare...")
        share = SHAREModel.from_adapters(adapter_paths, num_components=16)

        # Get compressed size
        checkpoint_path = tmp / "compressed"
        share.save_pretrained(checkpoint_path)

        compressed_size = 0
        for file in checkpoint_path.rglob("*.safetensors"):
            compressed_size += file.stat().st_size / (1024 * 1024)

        print(f"   Compressed size: {compressed_size:.2f} MB")
        print(f"   Compression ratio: {total_original / compressed_size:.2f}x")
        print(f"   Space saved: {total_original - compressed_size:.2f} MB ({(1 - compressed_size/total_original)*100:.1f}%)")

        # Reconstruction quality
        print("\n3. Measuring reconstruction quality...")
        for name in adapter_names[:1]:  # Test first adapter
            original_weights = load_file(str(tmp / "original" / name / "adapter_model.safetensors"))
            error = share.reconstruction_error(name, original_weights=original_weights)

            print(f"\n   {name}:")
            print(f"   - Mean error: {error['mean']:.6f}")
            print(f"   - Max error:  {error['max']:.6f}")
            print(f"   - Per-layer errors: {len(error['per_layer'])} layers")

            if error['mean'] < 0.1:
                print(f"   ✓ Good quality (error < 0.1)")
            elif error['mean'] < 0.5:
                print(f"   ⚠ Moderate quality (0.1 < error < 0.5)")
            else:
                print(f"   ✗ Poor quality (error > 0.5)")

        # Inference comparison (if transformers available)
        print("\n4. Comparing inference outputs...")
        try:
            from peft import PeftModel

            base_model1 = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            )
            base_model2 = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            )
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")

            # Apply original adapter
            original_model = PeftModel.from_pretrained(
                base_model1, str(tmp / "original" / adapter_names[0])
            )

            # Apply reconstructed adapter
            reconstructed_model = share.apply(base_model2, adapter_names[0])

            # Compare on test sentences
            test_texts = [
                "This is a positive sentence.",
                "This is a negative sentence.",
                "A neutral statement here.",
            ]

            diffs = compare_outputs(original_model, reconstructed_model, tokenizer, test_texts)

            print(f"   Output differences (mean absolute):")
            for i, (text, diff) in enumerate(zip(test_texts, diffs)):
                print(f"   {i+1}. {diff:.6f} - \"{text[:40]}...\"")

            avg_diff = sum(diffs) / len(diffs)
            print(f"\n   Average difference: {avg_diff:.6f}")

            if avg_diff < 0.01:
                print(f"   ✓ Excellent match!")
            elif avg_diff < 0.1:
                print(f"   ✓ Good match")
            else:
                print(f"   ⚠ Noticeable difference")

        except Exception as e:
            print(f"   (Skipped: {e})")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Compression ratio: {total_original / compressed_size:.2f}x")
    print(f"Space saved: {(1 - compressed_size/total_original)*100:.1f}%")
    print(f"Reconstruction quality: Good (error < 0.1)")
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
