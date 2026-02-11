"""
Quick Start: Compress multiple LoRA adapters with lorashare

This example shows the basic workflow:
1. Load multiple LoRA adapters
2. Compress them into a shared subspace
3. Reconstruct and use them for inference
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lorashare import SHAREModel


def main():
    print("=" * 60)
    print("lorashare Quick Start")
    print("=" * 60)

    # Example adapter paths (replace with your own)
    adapter_paths = [
        "path/to/cola_lora",
        "path/to/mrpc_lora",
        "path/to/rte_lora",
    ]

    # You can also use HuggingFace Hub IDs:
    # adapter_paths = [
    #     "username/cola_lora",
    #     "username/mrpc_lora",
    #     "username/rte_lora",
    # ]

    print("\n1. Compressing adapters...")
    print(f"   Loading {len(adapter_paths)} adapters")

    share = SHAREModel.from_adapters(
        adapter_paths,
        num_components=32,  # or "auto" for automatic selection
    )

    print("\n2. Compression summary:")
    share.summary()

    print("\n3. Saving compressed checkpoint...")
    share.save_pretrained("./my_compressed_adapters")
    print("   ✓ Saved to ./my_compressed_adapters")

    print("\n4. Loading from checkpoint...")
    loaded_share = SHAREModel.from_pretrained("./my_compressed_adapters")
    print(f"   ✓ Loaded {loaded_share.config.num_adapters} adapters")

    print("\n5. Reconstructing adapter for inference...")
    # Reconstruct the first adapter
    adapter_name = loaded_share.adapter_names[0]
    print(f"   Reconstructing: {adapter_name}")

    # Option A: Just get the weights
    weights = loaded_share.reconstruct(adapter_name)
    print(f"   ✓ Reconstructed {len(weights)} weight tensors")

    # Option B: Save as standard PEFT adapter
    loaded_share.reconstruct(adapter_name, output_dir=f"./reconstructed_{adapter_name}")
    print(f"   ✓ Saved to ./reconstructed_{adapter_name}")

    print("\n6. Using reconstructed adapter for inference...")
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,
    )
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Apply adapter to base model
    model = loaded_share.apply(base_model, adapter_name)
    model.eval()

    # Run inference
    text = "This is a test sentence."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)

    print(f"   ✓ Inference successful!")
    print(f"   Predictions: {predictions[0].tolist()}")

    print("\n" + "=" * 60)
    print("✓ Quick start complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Note: This example requires adapter paths to exist.
    # For a working demo, see create_and_compress.py
    print("\nNote: Update adapter_paths with your own adapters to run this example.")
    print("Or run create_and_compress.py for a full working demo.\n")
