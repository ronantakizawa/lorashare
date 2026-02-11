"""
Working with HuggingFace Hub

Demonstrates:
- Loading adapters from Hub
- Compressing Hub adapters
- Pushing compressed checkpoint to Hub
- Loading compressed checkpoint from Hub (when supported)
"""

from lorashare import SHAREModel


def example_compress_hub_adapters():
    """Load adapters from Hub and compress them."""
    print("=" * 60)
    print("Example 1: Compress Adapters from HuggingFace Hub")
    print("=" * 60)

    # Example Hub IDs (replace with real adapters)
    hub_adapters = [
        "username/cola_lora_r8",
        "username/mrpc_lora_r8",
        "username/rte_lora_r8",
    ]

    print("\n1. Loading adapters from Hub...")
    print(f"   Adapters: {hub_adapters}")

    try:
        share = SHAREModel.from_adapters(
            hub_adapters,
            num_components="auto",  # Automatic component selection
            variance_threshold=0.95,  # 95% variance explained
        )

        print("\n2. Compression complete!")
        share.summary()

        print("\n3. Saving locally...")
        share.save_pretrained("./compressed_from_hub")
        print("   ✓ Saved to ./compressed_from_hub")

        return share

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: Replace hub_adapters with real HuggingFace adapter IDs")
        return None


def example_push_to_hub():
    """Push compressed checkpoint to Hub."""
    print("\n" + "=" * 60)
    print("Example 2: Push Compressed Checkpoint to Hub")
    print("=" * 60)

    # Assuming you have a compressed model
    print("\n1. Loading local checkpoint...")
    try:
        share = SHAREModel.from_pretrained("./compressed_from_hub")

        print("\n2. Pushing to HuggingFace Hub...")
        # You need to be logged in: huggingface-cli login
        hub_url = share.push_to_hub(
            "username/my-compressed-adapters",
            # token="hf_...",  # Optional: provide token
            # private=True,    # Optional: make repo private
        )

        print(f"   ✓ Uploaded to: {hub_url}")
        print("\n3. Now others can use:")
        print(f"   share = SHAREModel.from_pretrained('username/my-compressed-adapters')")

    except FileNotFoundError:
        print("\n✗ No local checkpoint found")
        print("   Run example 1 first, or create a checkpoint")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: Make sure you're logged in with: huggingface-cli login")


def example_mixed_sources():
    """Compress adapters from both local and Hub."""
    print("\n" + "=" * 60)
    print("Example 3: Mix Local and Hub Adapters")
    print("=" * 60)

    adapters = {
        "cola": "username/cola_lora",      # From Hub
        "mrpc": "./local_adapters/mrpc",   # From local
        "rte": "username/rte_lora",        # From Hub
    }

    print("\n1. Compressing mixed sources...")
    print("   Sources:")
    for name, path in adapters.items():
        source = "Hub" if not path.startswith("./") else "Local"
        print(f"   - {name}: {path} ({source})")

    try:
        share = SHAREModel.from_adapters(adapters, num_components=32)

        print("\n2. Compression complete!")
        print(f"   Compressed {share.config.num_adapters} adapters")

        return share

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None


def example_reconstruct_and_push():
    """Reconstruct adapter and push to Hub as standard PEFT."""
    print("\n" + "=" * 60)
    print("Example 4: Reconstruct and Push Individual Adapter")
    print("=" * 60)

    try:
        share = SHAREModel.from_pretrained("./compressed_from_hub")

        adapter_name = share.adapter_names[0]
        print(f"\n1. Reconstructing: {adapter_name}")

        # Reconstruct to local directory
        output_dir = f"./reconstructed_{adapter_name}"
        share.reconstruct(adapter_name, output_dir=output_dir)
        print(f"   ✓ Saved to {output_dir}")

        print("\n2. Now you can push the reconstructed adapter:")
        print(f"   from huggingface_hub import HfApi")
        print(f"   api = HfApi()")
        print(f"   api.upload_folder(")
        print(f"       folder_path='{output_dir}',")
        print(f"       repo_id='username/{adapter_name}_reconstructed',")
        print(f"   )")

    except Exception as e:
        print(f"\n✗ Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HuggingFace Hub Integration Examples")
    print("=" * 60)
    print("\nNote: These examples require:")
    print("1. Valid HuggingFace Hub adapter IDs")
    print("2. HuggingFace authentication: huggingface-cli login")
    print("3. Network connection to download/upload")
    print("\n" + "=" * 60)

    # Run examples
    share = example_compress_hub_adapters()

    if share:
        example_push_to_hub()

    example_mixed_sources()
    example_reconstruct_and_push()

    print("\n" + "=" * 60)
    print("✓ Hub examples complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Replace 'username/...' with real Hub IDs")
    print("2. Login with: huggingface-cli login")
    print("3. Run this script to test Hub integration")


if __name__ == "__main__":
    main()
