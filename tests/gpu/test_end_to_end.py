"""
End-to-end pipeline tests and CLI tests for lorashare.

Tests the full workflow: compress -> save -> load -> reconstruct -> inference,
as well as the lorashare CLI commands.
"""

import subprocess

import torch

from lorashare import SHAREModel

from .conftest import (
    BASE_MODEL,
    TASK_NAMES,
    evaluate_adapter,
    requires_adapters,
    requires_gpu,
)


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


@requires_adapters
@requires_gpu
class TestFullPipeline:
    """Test the complete compress -> save -> load -> reconstruct -> infer pipeline."""

    def test_compress_save_load_reconstruct_infer(self, adapter_paths, tokenizer, tmp_path):
        """Full pipeline should produce valid predictions on real text."""
        from transformers import AutoModelForSequenceClassification

        # Compress
        share = SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")

        # Save
        checkpoint_dir = tmp_path / "share_checkpoint"
        share.save_pretrained(str(checkpoint_dir))

        # Load
        loaded = SHAREModel.from_pretrained(str(checkpoint_dir))
        assert set(loaded.adapter_names) == set(share.adapter_names)

        # Reconstruct + infer
        text = "The movie was absolutely wonderful and I loved every minute of it."
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")

        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2
        ).to("cuda")
        model = loaded.apply(base, "sst2")
        model.eval()

        with torch.no_grad():
            output = model(**inputs)

        assert output.logits.shape == (1, 2)
        pred = output.logits.argmax(dim=-1).item()
        assert pred in [0, 1], f"Invalid prediction: {pred}"

    def test_add_adapter_then_infer(self, adapter_paths, tokenizer, tmp_path):
        """Compress 3 adapters, add 4th, then infer with the 4th."""
        from transformers import AutoModelForSequenceClassification

        # Compress first 3
        first_three = {k: v for k, v in list(adapter_paths.items())[:3]}
        share = SHAREModel.from_adapters(first_three, num_components=32, device="cuda")
        assert len(share.adapter_names) == 3

        # Add 4th
        fourth_name = list(adapter_paths.keys())[3]
        fourth_path = adapter_paths[fourth_name]
        share.add_adapter(fourth_path, name=fourth_name, device="cuda")
        assert len(share.adapter_names) == 4
        assert fourth_name in share.adapter_names

        # Infer with 4th adapter
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2
        ).to("cuda")
        model = share.apply(base, fourth_name)
        model.eval()

        dummy = torch.randint(0, 1000, (1, 16)).to("cuda")
        with torch.no_grad():
            output = model(dummy)
        assert output.logits.shape == (1, 2)

    def test_remove_adapter_others_unaffected(self, adapter_paths):
        """Removing an adapter should not change reconstruction of remaining ones."""
        share = SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")

        # Record reconstruction of sst2 before removing cola
        recon_before = share.reconstruct("sst2")

        # Remove cola
        share.remove_adapter("cola")
        assert "cola" not in share.adapter_names
        assert len(share.adapter_names) == 3

        # Reconstruct sst2 again
        recon_after = share.reconstruct("sst2")

        # Should be identical
        for key in recon_before:
            if key in recon_after:
                assert torch.equal(recon_before[key], recon_after[key]), (
                    f"Reconstruction of sst2 changed after removing cola for key {key}"
                )

    def test_save_load_roundtrip_preserves_accuracy(
        self, adapter_paths, tokenizer, baselines, tmp_path
    ):
        """Accuracy after save/load roundtrip should match direct compression."""
        from transformers import AutoModelForSequenceClassification

        # Compress and save
        share = SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")
        checkpoint = tmp_path / "roundtrip_checkpoint"
        share.save_pretrained(str(checkpoint))

        # Load and evaluate
        loaded = SHAREModel.from_pretrained(str(checkpoint))

        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2
        ).to("cuda")
        model = loaded.apply(base, "sst2")
        model.eval()

        results = evaluate_adapter(model, tokenizer, "sst2", max_samples=128)
        baseline_acc = baselines["sst2"]["metric_value"]

        # Should still be close to baseline after roundtrip
        assert results["accuracy"] >= baseline_acc - 0.05, (
            f"SST-2 accuracy after roundtrip ({results['accuracy']:.4f}) "
            f"dropped too much from baseline ({baseline_acc:.4f})"
        )

    def test_reconstruct_all_adapters(self, adapter_paths, tmp_path):
        """All 4 adapters should reconstruct as valid PEFT format."""
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification

        share = SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")

        for task_name in TASK_NAMES:
            output_dir = tmp_path / f"recon_{task_name}"
            share.reconstruct(task_name, output_dir=str(output_dir))

            # Verify standard PEFT loading
            base = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL, num_labels=2
            )
            model = PeftModel.from_pretrained(base, str(output_dir))
            assert model is not None

    def test_multiple_reconstruct_calls_consistent(self, share_model):
        """Calling reconstruct() twice for the same adapter should give identical results."""
        recon1 = share_model.reconstruct("sst2")
        recon2 = share_model.reconstruct("sst2")

        assert recon1.keys() == recon2.keys()
        for key in recon1:
            assert torch.equal(recon1[key], recon2[key]), (
                f"Inconsistent reconstruction for key {key}"
            )


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


@requires_adapters
@requires_gpu
class TestCLI:
    """Test the lorashare CLI commands."""

    def _run_cli(self, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
        """Run lorashare CLI command and return result."""
        return subprocess.run(
            ["lorashare"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_cli_compress(self, adapter_paths, tmp_path):
        """lorashare compress should create a valid checkpoint."""
        output = tmp_path / "cli_checkpoint"
        paths = list(adapter_paths.values())

        result = self._run_cli([
            "compress",
            *paths,
            "-o", str(output),
            "-k", "32",
        ])

        assert result.returncode == 0, f"CLI compress failed:\n{result.stderr}"
        assert (output / "share_config.json").exists()
        assert (output / "shared_components.safetensors").exists()

    def test_cli_info(self, adapter_paths, tmp_path):
        """lorashare info should print checkpoint statistics."""
        # First create a checkpoint
        checkpoint = tmp_path / "info_checkpoint"
        share = SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")
        share.save_pretrained(str(checkpoint))

        result = self._run_cli(["info", str(checkpoint)])

        assert result.returncode == 0, f"CLI info failed:\n{result.stderr}"
        combined_output = result.stdout + result.stderr
        assert "Compression Summary" in combined_output, (
            f"Expected 'Compression Summary' in output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_cli_reconstruct_single(self, adapter_paths, tmp_path):
        """lorashare reconstruct --adapter should reconstruct one adapter."""
        checkpoint = tmp_path / "recon_checkpoint"
        share = SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")
        share.save_pretrained(str(checkpoint))

        output = tmp_path / "recon_output"
        result = self._run_cli([
            "reconstruct", str(checkpoint),
            "--adapter", "sst2",
            "-o", str(output),
        ])

        assert result.returncode == 0, f"CLI reconstruct failed:\n{result.stderr}"
        assert (output / "adapter_model.safetensors").exists()
        assert (output / "adapter_config.json").exists()

    def test_cli_reconstruct_all(self, adapter_paths, tmp_path):
        """lorashare reconstruct --all should reconstruct all adapters."""
        checkpoint = tmp_path / "recon_all_checkpoint"
        share = SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")
        share.save_pretrained(str(checkpoint))

        output = tmp_path / "recon_all_output"
        result = self._run_cli([
            "reconstruct", str(checkpoint),
            "--all",
            "-o", str(output),
        ])

        assert result.returncode == 0, f"CLI reconstruct --all failed:\n{result.stderr}"
        for task_name in TASK_NAMES:
            adapter_dir = output / task_name
            assert adapter_dir.exists(), f"Missing reconstructed adapter: {task_name}"
            assert (adapter_dir / "adapter_model.safetensors").exists()

    def test_cli_compress_with_gpu(self, adapter_paths, tmp_path):
        """lorashare compress --device cuda should use GPU."""
        output = tmp_path / "cli_gpu_checkpoint"
        paths = list(adapter_paths.values())

        result = self._run_cli([
            "compress",
            *paths,
            "-o", str(output),
            "-k", "32",
            "--device", "cuda",
        ])

        assert result.returncode == 0, f"CLI GPU compress failed:\n{result.stderr}"
        assert (output / "share_config.json").exists()

    def test_cli_compress_layer_by_layer(self, adapter_paths, tmp_path):
        """lorashare compress --layer-by-layer should complete successfully."""
        output = tmp_path / "cli_lbl_checkpoint"
        paths = list(adapter_paths.values())

        result = self._run_cli([
            "compress",
            *paths,
            "-o", str(output),
            "-k", "32",
            "--layer-by-layer",
        ])

        assert result.returncode == 0, f"CLI layer-by-layer failed:\n{result.stderr}"
        assert (output / "share_config.json").exists()

        # Verify the checkpoint is loadable
        loaded = SHAREModel.from_pretrained(str(output))
        assert len(loaded.adapter_names) == 4
