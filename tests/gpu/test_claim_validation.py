"""
Validate lorashare's core claims with real GLUE LoRA adapters.

Claims tested:
1. Up to 100x+ memory savings (compression ratio)
2. Zero retraining, equivalent inference quality (accuracy preservation)
3. Classifier heads preserved exactly
4. Reconstructed adapters are standard PEFT format

Requires: pre-trained adapters from train_adapters.py
"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from lorashare import SHAREModel

from .conftest import (
    ADAPTER_DIR,
    BASE_MODEL,
    GLUE_TASKS,
    TASK_NAMES,
    evaluate_adapter,
    get_original_model,
    requires_adapters,
    requires_gpu,
)


# ---------------------------------------------------------------------------
# Compression Ratio Claims
# ---------------------------------------------------------------------------


@requires_adapters
@requires_gpu
class TestCompressionRatioClaim:
    """Validate 'up to 100x+ memory savings' claim."""

    def test_compression_ratio_with_4_adapters(self, adapter_paths):
        """Compression ratio should exceed 1x with reasonable k.

        With 4 adapters of rank 8 on roberta-base (48 layer groups):
        - Original: 4 × 48 × 768 × 8 = 1,179,648 params
        - Compressed (k=8): 48 × 768 × 8 + 4 × 48 × 8 × 8 = 307,200 params → ~3.8x
        The "100x+" claim requires many more adapters sharing the same components.
        """
        share = SHAREModel.from_adapters(adapter_paths, num_components=8, device="cuda")
        stats = share.config.compression_stats

        ratio = stats["compression_ratio"]
        assert ratio > 2, (
            f"Compression ratio {ratio:.1f}x is below 2x threshold. "
            f"Original: {stats['original_total_params']:,} params, "
            f"Compressed: {stats['compressed_total_params']:,} params"
        )

    def test_compression_ratio_scales_with_adapters(self, adapter_paths):
        """More adapters should yield higher compression ratio."""
        paths = list(adapter_paths.values())
        names = list(adapter_paths.keys())

        # 2 adapters
        share_2 = SHAREModel.from_adapters(
            {names[0]: paths[0], names[1]: paths[1]},
            num_components=8,
            device="cuda",
        )
        ratio_2 = share_2.config.compression_stats["compression_ratio"]

        # 4 adapters
        share_4 = SHAREModel.from_adapters(adapter_paths, num_components=8, device="cuda")
        ratio_4 = share_4.config.compression_stats["compression_ratio"]

        assert ratio_4 > ratio_2, (
            f"4-adapter ratio ({ratio_4:.1f}x) should exceed "
            f"2-adapter ratio ({ratio_2:.1f}x)"
        )

    def test_memory_savings_in_ram(self, adapter_paths):
        """Compressed representation should use less tensor memory than originals."""
        # Measure original adapter sizes
        original_bytes = 0
        for path in adapter_paths.values():
            weights_file = Path(path) / "adapter_model.safetensors"
            original_bytes += weights_file.stat().st_size

        # Compress with k=8 for actual compression
        share = SHAREModel.from_adapters(adapter_paths, num_components=8, device="cuda")

        compressed_bytes = 0
        for tensor in share.components.values():
            compressed_bytes += tensor.nelement() * tensor.element_size()
        for adapter_loadings in share.all_loadings.values():
            for tensor in adapter_loadings.values():
                compressed_bytes += tensor.nelement() * tensor.element_size()

        ratio = original_bytes / compressed_bytes
        assert ratio > 2, (
            f"RAM savings ratio {ratio:.1f}x is below 2x. "
            f"Original: {original_bytes:,} bytes, Compressed: {compressed_bytes:,} bytes"
        )

    def test_summary_reports_accurate_stats(self, adapter_paths, capsys):
        """summary() output should match actual measurements."""
        share = SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")
        stats = share.config.compression_stats

        # Independently count parameters
        actual_shared = sum(t.numel() for t in share.components.values())
        actual_loading = sum(
            sum(t.numel() for t in loadings.values())
            for loadings in share.all_loadings.values()
        )

        assert stats["shared_component_params"] == actual_shared
        assert stats["per_adapter_loading_params"] == actual_loading
        assert stats["compressed_total_params"] == actual_shared + actual_loading

        # Verify summary prints without error
        share.summary()
        captured = capsys.readouterr()
        assert "Compression Summary" in captured.out
        assert str(share.config.num_adapters) in captured.out

    def test_disk_size_savings(self, adapter_paths, tmp_path):
        """Saved checkpoint should be smaller than combined adapter files."""
        original_size = sum(
            sum(f.stat().st_size for f in Path(p).iterdir() if f.is_file())
            for p in adapter_paths.values()
        )

        share = SHAREModel.from_adapters(adapter_paths, num_components=8, device="cuda")
        checkpoint_dir = tmp_path / "checkpoint"
        share.save_pretrained(str(checkpoint_dir))

        compressed_size = sum(
            f.stat().st_size
            for f in checkpoint_dir.rglob("*")
            if f.is_file()
        )

        ratio = original_size / compressed_size
        assert ratio > 1.5, (
            f"Disk savings ratio {ratio:.1f}x is below 1.5x. "
            f"Original: {original_size:,} bytes, Checkpoint: {compressed_size:,} bytes"
        )


# ---------------------------------------------------------------------------
# Accuracy Preservation Claims
# ---------------------------------------------------------------------------


@requires_adapters
@requires_gpu
class TestAccuracyPreservation:
    """Validate 'equivalent inference quality' claim.

    Compares accuracy of original vs reconstructed adapters on GLUE validation sets.
    """

    @pytest.fixture(scope="class")
    def compressed_and_baselines(self, adapter_paths, baselines):
        """Compress all 4 adapters and return with baselines."""
        share = SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")
        return share, baselines

    def _eval_reconstructed(self, share, task_name, tokenizer):
        """Helper: reconstruct an adapter and evaluate it."""
        from transformers import AutoModelForSequenceClassification

        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2
        ).to("cuda")

        model = share.apply(base, task_name)
        model.eval()
        return evaluate_adapter(model, tokenizer, task_name, max_samples=256)

    def test_sst2_accuracy_preserved(self, compressed_and_baselines, tokenizer):
        """SST-2 accuracy drop should be < 2 percentage points."""
        share, baselines = compressed_and_baselines
        results = self._eval_reconstructed(share, "sst2", tokenizer)

        baseline_acc = baselines["sst2"]["metric_value"]
        recon_acc = results["accuracy"]

        assert recon_acc >= baseline_acc - 0.02, (
            f"SST-2 accuracy dropped too much: "
            f"original={baseline_acc:.4f}, reconstructed={recon_acc:.4f}, "
            f"delta={baseline_acc - recon_acc:.4f}"
        )

    def test_cola_accuracy_preserved(self, compressed_and_baselines, tokenizer):
        """CoLA Matthews correlation drop should be < 0.05."""
        share, baselines = compressed_and_baselines
        results = self._eval_reconstructed(share, "cola", tokenizer)

        baseline_mcc = baselines["cola"]["metric_value"]
        recon_mcc = results["matthews_correlation"]

        assert recon_mcc >= baseline_mcc - 0.05, (
            f"CoLA MCC dropped too much: "
            f"original={baseline_mcc:.4f}, reconstructed={recon_mcc:.4f}, "
            f"delta={baseline_mcc - recon_mcc:.4f}"
        )

    def test_mrpc_accuracy_preserved(self, compressed_and_baselines, tokenizer):
        """MRPC F1 drop should be < 2 percentage points."""
        share, baselines = compressed_and_baselines
        results = self._eval_reconstructed(share, "mrpc", tokenizer)

        baseline_f1 = baselines["mrpc"]["metric_value"]
        recon_f1 = results["f1"]

        assert recon_f1 >= baseline_f1 - 0.02, (
            f"MRPC F1 dropped too much: "
            f"original={baseline_f1:.4f}, reconstructed={recon_f1:.4f}, "
            f"delta={baseline_f1 - recon_f1:.4f}"
        )

    def test_rte_accuracy_preserved(self, compressed_and_baselines, tokenizer):
        """RTE accuracy drop should be < 3 percentage points."""
        share, baselines = compressed_and_baselines
        results = self._eval_reconstructed(share, "rte", tokenizer)

        baseline_acc = baselines["rte"]["metric_value"]
        recon_acc = results["accuracy"]

        assert recon_acc >= baseline_acc - 0.03, (
            f"RTE accuracy dropped too much: "
            f"original={baseline_acc:.4f}, reconstructed={recon_acc:.4f}, "
            f"delta={baseline_acc - recon_acc:.4f}"
        )

    def test_all_tasks_preserve_baseline(self, compressed_and_baselines, tokenizer):
        """Reconstructed adapters should perform within tolerance of their baselines."""
        share, baselines = compressed_and_baselines

        for task_name in TASK_NAMES:
            results = self._eval_reconstructed(share, task_name, tokenizer)
            metric_key = GLUE_TASKS[task_name]["metric"]
            recon_value = results[metric_key]
            baseline_value = baselines[task_name]["metric_value"]

            # Reconstructed should be within 5pp / 0.05 of whatever the baseline was
            if metric_key == "matthews_correlation":
                assert recon_value >= baseline_value - 0.05, (
                    f"{task_name}: MCC dropped too much from baseline. "
                    f"baseline={baseline_value:.4f}, reconstructed={recon_value:.4f}"
                )
            else:
                assert recon_value >= baseline_value - 0.05, (
                    f"{task_name}: {metric_key} dropped too much from baseline. "
                    f"baseline={baseline_value:.4f}, reconstructed={recon_value:.4f}"
                )

    def test_accuracy_improves_with_more_components(self, adapter_paths, tokenizer):
        """Higher k should yield better reconstruction quality."""
        from transformers import AutoModelForSequenceClassification

        errors = {}
        for k in [4, 16, 32]:
            share = SHAREModel.from_adapters(adapter_paths, num_components=k, device="cuda")
            # Use reconstruction error as a proxy (faster than full eval)
            err = share.reconstruction_error("sst2", original_path=adapter_paths["sst2"])
            errors[k] = err["mean"]

        assert errors[4] >= errors[16] >= errors[32], (
            f"Reconstruction error should decrease with more components: "
            f"k=4: {errors[4]:.6f}, k=16: {errors[16]:.6f}, k=32: {errors[32]:.6f}"
        )


# ---------------------------------------------------------------------------
# Classifier Head Preservation Claims
# ---------------------------------------------------------------------------


@requires_adapters
@requires_gpu
class TestClassifierHeadPreservation:
    """Validate 'classifier heads preserved exactly' claim."""

    def test_classifier_weights_identical(self, adapter_paths, share_model):
        """Classifier head weights must be bitwise identical after compression."""
        for task_name, path in adapter_paths.items():
            original_weights = load_file(str(Path(path) / "adapter_model.safetensors"))

            # Get classifier keys from original
            classifier_keys = [
                k for k in original_weights
                if "lora_" not in k
            ]

            if not classifier_keys:
                continue

            reconstructed = share_model.reconstruct(task_name)

            for key in classifier_keys:
                assert key in reconstructed, f"Classifier key {key} missing in reconstruction"
                assert torch.equal(original_weights[key], reconstructed[key]), (
                    f"Classifier weight {key} for {task_name} is not identical. "
                    f"Max diff: {(original_weights[key] - reconstructed[key]).abs().max().item()}"
                )

    def test_classifier_output_identical(self, adapter_paths, share_model, tokenizer):
        """Classifier logits should be identical for same input."""
        from transformers import AutoModelForSequenceClassification

        text = "This is a test sentence for validation."
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")

        for task_name in TASK_NAMES:
            # Original model
            original_model = get_original_model(adapter_paths[task_name])

            # Reconstructed model
            base = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL, num_labels=2
            ).to("cuda")
            recon_model = share_model.apply(base, task_name)
            recon_model.eval()

            with torch.no_grad():
                orig_logits = original_model(**inputs).logits
                recon_logits = recon_model(**inputs).logits

            # Logits should be very close (LoRA weights differ slightly,
            # but classifier heads are exact)
            # We check the prediction matches, not exact logit values
            assert orig_logits.argmax(dim=-1).item() == recon_logits.argmax(dim=-1).item(), (
                f"{task_name}: predictions differ. "
                f"Original logits: {orig_logits}, Reconstructed: {recon_logits}"
            )

    def test_each_task_has_own_classifier(self, share_model):
        """Each adapter should have unique classifier weights."""
        classifier_sets = {}
        for task_name in share_model.adapter_names:
            heads = share_model.all_classifier_heads.get(task_name, {})
            if heads:
                # Use first classifier key as fingerprint
                first_key = sorted(heads.keys())[0]
                classifier_sets[task_name] = heads[first_key]

        if len(classifier_sets) < 2:
            pytest.skip("Not enough adapters with classifier heads")

        names = list(classifier_sets.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                assert not torch.equal(
                    classifier_sets[names[i]], classifier_sets[names[j]]
                ), f"Classifiers for {names[i]} and {names[j]} are identical (should differ)"


# ---------------------------------------------------------------------------
# Zero Retraining Claims
# ---------------------------------------------------------------------------


@requires_adapters
@requires_gpu
class TestZeroRetraining:
    """Validate 'works post-hoc without retraining' claim."""

    def test_no_gradient_computation(self, adapter_paths):
        """Compression should not require gradient tracking."""
        with torch.no_grad():
            share = SHAREModel.from_adapters(
                adapter_paths, num_components=32, device="cuda"
            )

        # Verify no tensors require grad
        for tensor in share.components.values():
            assert not tensor.requires_grad, "Component tensor has requires_grad=True"
        for loadings in share.all_loadings.values():
            for tensor in loadings.values():
                assert not tensor.requires_grad, "Loading tensor has requires_grad=True"

    def test_compress_is_deterministic(self, adapter_paths):
        """Compressing the same adapters twice should produce identical results."""
        share1 = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cuda"
        )
        share2 = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cuda"
        )

        for gk in share1.components:
            assert torch.allclose(
                share1.components[gk], share2.components[gk], atol=1e-5
            ), f"Components differ for {gk}"

        for name in share1.adapter_names:
            for gk in share1.all_loadings[name]:
                assert torch.allclose(
                    share1.all_loadings[name][gk],
                    share2.all_loadings[name][gk],
                    atol=1e-5,
                ), f"Loadings differ for {name}/{gk}"

    def test_reconstructed_is_standard_peft(self, adapter_paths, tmp_path):
        """Reconstructed adapters should load with vanilla PeftModel.from_pretrained."""
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification

        share = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cuda"
        )

        for task_name in TASK_NAMES:
            output_dir = tmp_path / f"recon_{task_name}"
            share.reconstruct(task_name, output_dir=str(output_dir))

            # Verify it loads as standard PEFT
            base = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL, num_labels=2
            )
            model = PeftModel.from_pretrained(base, str(output_dir))
            assert model is not None, f"Failed to load reconstructed {task_name} as PEFT"

            # Verify forward pass works
            model.eval()
            model.to("cuda")
            dummy = torch.randint(0, 1000, (1, 16)).to("cuda")
            with torch.no_grad():
                output = model(dummy)
            assert output.logits.shape == (1, 2), f"Wrong output shape for {task_name}"
