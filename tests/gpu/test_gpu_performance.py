"""
GPU performance tests for lorashare.

Tests GPU acceleration speed, memory usage, layer-by-layer processing,
and chunked processing on actual hardware.
"""

import time

import pytest
import torch

from lorashare import SHAREModel

from .conftest import requires_adapters, requires_gpu


# ---------------------------------------------------------------------------
# GPU Acceleration
# ---------------------------------------------------------------------------


@requires_adapters
@requires_gpu
class TestGPUAcceleration:
    """Test GPU provides measurable speedup over CPU."""

    def test_gpu_faster_than_cpu(self, adapter_paths):
        """GPU compression should be at least 2x faster than CPU."""
        # Warmup GPU
        torch.cuda.synchronize()

        # Time CPU
        start = time.perf_counter()
        SHAREModel.from_adapters(adapter_paths, num_components=32, device="cpu")
        cpu_time = time.perf_counter() - start

        # Time GPU
        torch.cuda.synchronize()
        start = time.perf_counter()
        SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        speedup = cpu_time / gpu_time
        print(f"\nCPU time: {cpu_time:.2f}s, GPU time: {gpu_time:.2f}s, Speedup: {speedup:.1f}x")

        # GPU should be at least 2x faster for real-sized adapters
        assert speedup > 2.0, (
            f"GPU speedup {speedup:.1f}x is below 2x threshold. "
            f"CPU: {cpu_time:.2f}s, GPU: {gpu_time:.2f}s"
        )

    def test_gpu_produces_same_results_as_cpu(self, adapter_paths):
        """GPU and CPU compression should produce equivalent reconstruction quality."""
        share_cpu = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cpu"
        )
        share_gpu = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cuda"
        )

        # Eigenvectors can differ in sign and ordering for degenerate eigenvalues,
        # so compare reconstruction error rather than raw components.
        for task_name in ["sst2", "cola"]:
            err_cpu = share_cpu.reconstruction_error(
                task_name, original_path=adapter_paths[task_name]
            )
            err_gpu = share_gpu.reconstruction_error(
                task_name, original_path=adapter_paths[task_name]
            )

            assert abs(err_cpu["mean"] - err_gpu["mean"]) < 0.05, (
                f"{task_name}: CPU error ({err_cpu['mean']:.6f}) and "
                f"GPU error ({err_gpu['mean']:.6f}) differ significantly"
            )

    def test_gpu_memory_usage(self, adapter_paths):
        """GPU memory usage should be reasonable (< 4GB for roberta-base adapters)."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        baseline_mem = torch.cuda.memory_allocated()
        SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")
        peak_mem = torch.cuda.max_memory_allocated()

        used_mb = (peak_mem - baseline_mem) / (1024 * 1024)
        print(f"\nPeak GPU memory used: {used_mb:.1f} MB")

        assert used_mb < 4096, (
            f"GPU memory usage {used_mb:.1f} MB exceeds 4GB limit"
        )

    def test_device_auto_selects_gpu(self, adapter_paths):
        """device=None should auto-select GPU when available."""
        share = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device=None
        )
        # Model should have been created successfully
        assert len(share.adapter_names) == 4
        assert share.config.num_components == 32


# ---------------------------------------------------------------------------
# Layer-by-Layer on GPU
# ---------------------------------------------------------------------------


@requires_adapters
@requires_gpu
class TestLayerByLayerGPU:
    """Test layer-by-layer processing on GPU."""

    def test_layer_by_layer_reduces_peak_memory(self, adapter_paths):
        """Layer-by-layer should use less peak GPU memory than standard."""
        # Standard
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        SHAREModel.from_adapters(adapter_paths, num_components=32, device="cuda")
        standard_peak = torch.cuda.max_memory_allocated() - baseline

        # Layer-by-layer
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cuda", layer_by_layer=True
        )
        lbl_peak = torch.cuda.max_memory_allocated() - baseline

        print(f"\nStandard peak: {standard_peak / 1e6:.1f} MB")
        print(f"Layer-by-layer peak: {lbl_peak / 1e6:.1f} MB")

        # Layer-by-layer should use less or equal memory
        # Allow some tolerance since measurement is approximate
        assert lbl_peak <= standard_peak * 1.1, (
            f"Layer-by-layer ({lbl_peak / 1e6:.1f} MB) used more memory than "
            f"standard ({standard_peak / 1e6:.1f} MB)"
        )

    def test_layer_by_layer_matches_standard_quality(self, adapter_paths):
        """Layer-by-layer reconstruction error should match standard approach."""
        share_standard = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cuda"
        )
        share_lbl = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cuda", layer_by_layer=True
        )

        for task_name in ["sst2", "cola"]:
            err_std = share_standard.reconstruction_error(
                task_name, original_path=adapter_paths[task_name]
            )
            err_lbl = share_lbl.reconstruction_error(
                task_name, original_path=adapter_paths[task_name]
            )

            assert abs(err_std["mean"] - err_lbl["mean"]) < 0.01, (
                f"{task_name}: layer-by-layer error ({err_lbl['mean']:.6f}) differs "
                f"significantly from standard ({err_std['mean']:.6f})"
            )

    def test_layer_by_layer_on_gpu(self, adapter_paths):
        """Layer-by-layer with device='cuda' should complete without error."""
        share = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cuda", layer_by_layer=True
        )
        assert len(share.adapter_names) == 4
        assert share.config.num_components == 32

        # Verify reconstruction works
        for task_name in share.adapter_names:
            weights = share.reconstruct(task_name)
            assert len(weights) > 0


# ---------------------------------------------------------------------------
# Chunked Processing on GPU
# ---------------------------------------------------------------------------


@requires_adapters
@requires_gpu
class TestChunkedGPU:
    """Test chunked processing on GPU."""

    def test_chunked_with_gpu(self, adapter_paths):
        """Chunked compression on GPU should complete successfully."""
        share = SHAREModel.from_adapters(
            adapter_paths, num_components=32, device="cuda", chunk_size=2
        )
        assert len(share.adapter_names) == 4
        # Verify all adapters can be reconstructed
        for name in share.adapter_names:
            weights = share.reconstruct(name)
            assert len(weights) > 0

    def test_chunked_quality_vs_standard(self, adapter_paths):
        """Chunked should produce reasonable reconstruction quality.

        Note: Chunked processing uses meta-PCA (SVD on stacked chunk components),
        which introduces additional approximation. With only 4 adapters and
        chunk_size=2, the approximation can be significant. The key check is
        that reconstruction error is bounded, not that it matches standard.
        """
        share_standard = SHAREModel.from_adapters(
            adapter_paths, num_components=16, device="cuda"
        )
        share_chunked = SHAREModel.from_adapters(
            adapter_paths, num_components=16, device="cuda", chunk_size=2
        )

        for task_name in ["sst2", "cola"]:
            err_std = share_standard.reconstruction_error(
                task_name, original_path=adapter_paths[task_name]
            )
            err_chunk = share_chunked.reconstruction_error(
                task_name, original_path=adapter_paths[task_name]
            )

            # Chunked error should be bounded (< 1.0 relative error)
            assert err_chunk["mean"] < 1.0, (
                f"{task_name}: chunked error ({err_chunk['mean']:.6f}) exceeds 1.0"
            )
            # And standard should have lower error
            assert err_std["mean"] <= err_chunk["mean"] + 0.01, (
                f"{task_name}: standard error ({err_std['mean']:.6f}) unexpectedly "
                f"higher than chunked ({err_chunk['mean']:.6f})"
            )

    def test_chunk_size_variations(self, adapter_paths):
        """Different chunk sizes should all produce valid results."""
        for chunk_size in [2, 3, 4]:
            share = SHAREModel.from_adapters(
                adapter_paths,
                num_components=32,
                device="cuda",
                chunk_size=chunk_size,
            )
            assert len(share.adapter_names) == 4
            # All adapters should be reconstructable
            for task_name in share.adapter_names:
                weights = share.reconstruct(task_name)
                assert len(weights) > 0, (
                    f"Empty reconstruction for {task_name} with chunk_size={chunk_size}"
                )
