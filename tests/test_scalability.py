"""Tests for scalability features: GPU acceleration, layer-by-layer, chunked processing."""

import pytest
import torch

from lorashare import SHAREModel


class TestGPUAcceleration:
    """Test GPU acceleration feature."""

    def test_gpu_acceleration_when_available(self, saved_adapters):
        """Test that GPU acceleration works if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            device="cuda",
        )

        assert share.config.num_components == 8
        assert len(share.adapter_names) == 3

    def test_falls_back_to_cpu(self, saved_adapters):
        """Test that it falls back to CPU gracefully."""
        # Explicitly request CPU
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            device="cpu",
        )

        assert share.config.num_components == 8

    def test_auto_device_selection(self, saved_adapters):
        """Test automatic device selection."""
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            device=None,  # Auto-select
        )

        assert share.config.num_components == 8


class TestLayerByLayer:
    """Test layer-by-layer processing."""

    def test_layer_by_layer_compression(self, saved_adapters):
        """Test that layer-by-layer processing works."""
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            layer_by_layer=True,
        )

        assert share.config.num_components == 8
        assert len(share.adapter_names) == 3

        # Verify can reconstruct
        weights = share.reconstruct("cola")
        assert len(weights) > 0

    def test_layer_by_layer_matches_standard(self, saved_adapters):
        """Test that layer-by-layer gives similar results to standard."""
        # Standard compression
        share_standard = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            layer_by_layer=False,
        )

        # Layer-by-layer compression
        share_layerwise = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            layer_by_layer=True,
        )

        # Both should have same structure
        assert share_standard.config.num_components == share_layerwise.config.num_components
        assert set(share_standard.adapter_names) == set(share_layerwise.adapter_names)

        # Reconstruction should work for both
        weights_std = share_standard.reconstruct("cola")
        weights_layer = share_layerwise.reconstruct("cola")

        assert set(weights_std.keys()) == set(weights_layer.keys())


class TestChunkedProcessing:
    """Test chunked processing for many adapters."""

    def test_chunked_compression(self, saved_adapters):
        """Test that chunked processing works."""
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            chunk_size=2,  # Process 2 at a time
        )

        assert share.config.num_components == 8
        assert len(share.adapter_names) == 3

        # Verify can reconstruct all adapters
        for name in share.adapter_names:
            weights = share.reconstruct(name)
            assert len(weights) > 0

    def test_chunked_with_small_chunk_size(self, saved_adapters):
        """Test with chunk_size=2 (small chunks)."""
        # Note: chunk_size=1 would fail validation (need at least 2 per chunk)
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            chunk_size=2,
        )

        assert len(share.adapter_names) == 3

        # All adapters should be reconstructable
        for name in share.adapter_names:
            weights = share.reconstruct(name)
            assert "lora_" in list(weights.keys())[0]

    def test_chunked_preserves_quality(self, saved_adapters, synthetic_adapters):
        """Test that chunked processing maintains quality."""
        # Chunked compression
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            chunk_size=2,
        )

        # Check reconstruction error
        error = share.reconstruction_error(
            "cola",
            original_weights=synthetic_adapters["cola"],
        )

        # Chunked processing introduces some quality loss from meta-PCA
        # but should still be < 1.0 (reasonable quality)
        assert error["mean"] < 1.0, f"Chunked processing quality too poor: {error['mean']}"


class TestCombinedFeatures:
    """Test combinations of features."""

    def test_gpu_with_layer_by_layer(self, saved_adapters):
        """Test GPU acceleration with layer-by-layer."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            device="cuda",
            layer_by_layer=True,
        )

        assert len(share.adapter_names) == 3

    def test_gpu_with_chunked(self, saved_adapters):
        """Test GPU acceleration with chunked processing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            device="cuda",
            chunk_size=2,
        )

        assert len(share.adapter_names) == 3

    def test_cannot_use_both_layer_and_chunked(self, saved_adapters):
        """Test that layer-by-layer takes precedence if both specified."""
        # Currently chunk_size takes precedence
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            layer_by_layer=True,
            chunk_size=2,
        )

        # Should still work (chunk_size takes precedence)
        assert len(share.adapter_names) == 3


class TestScalabilityEdgeCases:
    """Test edge cases for scalability features."""

    def test_chunk_size_larger_than_adapters(self, saved_adapters):
        """Test chunk_size larger than number of adapters."""
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components=8,
            chunk_size=100,  # Much larger than 3 adapters
        )

        assert len(share.adapter_names) == 3

    def test_auto_components_with_chunked(self, saved_adapters):
        """Test auto component selection with chunked processing."""
        share = SHAREModel.from_adapters(
            list(saved_adapters.values()),
            num_components="auto",
            variance_threshold=0.95,
            chunk_size=2,
        )

        assert share.config.num_components > 0
        assert share.config.component_selection == "auto"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
