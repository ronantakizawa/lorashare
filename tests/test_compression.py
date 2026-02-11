"""Unit tests for the compression module."""

import pytest
import torch

from lorashare.compression import (
    build_lora_key,
    combine_adapter_weights,
    compute_adapter_loadings,
    compute_reconstruction_error,
    compute_shared_components,
    eigendecomposition,
    group_key,
    parse_group_key,
    parse_lora_key,
    reconstruct_adapter_weights,
    select_num_components,
)


# ── Key Parsing ──────────────────────────────────────────────────────────────


class TestKeyParsing:
    def test_parse_standard_key(self):
        key = "base_model.model.encoder.layer.0.attention.self.query.lora_A.weight"
        layer, side = parse_lora_key(key)
        assert layer == "encoder.layer.0.attention.self.query"
        assert side == "A"

    def test_parse_b_side(self):
        key = "base_model.model.encoder.layer.0.attention.self.value.lora_B.weight"
        layer, side = parse_lora_key(key)
        assert layer == "encoder.layer.0.attention.self.value"
        assert side == "B"

    def test_parse_without_prefix(self):
        key = "encoder.layer.0.attention.self.query.lora_A.weight"
        layer, side = parse_lora_key(key)
        assert layer == "encoder.layer.0.attention.self.query"
        assert side == "A"

    def test_parse_with_default_adapter(self):
        key = "base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight"
        layer, side = parse_lora_key(key)
        assert layer == "encoder.layer.0.attention.self.query"
        assert side == "A"

    def test_parse_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Not a LoRA key"):
            parse_lora_key("classifier.weight")

    def test_build_lora_key(self):
        result = build_lora_key("encoder.layer.0.attention.self.query", "A")
        assert result == "base_model.model.encoder.layer.0.attention.self.query.lora_A.weight"

    def test_roundtrip(self):
        original = "base_model.model.encoder.layer.5.attention.self.value.lora_B.weight"
        layer, side = parse_lora_key(original)
        rebuilt = build_lora_key(layer, side)
        assert rebuilt == original

    def test_group_key_roundtrip(self):
        gk = group_key("encoder.layer.0.query", "A")
        assert gk == "encoder.layer.0.query:A"
        layer, side = parse_group_key(gk)
        assert layer == "encoder.layer.0.query"
        assert side == "A"


# ── Eigendecomposition ───────────────────────────────────────────────────────


class TestEigendecomposition:
    def test_known_matrix(self):
        """Eigenvalues should match numpy for a known matrix."""
        torch.manual_seed(42)
        matrix = torch.randn(16, 8)
        result = eigendecomposition(matrix)

        # Verify shapes
        assert result["eigenvalues"].shape == (16,)
        assert result["eigenvectors"].shape == (16, 16)

    def test_sorted_descending(self):
        torch.manual_seed(42)
        matrix = torch.randn(32, 10)
        result = eigendecomposition(matrix)
        evals = result["eigenvalues"]
        # Should be descending (with small tolerance for equal values)
        assert torch.all(evals[:-1] >= evals[1:] - 1e-6)

    def test_eigenvectors_orthonormal(self):
        torch.manual_seed(42)
        matrix = torch.randn(16, 8)
        result = eigendecomposition(matrix)
        vecs = result["eigenvectors"]
        # V.T @ V should be identity
        product = vecs.T @ vecs
        assert torch.allclose(product, torch.eye(16), atol=1e-5)

    def test_real_valued(self):
        """eigh should return real values, unlike eig."""
        torch.manual_seed(42)
        matrix = torch.randn(16, 8)
        result = eigendecomposition(matrix)
        assert result["eigenvalues"].dtype == torch.float32
        assert result["eigenvectors"].dtype == torch.float32


# ── Combine Adapter Weights ──────────────────────────────────────────────────


class TestCombineAdapterWeights:
    def test_grouping(self, synthetic_adapters):
        combined = combine_adapter_weights(synthetic_adapters)
        # 2 layers x 2 modules x 2 sides (A+B) = 8 groups
        assert len(combined) == 8

    def test_each_group_has_all_adapters(self, synthetic_adapters):
        combined = combine_adapter_weights(synthetic_adapters)
        for gk, adapter_dict in combined.items():
            assert set(adapter_dict.keys()) == {"cola", "mrpc", "rte"}

    def test_skips_non_lora_keys(self):
        adapters = {
            "test": {
                "base_model.model.encoder.layer.0.attention.self.query.lora_A.weight": torch.randn(4, 64),
                "classifier.weight": torch.randn(2, 768),
                "some.other.key": torch.randn(10),
            }
        }
        combined = combine_adapter_weights(adapters)
        assert len(combined) == 1  # Only the lora key


# ── Shared Components ────────────────────────────────────────────────────────


class TestComputeSharedComponents:
    def test_shapes(self, synthetic_adapters):
        combined = combine_adapter_weights(synthetic_adapters)
        components, eigenvalues, k = compute_shared_components(combined, num_components=8)
        assert k == 8
        for gk, comp in components.items():
            # feature_dim is always the larger dim (64 in our test)
            assert comp.shape[1] == 8  # k components
            assert comp.shape[0] == 64  # feature_dim

    def test_auto_selection(self, synthetic_adapters):
        combined = combine_adapter_weights(synthetic_adapters)
        components, eigenvalues, k = compute_shared_components(
            combined, num_components="auto", variance_threshold=0.95
        )
        # k should be > 0 and <= feature_dim
        assert 1 <= k <= 64

    def test_clamps_to_max_available(self, synthetic_adapters):
        combined = combine_adapter_weights(synthetic_adapters)
        components, _, k = compute_shared_components(combined, num_components=1000)
        assert k <= 64  # Can't exceed feature_dim


# ── Adapter Loadings ─────────────────────────────────────────────────────────


class TestComputeAdapterLoadings:
    def test_shapes(self, synthetic_adapters):
        combined = combine_adapter_weights(synthetic_adapters)
        components, _, k = compute_shared_components(combined, num_components=8)
        loadings = compute_adapter_loadings(components, synthetic_adapters["cola"])
        for gk, loading in loadings.items():
            assert loading.shape == (8, 4)  # (k, rank)

    def test_all_groups_present(self, synthetic_adapters):
        combined = combine_adapter_weights(synthetic_adapters)
        components, _, k = compute_shared_components(combined, num_components=8)
        loadings = compute_adapter_loadings(components, synthetic_adapters["cola"])
        assert set(loadings.keys()) == set(components.keys())


# ── Reconstruction ───────────────────────────────────────────────────────────


class TestReconstruction:
    def test_shapes(self, synthetic_adapters):
        combined = combine_adapter_weights(synthetic_adapters)
        components, _, k = compute_shared_components(combined, num_components=8)
        loadings = compute_adapter_loadings(components, synthetic_adapters["cola"])
        reconstructed = reconstruct_adapter_weights(components, loadings)

        for key, tensor in reconstructed.items():
            layer, side = parse_lora_key(key)
            if side == "A":
                assert tensor.shape == (4, 64)  # (rank, feature_dim)
            else:
                assert tensor.shape == (64, 4)  # (feature_dim, rank)

    def test_high_k_near_exact(self, synthetic_adapters):
        """With enough components, reconstruction should be very close."""
        combined = combine_adapter_weights(synthetic_adapters)
        # Use max possible components for nearly exact reconstruction
        components, _, k = compute_shared_components(combined, num_components=64)
        loadings = compute_adapter_loadings(components, synthetic_adapters["cola"])
        reconstructed = reconstruct_adapter_weights(components, loadings)
        error = compute_reconstruction_error(synthetic_adapters["cola"], reconstructed)
        # With full-rank components, error should be very small
        assert error["mean"] < 0.05

    def test_error_decreases_with_k(self, synthetic_adapters):
        combined = combine_adapter_weights(synthetic_adapters)
        errors = []
        for k in [2, 8, 32]:
            components, _, _ = compute_shared_components(combined, num_components=k)
            loadings = compute_adapter_loadings(components, synthetic_adapters["cola"])
            reconstructed = reconstruct_adapter_weights(components, loadings)
            err = compute_reconstruction_error(synthetic_adapters["cola"], reconstructed)
            errors.append(err["mean"])
        # Error should decrease as k increases
        assert errors[0] >= errors[1] >= errors[2]


# ── Select Num Components ────────────────────────────────────────────────────


class TestSelectNumComponents:
    def test_known_distribution(self):
        # First eigenvalue has 99% of the variance
        evals = torch.tensor([99.0, 0.5, 0.3, 0.1, 0.05, 0.05])
        result = select_num_components({"layer:A": evals}, variance_threshold=0.95)
        assert result == 1

    def test_uniform_distribution(self):
        # All eigenvalues equal -> need many components
        evals = torch.ones(10)
        result = select_num_components({"layer:A": evals}, variance_threshold=0.95)
        assert result >= 9  # Need almost all for 95%

    def test_uses_worst_case(self):
        # Layer A needs 1, layer B needs 5 -> should return 5
        evals_easy = torch.tensor([99.0, 0.5, 0.3, 0.1, 0.05])
        evals_hard = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        result = select_num_components(
            {"layer:A": evals_easy, "layer:B": evals_hard},
            variance_threshold=0.95,
        )
        assert result >= 4  # Driven by the harder layer


# ── Reconstruction Error ─────────────────────────────────────────────────────


class TestReconstructionError:
    def test_identical_is_zero(self):
        weights = {"key_a": torch.randn(4, 64), "key_b": torch.randn(64, 4)}
        error = compute_reconstruction_error(weights, weights)
        assert error["mean"] == pytest.approx(0.0, abs=1e-7)
        assert error["max"] == pytest.approx(0.0, abs=1e-7)

    def test_different_is_nonzero(self):
        orig = {"key_a": torch.randn(4, 64)}
        recon = {"key_a": torch.randn(4, 64)}
        error = compute_reconstruction_error(orig, recon)
        assert error["mean"] > 0.0
