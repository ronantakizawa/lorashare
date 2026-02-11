"""Core compression algorithm: PCA-based LoRA adapter compression."""

from __future__ import annotations

import re
import warnings
from typing import Any

import torch


# ── Key Parsing ──────────────────────────────────────────────────────────────

_LORA_KEY_PATTERN = re.compile(
    r"^(?:base_model\.model\.)?(.+?)\.lora_([AB])(?:\.default)?\.weight$"
)


def parse_lora_key(full_key: str) -> tuple[str, str]:
    """Parse a PEFT LoRA state dict key into (layer_key, side).

    Handles keys with or without the ``base_model.model.`` prefix and
    optional ``.default`` adapter name.

    Args:
        full_key: e.g. ``"base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight"``

    Returns:
        Tuple of (layer_key, side) where side is ``"A"`` or ``"B"``.

    Raises:
        ValueError: If key doesn't match expected LoRA format.
    """
    m = _LORA_KEY_PATTERN.match(full_key)
    if not m:
        raise ValueError(f"Not a LoRA key: {full_key}")
    return m.group(1), m.group(2)


def build_lora_key(layer_key: str, side: str) -> str:
    """Reconstruct a standard PEFT LoRA state dict key.

    Args:
        layer_key: e.g. ``"encoder.layer.0.attention.self.query"``
        side: ``"A"`` or ``"B"``

    Returns:
        Key like ``"base_model.model.encoder.layer.0.attention.self.query.lora_A.weight"``
    """
    return f"base_model.model.{layer_key}.lora_{side}.weight"


def group_key(layer_key: str, side: str) -> str:
    """Create a group key for a (layer, side) pair."""
    return f"{layer_key}:{side}"


def parse_group_key(gk: str) -> tuple[str, str]:
    """Split a group key back into (layer_key, side)."""
    layer_key, side = gk.rsplit(":", 1)
    return layer_key, side


# ── Core Math ────────────────────────────────────────────────────────────────


def eigendecomposition(matrix: torch.Tensor) -> dict[str, torch.Tensor]:
    """Eigendecompose a centered covariance matrix.

    Centers data, computes ``cov = X @ X.T``, then uses
    ``torch.linalg.eigh`` (symmetric eigendecomposition for PSD matrices).

    Args:
        matrix: Input data matrix, shape ``(feature_dim, num_samples)``.

    Returns:
        Dict with ``"eigenvalues"`` of shape ``(feature_dim,)`` sorted
        descending and ``"eigenvectors"`` of shape ``(feature_dim, feature_dim)``
        with columns sorted correspondingly.
    """
    matrix = matrix.to(torch.float32)
    mean = matrix.mean(dim=1, keepdim=True)
    centered = matrix - mean
    cov = centered @ centered.T

    # eigh returns eigenvalues in ascending order
    eigenvals, eigenvecs = torch.linalg.eigh(cov)

    # Reverse to descending order
    idx = torch.arange(eigenvals.size(0) - 1, -1, -1, device=eigenvals.device)
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    return {"eigenvalues": eigenvals, "eigenvectors": eigenvecs}


def _orient_weight(tensor: torch.Tensor) -> torch.Tensor:
    """Orient a weight tensor so the larger dim is dim-0 (tall matrix).

    For lora_A (r, d) where r < d, this transposes to (d, r).
    For lora_B (d, r) where d > r, this keeps (d, r).
    """
    if tensor.shape[0] < tensor.shape[1]:
        return tensor.T
    return tensor


def combine_adapter_weights(
    adapters: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Group LoRA weights by (layer_key, side) across all adapters.

    Args:
        adapters: ``{adapter_name: {full_peft_key: weight_tensor}}``
            Only keys matching LoRA format are included; others are skipped.

    Returns:
        ``{group_key: {adapter_name: weight_tensor}}`` where group_key
        is ``"layer_key:side"``.
    """
    combined: dict[str, dict[str, torch.Tensor]] = {}
    for adapter_name, state_dict in adapters.items():
        for full_key, tensor in state_dict.items():
            try:
                layer_key, side = parse_lora_key(full_key)
            except ValueError:
                continue  # skip non-LoRA keys (classifier, etc.)
            gk = group_key(layer_key, side)
            if gk not in combined:
                combined[gk] = {}
            combined[gk][adapter_name] = tensor
    return combined


def compute_shared_components(
    combined: dict[str, dict[str, torch.Tensor]],
    num_components: int | str = 32,
    variance_threshold: float = 0.95,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """Compute PCA principal components for each (layer, side) group.

    For each group:
      1. Orient each adapter's weight tensor to be tall (feature_dim x rank)
      2. Concatenate across adapters: ``(feature_dim, N*rank)``
      3. Run eigendecomposition
      4. Take top-k eigenvectors as components

    Args:
        combined: Output of :func:`combine_adapter_weights`.
        num_components: Number of components to retain, or ``"auto"``.
        variance_threshold: Target explained variance when ``num_components="auto"``.

    Returns:
        Tuple of ``(components, eigenvalues, effective_k)`` where:
        - components: ``{group_key: tensor(feature_dim, k)}``
        - eigenvalues: ``{group_key: tensor(feature_dim,)}``
        - effective_k: actual number of components used
    """
    all_eigenvalues: dict[str, torch.Tensor] = {}
    all_eigenvectors: dict[str, torch.Tensor] = {}

    for gk, adapter_weights in combined.items():
        tensors = []
        for tensor in adapter_weights.values():
            oriented = _orient_weight(tensor).to(torch.float32)
            tensors.append(oriented)
        stacked = torch.cat(tensors, dim=1)  # (feature_dim, N*rank)

        result = eigendecomposition(stacked)
        all_eigenvalues[gk] = result["eigenvalues"]
        all_eigenvectors[gk] = result["eigenvectors"]

    # Determine k
    if num_components == "auto":
        k = select_num_components(all_eigenvalues, variance_threshold)
    else:
        k = int(num_components)

    # Clamp k to the max available eigenvectors per group
    max_available = min(ev.shape[0] for ev in all_eigenvectors.values())
    if k > max_available:
        warnings.warn(
            f"num_components={k} exceeds max available ({max_available}), clamping."
        )
        k = max_available

    components = {
        gk: eigvecs[:, :k].contiguous()
        for gk, eigvecs in all_eigenvectors.items()
    }

    return components, all_eigenvalues, k


def select_num_components(
    eigenvalues: dict[str, torch.Tensor],
    variance_threshold: float = 0.95,
) -> int:
    """Auto-select num_components based on explained variance.

    Finds the minimum k such that explained variance >= threshold
    for ALL layer groups (uses the worst-case maximum k needed).

    Args:
        eigenvalues: ``{group_key: tensor}`` of sorted descending eigenvalues.
        variance_threshold: Target explained variance ratio.

    Returns:
        Number of components needed.
    """
    max_k = 1
    for gk, evals in eigenvalues.items():
        # Clamp negative eigenvalues (numerical noise) to zero
        evals = evals.clamp(min=0)
        total = evals.sum()
        if total < 1e-12:
            max_k = max(max_k, 1)
            continue
        cumulative = evals.cumsum(0) / total
        # Find first index where cumulative >= threshold
        above = (cumulative >= variance_threshold).nonzero(as_tuple=True)[0]
        if len(above) > 0:
            k = above[0].item() + 1  # +1 because 0-indexed
        else:
            k = evals.shape[0]
        max_k = max(max_k, k)
    return max_k


def compute_adapter_loadings(
    components: dict[str, torch.Tensor],
    adapter_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Project one adapter's weights onto shared components.

    For lora_A (shape r, d): ``loadings = components.T @ lora_A.T`` -> ``(k, r)``
    For lora_B (shape d, r): ``loadings = components.T @ lora_B``   -> ``(k, r)``

    This matches the projection in the SHARE paper's ``calculate_eigenflux()``.

    Args:
        components: ``{group_key: tensor(feature_dim, k)}``
        adapter_weights: ``{full_peft_key: weight_tensor}`` for a single adapter.

    Returns:
        ``{group_key: tensor(k, r)}`` loadings for each layer group.
    """
    loadings: dict[str, torch.Tensor] = {}
    for full_key, weight in adapter_weights.items():
        try:
            layer_key, side = parse_lora_key(full_key)
        except ValueError:
            continue
        gk = group_key(layer_key, side)
        if gk not in components:
            continue

        comp = components[gk].to(torch.float32)  # (feature_dim, k)
        w = weight.to(torch.float32)

        if side == "A":
            # lora_A shape: (r, d). Project: components.T @ lora_A.T = (k, d) @ (d, r) = (k, r)
            loading = comp.T @ w.T
        else:
            # lora_B shape: (d, r). Project: components.T @ lora_B = (k, d) @ (d, r) = (k, r)
            loading = comp.T @ w

        loadings[gk] = loading
    return loadings


def reconstruct_adapter_weights(
    components: dict[str, torch.Tensor],
    loadings: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reconstruct standard PEFT LoRA weights from components + loadings.

    For side A: ``lora_A = (components @ loadings).T`` -> ``(r, feature_dim)``
    For side B: ``lora_B = components @ loadings``      -> ``(feature_dim, r)``

    Args:
        components: ``{group_key: tensor(feature_dim, k)}``
        loadings: ``{group_key: tensor(k, r)}``

    Returns:
        Dict with standard PEFT keys mapping to reconstructed weight tensors.
    """
    reconstructed: dict[str, torch.Tensor] = {}
    for gk, loading in loadings.items():
        layer_key, side = parse_group_key(gk)
        comp = components[gk]  # (feature_dim, k)

        # recons = comp @ loading -> (feature_dim, r)
        recons = comp @ loading

        if side == "A":
            # Original lora_A is (r, feature_dim)
            weight = recons.T.contiguous()
        else:
            # Original lora_B is (feature_dim, r)
            weight = recons.contiguous()

        full_key = build_lora_key(layer_key, side)
        reconstructed[full_key] = weight
    return reconstructed


def compute_reconstruction_error(
    original: dict[str, torch.Tensor],
    reconstructed: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compute per-layer and overall reconstruction error (relative Frobenius norm).

    Args:
        original: Original LoRA state dict (full PEFT keys).
        reconstructed: Reconstructed LoRA state dict (full PEFT keys).

    Returns:
        Dict with ``"per_layer"`` errors, ``"mean"`` error, and ``"max"`` error.
    """
    per_layer: dict[str, float] = {}
    for key in original:
        if key not in reconstructed:
            continue
        orig = original[key].to(torch.float32)
        recon = reconstructed[key].to(torch.float32)
        orig_norm = torch.norm(orig)
        if orig_norm < 1e-12:
            per_layer[key] = 0.0
        else:
            per_layer[key] = (torch.norm(orig - recon) / orig_norm).item()

    errors = list(per_layer.values())
    return {
        "per_layer": per_layer,
        "mean": sum(errors) / len(errors) if errors else 0.0,
        "max": max(errors) if errors else 0.0,
    }


def compute_compression_stats(
    components: dict[str, torch.Tensor],
    all_loadings: dict[str, dict[str, torch.Tensor]],
    eigenvalues: dict[str, torch.Tensor],
    num_adapters: int,
    lora_rank: int,
) -> dict[str, Any]:
    """Compute compression statistics for summary display.

    Args:
        components: Shared component tensors.
        all_loadings: Per-adapter loading tensors.
        eigenvalues: Per-group eigenvalues.
        num_adapters: Number of adapters compressed.
        lora_rank: LoRA rank.

    Returns:
        Dict of statistics including param counts, ratios, and explained variance.
    """
    # Count original params: N adapters x layers x (r*d + d*r) per layer pair
    # Count compressed: shared components + all adapter loadings
    shared_params = sum(t.numel() for t in components.values())
    per_adapter_params = 0
    for adapter_loadings in all_loadings.values():
        per_adapter_params += sum(t.numel() for t in adapter_loadings.values())

    compressed_total = shared_params + per_adapter_params

    # Original: each adapter has the same layer structure as components but with rank r
    # For each group_key, original weight has feature_dim * rank params
    original_per_adapter = 0
    for gk, comp in components.items():
        feature_dim = comp.shape[0]
        original_per_adapter += feature_dim * lora_rank
    original_total = original_per_adapter * num_adapters

    # Explained variance per layer
    per_layer_variance: dict[str, float] = {}
    k = next(iter(components.values())).shape[1]
    for gk, evals in eigenvalues.items():
        evals_clamped = evals.clamp(min=0)
        total_var = evals_clamped.sum()
        if total_var > 1e-12:
            explained = evals_clamped[:k].sum() / total_var
            per_layer_variance[gk] = explained.item()
        else:
            per_layer_variance[gk] = 1.0

    variance_values = list(per_layer_variance.values())

    return {
        "original_total_params": original_total,
        "compressed_total_params": compressed_total,
        "compression_ratio": original_total / compressed_total if compressed_total > 0 else float("inf"),
        "shared_component_params": shared_params,
        "per_adapter_loading_params": per_adapter_params,
        "mean_explained_variance": sum(variance_values) / len(variance_values) if variance_values else 0.0,
        "min_explained_variance": min(variance_values) if variance_values else 0.0,
        "per_layer_explained_variance": per_layer_variance,
    }
