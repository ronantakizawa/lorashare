"""Chunked adapter processing for scalability to 100+ adapters.

Loads and processes adapters in batches, then merges the results.
Enables compression of arbitrarily many adapters without memory issues.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import torch

from lorashare.compression import (
    combine_adapter_weights,
    compute_adapter_loadings,
    compute_shared_components,
)
from lorashare.io import load_peft_adapter, validate_adapters

logger = logging.getLogger(__name__)


def compress_chunked(
    adapter_map: dict[str, str],
    num_components: int | str,
    variance_threshold: float,
    device: str | None,
    chunk_size: int = 10,
    on_error: str = "raise",
) -> tuple[
    dict[str, torch.Tensor],  # components
    dict[str, dict[str, torch.Tensor]],  # all_loadings
    dict[str, torch.Tensor],  # eigenvalues
    int,  # k
    dict[str, dict[str, Any]],  # configs
    dict[str, dict[str, torch.Tensor]],  # classifier_heads
]:
    """Compress adapters in chunks, then merge results.

    Args:
        adapter_map: Dict mapping adapter names to paths.
        num_components: Number of components or "auto".
        variance_threshold: Target variance for auto selection.
        device: Device for computation.
        chunk_size: Number of adapters to process per chunk.
        on_error: ``"raise"`` to abort on failure, ``"skip"`` to skip bad adapters.

    Returns:
        Tuple of (components, all_loadings, eigenvalues, k, configs, classifier_heads).
    """
    adapter_names = list(adapter_map.keys())
    n_adapters = len(adapter_names)
    n_chunks = (n_adapters + chunk_size - 1) // chunk_size

    logger.info(
        f"Using chunked processing: {n_adapters} adapters in {n_chunks} chunks "
        f"of size {chunk_size}"
    )

    # Process chunks
    chunk_results = []
    all_configs = {}
    all_classifier_heads = {}

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_adapters)
        chunk_names = adapter_names[start:end]

        logger.info(
            f"Processing chunk {i+1}/{n_chunks}: adapters {start+1}-{end} "
            f"({len(chunk_names)} adapters)"
        )

        # Create chunk adapter map
        chunk_map = {name: adapter_map[name] for name in chunk_names}

        # Load chunk adapters
        chunk_weights = {}
        chunk_configs = {}
        chunk_classifier_heads = {}

        for name, path in chunk_map.items():
            try:
                weights, classifier_heads, config_dict, _ = load_peft_adapter(
                    path, adapter_name=name
                )
            except Exception as e:
                if on_error == "skip":
                    logger.warning("Skipping adapter %s: %s", name, e)
                    continue
                raise
            chunk_weights[name] = weights
            chunk_configs[name] = config_dict
            all_configs[name] = config_dict
            if classifier_heads:
                chunk_classifier_heads[name] = classifier_heads
                all_classifier_heads[name] = classifier_heads

        # Skip empty chunks (all adapters in chunk failed to load)
        if not chunk_weights:
            logger.warning("Chunk %d has no valid adapters, skipping", i + 1)
            continue

        # Validate this chunk
        if not chunk_results:
            # First chunk with results — needs at least 2 configs for validation,
            # so combine with any previously loaded configs from skipped chunks
            combined_configs = {**all_configs, **chunk_configs}
            if len(combined_configs) >= 2:
                validate_adapters(combined_configs)
        else:
            # Validate new chunk against first successful adapter
            ref_name = next(iter(all_configs))
            validate_adapters({**{ref_name: all_configs[ref_name]}, **chunk_configs})

        # Compress this chunk
        combined = combine_adapter_weights(chunk_weights)
        chunk_components, chunk_eigenvalues, k = compute_shared_components(
            combined,
            num_components=num_components,
            variance_threshold=variance_threshold,
            device=device,
        )

        # Compute loadings for this chunk
        chunk_loadings = {}
        for name, weights in chunk_weights.items():
            chunk_loadings[name] = compute_adapter_loadings(chunk_components, weights)

        chunk_results.append({
            "components": chunk_components,
            "eigenvalues": chunk_eigenvalues,
            "loadings": chunk_loadings,
            "k": k,
        })

        # Free memory
        del chunk_weights, combined, chunk_components, chunk_loadings
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if not chunk_results:
        raise ValueError(
            "No chunks had enough adapters to compress. "
            "Need at least 2 valid adapters."
        )

    # Verify total adapter count across all chunks
    total_adapters = sum(
        len(cr["loadings"]) for cr in chunk_results
    )
    if total_adapters < 2:
        raise ValueError(
            f"Need at least 2 valid adapters, got {total_adapters} "
            f"after skipping failures"
        )

    logger.info("All chunks processed, merging results...")

    # Merge chunks
    merged_components, merged_loadings, merged_eigenvalues, final_k = merge_chunks(
        chunk_results, num_components, variance_threshold, device
    )

    logger.info(f"Merge complete, final k={final_k}")

    return (
        merged_components,
        merged_loadings,
        merged_eigenvalues,
        final_k,
        all_configs,
        all_classifier_heads,
    )


def merge_chunks(
    chunk_results: list[dict],
    num_components: int | str,
    variance_threshold: float,
    device: str | None,
) -> tuple[
    dict[str, torch.Tensor],  # components
    dict[str, dict[str, torch.Tensor]],  # all_loadings
    dict[str, torch.Tensor],  # eigenvalues
    int,  # k
]:
    """Merge multiple chunk results into a single SHARE model.

    Uses hierarchical merging:
    1. Combine components from all chunks
    2. Compute meta-PCA (components of components)
    3. Re-project all adapter loadings onto merged components

    Args:
        chunk_results: List of dicts from compress_chunked chunks.
        num_components: Target number of components.
        variance_threshold: Target variance for auto selection.
        device: Device for computation.

    Returns:
        Tuple of (merged_components, merged_loadings, merged_eigenvalues, k).
    """
    # Collect all components by group key
    all_components_by_group: dict[str, list[torch.Tensor]] = {}

    for chunk in chunk_results:
        for group_key, comp in chunk["components"].items():
            if group_key not in all_components_by_group:
                all_components_by_group[group_key] = []
            all_components_by_group[group_key].append(comp)

    # Meta-PCA: compute components of components
    logger.info("Computing meta-PCA on chunk components...")

    merged_components = {}
    merged_eigenvalues = {}

    for group_key, comp_list in all_components_by_group.items():
        # Stack all chunk components: (d, k1 + k2 + k3 + ...)
        stacked = torch.cat(comp_list, dim=1)

        # Move to device for SVD
        stacked = stacked.to(device if device else "cpu")

        # SVD on the stacked components
        U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)

        # Determine k for this group
        if num_components == "auto":
            # Use variance threshold
            variance = (S ** 2) / (S ** 2).sum()
            cumsum = torch.cumsum(variance, dim=0)
            k = int((cumsum < variance_threshold).sum()) + 1
            k = min(k, len(S))
        else:
            k = min(int(num_components), len(S))

        # Take top-k components
        merged_components[group_key] = U[:, :k].cpu().contiguous()
        merged_eigenvalues[group_key] = S[:k].cpu()

        del stacked, U, S, Vh
        gc.collect()

    # Determine final k (use minimum across all groups)
    final_k = min(comp.shape[1] for comp in merged_components.values())

    # Trim all components to final_k
    for group_key in merged_components:
        merged_components[group_key] = merged_components[group_key][:, :final_k].contiguous()
        merged_eigenvalues[group_key] = merged_eigenvalues[group_key][:final_k]

    logger.info("Meta-PCA complete, re-projecting adapter loadings...")

    # Re-project all adapter loadings onto merged components
    merged_loadings: dict[str, dict[str, torch.Tensor]] = {}

    for chunk_idx, chunk in enumerate(chunk_results):
        logger.debug(f"Re-projecting chunk {chunk_idx + 1}/{len(chunk_results)}")

        for adapter_name, loadings in chunk["loadings"].items():
            if adapter_name not in merged_loadings:
                merged_loadings[adapter_name] = {}

            for group_key, loading in loadings.items():
                # loading: (old_k, r)
                old_comp = chunk["components"][group_key]  # (d, old_k)
                new_comp = merged_components[group_key]  # (d, final_k)

                # Reconstruct to feature space: old_comp @ loading -> (d, r)
                reconstructed = old_comp @ loading

                # Project onto merged components: new_comp.T @ reconstructed -> (final_k, r)
                new_loading = new_comp.T @ reconstructed

                merged_loadings[adapter_name][group_key] = new_loading.contiguous()

    return merged_components, merged_loadings, merged_eigenvalues, final_k
