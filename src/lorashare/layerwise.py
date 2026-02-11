"""Layer-by-layer processing for memory-efficient compression.

Processes one layer at a time to minimize peak memory usage,
enabling compression of large models that wouldn't fit in memory otherwise.
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
    parse_lora_key,
)
from lorashare.io import load_peft_adapter

logger = logging.getLogger(__name__)


def get_layer_keys(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Extract unique layer keys from state dict."""
    layer_keys = set()
    for key in state_dict:
        try:
            layer_key, side = parse_lora_key(key)
            # Get just the layer part (e.g., "encoder.layer.0")
            # Remove the module part (e.g., "attention.self.query")
            layer_base = layer_key.split(".attention")[0] if ".attention" in layer_key else layer_key
            layer_keys.add(layer_base)
        except ValueError:
            continue
    return sorted(layer_keys)


def compress_layer_by_layer(
    adapter_map: dict[str, str],
    num_components: int | str,
    variance_threshold: float,
    device: str | None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
    int,
]:
    """Compress adapters one layer at a time to minimize memory usage.

    Args:
        adapter_map: Dict mapping adapter names to paths.
        num_components: Number of components or "auto".
        variance_threshold: Target variance for auto selection.
        device: Device for computation.

    Returns:
        Tuple of (components, all_loadings, eigenvalues, k).
    """
    logger.info("Using layer-by-layer processing for memory efficiency")

    # Load first adapter to get layer structure
    first_path = next(iter(adapter_map.values()))
    first_weights, _, _, _ = load_peft_adapter(first_path, adapter_name="temp")
    layer_keys = get_layer_keys(first_weights)
    del first_weights
    gc.collect()

    logger.info(f"Found {len(layer_keys)} layer groups to process")

    all_components: dict[str, torch.Tensor] = {}
    all_loadings: dict[str, dict[str, torch.Tensor]] = {
        name: {} for name in adapter_map.keys()
    }
    all_eigenvalues: dict[str, torch.Tensor] = {}

    for i, layer_key in enumerate(layer_keys):
        logger.info(f"Processing layer {i+1}/{len(layer_keys)}: {layer_key}")

        # Load only this layer from all adapters
        layer_weights: dict[str, dict[str, torch.Tensor]] = {}

        for name, path in adapter_map.items():
            weights, _, _, _ = load_peft_adapter(path, adapter_name=name)

            # Extract weights for this layer only
            layer_subset = {
                k: v
                for k, v in weights.items()
                if layer_key in k and "lora_" in k
            }

            if layer_subset:
                layer_weights[name] = layer_subset

            del weights  # Free immediately
            gc.collect()

        if not layer_weights:
            logger.debug(f"No LoRA weights found for layer {layer_key}, skipping")
            continue

        # Compress this layer
        combined = combine_adapter_weights(layer_weights)

        layer_components, layer_eigenvalues, k = compute_shared_components(
            combined,
            num_components=num_components,
            variance_threshold=variance_threshold,
            device=device,
        )

        all_components.update(layer_components)
        all_eigenvalues.update(layer_eigenvalues)

        # Compute loadings for this layer
        for name, weights in layer_weights.items():
            layer_loadings = compute_adapter_loadings(layer_components, weights)
            all_loadings[name].update(layer_loadings)

        del layer_weights, combined, layer_components, layer_loadings
        gc.collect()

        if device == "cuda":
            torch.cuda.empty_cache()

    return all_components, all_loadings, all_eigenvalues, k
