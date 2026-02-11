"""SHAREModel: main API for compressing and reconstructing LoRA adapters."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import torch

from lorashare.compression import (
    combine_adapter_weights,
    compute_adapter_loadings,
    compute_compression_stats,
    compute_explained_variance,
    compute_reconstruction_error,
    compute_shared_components,
    reconstruct_adapter_weights,
)
from lorashare.config import SHAREConfig
from lorashare.io import (
    load_peft_adapter,
    load_share_checkpoint,
    push_share_to_hub,
    save_reconstructed_adapter,
    save_share_checkpoint,
    validate_adapters,
)

logger = logging.getLogger(__name__)


class SHAREModel:
    """Compressed representation of multiple PEFT LoRA adapters in a shared subspace.

    Use :meth:`from_adapters` to compress existing LoRA adapters, then
    :meth:`reconstruct`, :meth:`apply`, or :meth:`save_pretrained`.

    Example::

        share = SHAREModel.from_adapters(
            ["path/to/cola_lora", "path/to/mrpc_lora"],
            num_components=32,
        )
        share.summary()
        share.reconstruct("cola_lora", output_dir="./reconstructed/cola")
        share.save_pretrained("./my_share_checkpoint")
    """

    def __init__(
        self,
        config: SHAREConfig,
        components: dict[str, torch.Tensor],
        all_loadings: dict[str, dict[str, torch.Tensor]],
        adapter_configs: dict[str, dict[str, Any]],
        all_classifier_heads: dict[str, dict[str, torch.Tensor]] | None = None,
        adapter_sources: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.components = components
        self.all_loadings = all_loadings
        self.adapter_configs = adapter_configs
        self.all_classifier_heads = all_classifier_heads or {}
        self.adapter_sources: dict[str, str] = adapter_sources or {}

    @classmethod
    def from_adapters(
        cls,
        adapters: list[str] | dict[str, str],
        num_components: int | str = 32,
        variance_threshold: float = 0.95,
        device: str | None = None,
        layer_by_layer: bool = False,
        chunk_size: int | None = None,
        on_error: str = "raise",
    ) -> SHAREModel:
        """Compress multiple PEFT LoRA adapters into a shared subspace.

        Args:
            adapters: List of adapter paths/Hub IDs, or dict mapping
                ``{name: path}`` for explicit naming.
            num_components: Number of shared components per layer, or ``"auto"``.
            variance_threshold: Target explained variance when ``num_components="auto"``.
            device: Device for computation ("cuda", "cpu", or None for auto).
                Using "cuda" provides 10-100x speedup for eigendecomposition.
            layer_by_layer: Process one layer at a time to reduce peak memory usage.
                Recommended for large models or limited memory.
            chunk_size: Process adapters in chunks of this size. Enables compression
                of 100+ adapters. If None, all adapters loaded at once.
                Recommended: 10-20 for 100+ adapters.
            on_error: Error handling strategy for individual adapter failures.
                ``"raise"`` (default) aborts on any failure.
                ``"skip"`` logs a warning and continues without the failed adapter.

        Returns:
            SHAREModel with compressed adapters.

        Raises:
            ValueError: If adapters are incompatible, or if fewer than 2
                adapters remain after skipping failures.
        """
        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

        # Normalize to dict[name, path]
        if isinstance(adapters, list):
            adapter_map: dict[str, str] = {}
            for path in adapters:
                try:
                    _, _, _, name = load_peft_adapter(path)
                except Exception as e:
                    if on_error == "skip":
                        logger.warning("Skipping adapter at %s: %s", path, e)
                        continue
                    raise
                # Handle duplicate names
                base_name = name
                counter = 1
                while name in adapter_map:
                    name = f"{base_name}_{counter}"
                    counter += 1
                adapter_map[name] = path
        else:
            adapter_map = dict(adapters)

        if chunk_size is not None:
            # Chunked processing for 100+ adapters
            from lorashare.chunked import compress_chunked

            (
                components,
                all_loadings,
                eigenvalues,
                k,
                all_configs,
                all_classifier_heads,
            ) = compress_chunked(
                adapter_map,
                num_components=num_components,
                variance_threshold=variance_threshold,
                device=device,
                chunk_size=chunk_size,
                on_error=on_error,
            )
            logger.info(f"Selected {k} components")

        elif layer_by_layer:
            # Layer-by-layer processing for memory efficiency
            from lorashare.layerwise import compress_layer_by_layer

            # Still need to load configs for validation
            logger.info("Loading adapter configs for validation...")
            all_configs: dict[str, dict[str, Any]] = {}
            all_classifier_heads: dict[str, dict[str, torch.Tensor]] = {}
            skipped: list[str] = []
            for name, path in list(adapter_map.items()):
                try:
                    _, classifier_heads, config_dict, _ = load_peft_adapter(path, adapter_name=name)
                except Exception as e:
                    if on_error == "skip":
                        logger.warning("Skipping adapter %s: %s", name, e)
                        skipped.append(name)
                        continue
                    raise
                all_configs[name] = config_dict
                if classifier_heads:
                    all_classifier_heads[name] = classifier_heads

            # Remove skipped adapters from the map
            for name in skipped:
                adapter_map.pop(name, None)

            if len(all_configs) < 2:
                raise ValueError(
                    f"Need at least 2 valid adapters, got {len(all_configs)} "
                    f"after skipping failures"
                )

            # Validate compatibility
            logger.info("Validating adapter compatibility...")
            validate_adapters(all_configs)

            # Compress layer by layer
            components, all_loadings, eigenvalues, k = compress_layer_by_layer(
                adapter_map,
                num_components=num_components,
                variance_threshold=variance_threshold,
                device=device,
            )
            logger.info(f"Selected {k} components")

        else:
            # Standard processing - load all adapters into memory
            # Step 1: Load all adapters
            logger.info(f"Loading {len(adapter_map)} adapters...")
            all_weights: dict[str, dict[str, torch.Tensor]] = {}
            all_configs: dict[str, dict[str, Any]] = {}
            all_classifier_heads: dict[str, dict[str, torch.Tensor]] = {}
            for name, path in list(adapter_map.items()):
                try:
                    logger.debug(f"Loading adapter: {name} from {path}")
                    weights, classifier_heads, config_dict, _ = load_peft_adapter(path, adapter_name=name)
                except Exception as e:
                    if on_error == "skip":
                        logger.warning("Skipping adapter %s: %s", name, e)
                        adapter_map.pop(name, None)
                        continue
                    raise
                all_weights[name] = weights
                all_configs[name] = config_dict
                if classifier_heads:
                    all_classifier_heads[name] = classifier_heads

            if len(all_weights) < 2:
                raise ValueError(
                    f"Need at least 2 valid adapters, got {len(all_weights)} "
                    f"after skipping failures"
                )

            # Step 2: Validate compatibility
            logger.info("Validating adapter compatibility...")
            validate_adapters(all_configs)

            # Step 3: Group weights by (layer, side)
            logger.info("Grouping weights by layer...")
            combined = combine_adapter_weights(all_weights)

            # Step 4-5: Compute shared components
            logger.info(f"Computing shared components (k={num_components})...")
            components, eigenvalues, k = compute_shared_components(
                combined,
                num_components=num_components,
                variance_threshold=variance_threshold,
                device=device,
            )
            logger.info(f"Selected {k} components")

            # Step 6: Project each adapter onto shared components
            logger.info("Computing per-adapter loadings...")
            all_loadings: dict[str, dict[str, torch.Tensor]] = {}
            for name, weights in all_weights.items():
                logger.debug(f"Computing loadings for: {name}")
                all_loadings[name] = compute_adapter_loadings(components, weights)

        # Extract metadata from first adapter config
        ref_config = next(iter(all_configs.values()))
        layer_keys = sorted(
            {pk.rsplit(":", 1)[0] for pk in components.keys()}
        )

        # Compute stats
        stats = compute_compression_stats(
            components, all_loadings, eigenvalues,
            num_adapters=len(all_loadings),
            lora_rank=ref_config.get("r", 0),
        )

        config = SHAREConfig(
            num_components=k,
            component_selection="auto" if num_components == "auto" else "fixed",
            variance_threshold=variance_threshold,
            base_model_name_or_path=ref_config.get("base_model_name_or_path", ""),
            target_modules=ref_config.get("target_modules", []),
            lora_rank=ref_config.get("r", 0),
            lora_alpha=ref_config.get("lora_alpha", 0),
            adapter_names=list(all_loadings.keys()),
            num_adapters=len(all_loadings),
            layer_keys=layer_keys,
            compression_stats=stats,
        )

        return cls(
            config=config,
            components=components,
            all_loadings=all_loadings,
            adapter_configs=all_configs,
            all_classifier_heads=all_classifier_heads,
            adapter_sources=dict(adapter_map),
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> SHAREModel:
        """Load a previously saved SHARE checkpoint.

        Args:
            path: Local directory containing ``share_config.json``,
                ``shared_components.safetensors``, and ``adapters/`` subdirectory.
        """
        config, components, all_loadings, adapter_configs, all_classifier_heads, adapter_sources = load_share_checkpoint(path)
        return cls(
            config=config,
            components=components,
            all_loadings=all_loadings,
            adapter_configs=adapter_configs,
            all_classifier_heads=all_classifier_heads,
            adapter_sources=adapter_sources,
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        """Save SHARE checkpoint to disk."""
        save_share_checkpoint(
            output_dir,
            self.config,
            self.components,
            self.all_loadings,
            self.adapter_configs,
            self.all_classifier_heads,
            adapter_sources=self.adapter_sources,
        )

    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool = False,
    ) -> str:
        """Push SHARE checkpoint to HuggingFace Hub.

        Returns:
            URL of the Hub repository.
        """
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp)
            return push_share_to_hub(tmp, repo_id, token=token, private=private)

    def reconstruct(
        self,
        adapter_name: str,
        output_dir: str | Path | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reconstruct a single adapter's LoRA weights from the shared subspace.

        If ``output_dir`` is provided, saves as a standard PEFT LoRA adapter.

        Args:
            adapter_name: Name of the adapter to reconstruct.
            output_dir: Optional path to save reconstructed adapter.

        Returns:
            State dict with standard PEFT LoRA keys (including classifier heads).

        Raises:
            KeyError: If adapter_name not found.
        """
        if adapter_name not in self.all_loadings:
            raise KeyError(
                f"Adapter {adapter_name!r} not found. "
                f"Available: {list(self.all_loadings.keys())}"
            )

        loadings = self.all_loadings[adapter_name]
        weights = reconstruct_adapter_weights(self.components, loadings)

        # Merge classifier heads if available
        classifier_heads = self.all_classifier_heads.get(adapter_name, {})
        full_weights = dict(weights)
        if classifier_heads:
            full_weights.update(classifier_heads)

        if output_dir is not None:
            original_config = self.adapter_configs.get(adapter_name, {})
            save_reconstructed_adapter(output_dir, weights, original_config, classifier_heads)

        return full_weights

    def apply(
        self,
        base_model: Any,
        adapter_name: str,
    ) -> Any:
        """Apply a reconstructed adapter to a base model.

        Reconstructs LoRA weights and loads them via ``PeftModel.from_pretrained()``.

        Args:
            base_model: A HuggingFace PreTrainedModel instance.
            adapter_name: Name of the adapter to apply.

        Returns:
            A ``peft.PeftModel`` with the reconstructed LoRA adapter applied.
        """
        from peft import PeftModel

        loadings = self.all_loadings[adapter_name]
        lora_weights = reconstruct_adapter_weights(self.components, loadings)
        original_config = self.adapter_configs.get(adapter_name, {})
        classifier_heads = self.all_classifier_heads.get(adapter_name, {})

        with tempfile.TemporaryDirectory() as tmp:
            save_reconstructed_adapter(tmp, lora_weights, original_config, classifier_heads)
            model = PeftModel.from_pretrained(base_model, tmp)

        return model

    def summary(self) -> None:
        """Print a summary table of compression statistics."""
        stats = self.config.compression_stats
        cfg = self.config

        print(f"\n{'='*60}")
        print("  SHARE Compression Summary")
        print(f"{'='*60}")
        print(f"  Base model:       {cfg.base_model_name_or_path}")
        print(f"  Adapters:         {cfg.num_adapters} ({', '.join(cfg.adapter_names)})")
        print(f"  LoRA rank:        {cfg.lora_rank}")
        print(f"  Components (k):   {cfg.num_components}")
        print(f"  Target modules:   {cfg.target_modules}")
        print()

        if stats:
            orig = stats.get("original_total_params", 0)
            comp = stats.get("compressed_total_params", 0)
            ratio = stats.get("compression_ratio", 0)
            shared = stats.get("shared_component_params", 0)
            per_adapter = stats.get("per_adapter_loading_params", 0)
            mean_var = stats.get("mean_explained_variance", 0)
            min_var = stats.get("min_explained_variance", 0)

            print(f"  Original params:  {orig:,}")
            print(f"  Compressed:       {comp:,} ({ratio:.1f}x compression)")
            print(f"    Shared:         {shared:,}")
            print(f"    Per-adapter:    {per_adapter:,}")
            print()
            orig_mb = orig * 4 / 1024 / 1024  # float32
            comp_mb = comp * 4 / 1024 / 1024
            print(f"  Original size:    {orig_mb:.2f} MB (float32)")
            print(f"  Compressed size:  {comp_mb:.2f} MB (float32)")
            print()
            print(f"  Explained variance (mean): {mean_var:.4f}")
            print(f"  Explained variance (min):  {min_var:.4f}")

        print(f"{'='*60}\n")

    @property
    def adapter_names(self) -> list[str]:
        """List of adapter names in this SHARE model."""
        return list(self.all_loadings.keys())

    def reconstruction_error(
        self,
        adapter_name: str,
        original_weights: dict[str, torch.Tensor] | None = None,
        original_path: str | None = None,
    ) -> dict[str, Any]:
        """Compute reconstruction error for an adapter.

        Must provide either ``original_weights`` or ``original_path``
        to compare against.

        Returns:
            Dict with ``"per_layer"`` errors, ``"mean"`` error, and ``"max"`` error.
        """
        if original_weights is None and original_path is not None:
            original_weights, _, _, _ = load_peft_adapter(original_path)

        if original_weights is None:
            raise ValueError(
                "Must provide either original_weights or original_path"
            )

        reconstructed = self.reconstruct(adapter_name)
        return compute_reconstruction_error(original_weights, reconstructed)

    def add_adapter(
        self,
        adapter: str,
        name: str | None = None,
        variance_threshold: float | None = None,
        device: str | None = None,
        force_recompute: bool = False,
    ) -> dict[str, Any]:
        """Add a new adapter to the compressed model.

        Tries a fast path first: project the new adapter onto existing shared
        components and check explained variance. If the existing subspace
        captures the new adapter well enough, only loadings are computed
        (one matrix multiply per layer).

        If the subspace is insufficient, falls back to a full recomputation:
        reconstruct all existing adapters, combine with the new one, and
        re-run PCA from scratch.

        Args:
            adapter: Path or HuggingFace Hub ID of the new PEFT LoRA adapter.
            name: Override the inferred adapter name.
            variance_threshold: Minimum explained variance to accept the fast
                path. Defaults to ``self.config.variance_threshold``.
            device: Device for computation ("cuda", "cpu", or None for auto).
            force_recompute: Skip the fast path and always recompute.

        Returns:
            Dict with ``"method"`` (``"projection"`` or ``"recompute"``),
            ``"adapter_name"``, and variance diagnostics.

        Raises:
            ValueError: If adapter name already exists or adapter is incompatible.
        """
        if variance_threshold is None:
            variance_threshold = self.config.variance_threshold

        # Load the new adapter
        weights, classifier_heads, config_dict, inferred_name = load_peft_adapter(
            adapter, adapter_name=name,
        )
        adapter_name = name or inferred_name

        # Reject duplicate names
        if adapter_name in self.all_loadings:
            raise ValueError(
                f"Adapter {adapter_name!r} already exists. "
                f"Use the 'name' parameter to provide a different name."
            )

        # Validate compatibility against an existing adapter
        ref_name = self.adapter_names[0]
        ref_config = self.adapter_configs[ref_name]
        validate_adapters({ref_name: ref_config, adapter_name: config_dict})

        # ── Fast path: project onto existing components ──────────────────
        if not force_recompute:
            variances = compute_explained_variance(self.components, weights)
            min_variance = min(variances.values()) if variances else 0.0
            mean_variance = (
                sum(variances.values()) / len(variances) if variances else 0.0
            )

            if min_variance >= variance_threshold:
                logger.info(
                    f"Fast path: new adapter explained variance "
                    f"{min_variance:.4f} >= {variance_threshold:.4f}"
                )
                loadings = compute_adapter_loadings(self.components, weights)

                self.all_loadings[adapter_name] = loadings
                self.adapter_configs[adapter_name] = config_dict
                if classifier_heads:
                    self.all_classifier_heads[adapter_name] = classifier_heads
                self.adapter_sources[adapter_name] = adapter

                self.config.adapter_names.append(adapter_name)
                self.config.num_adapters += 1

                return {
                    "method": "projection",
                    "adapter_name": adapter_name,
                    "min_explained_variance": min_variance,
                    "mean_explained_variance": mean_variance,
                }

        # ── Slow path: full recomputation ────────────────────────────────
        logger.info(
            "Recomputing shared subspace with new adapter included..."
        )

        # Re-load existing adapters from original sources when available,
        # falling back to reconstructed weights to avoid double compression.
        all_weights: dict[str, dict[str, torch.Tensor]] = {}
        all_classifier_heads = dict(self.all_classifier_heads)
        all_configs = dict(self.adapter_configs)

        for existing_name in list(self.all_loadings.keys()):
            source = self.adapter_sources.get(existing_name)
            if source is not None:
                try:
                    orig_w, orig_cls, _, _ = load_peft_adapter(
                        source, adapter_name=existing_name,
                    )
                    all_weights[existing_name] = orig_w
                    if orig_cls:
                        all_classifier_heads[existing_name] = orig_cls
                    logger.debug(f"Re-loaded {existing_name} from {source}")
                    continue
                except Exception:
                    logger.debug(
                        f"Could not re-load {existing_name} from {source}, "
                        "using reconstructed weights"
                    )
            all_weights[existing_name] = reconstruct_adapter_weights(
                self.components, self.all_loadings[existing_name],
            )

        # Add the new adapter (original weights, not reconstructed)
        all_weights[adapter_name] = weights

        if classifier_heads:
            all_classifier_heads[adapter_name] = classifier_heads

        all_configs[adapter_name] = config_dict

        # Re-run full compression pipeline
        combined = combine_adapter_weights(all_weights)
        components, eigenvalues, k = compute_shared_components(
            combined,
            num_components=self.config.num_components,
            variance_threshold=variance_threshold,
            device=device,
        )

        all_loadings: dict[str, dict[str, torch.Tensor]] = {}
        for a_name, a_weights in all_weights.items():
            all_loadings[a_name] = compute_adapter_loadings(components, a_weights)

        # Recompute stats
        ref_config = next(iter(all_configs.values()))
        stats = compute_compression_stats(
            components, all_loadings, eigenvalues,
            num_adapters=len(all_loadings),
            lora_rank=ref_config.get("r", 0),
        )
        layer_keys = sorted(
            {pk.rsplit(":", 1)[0] for pk in components.keys()}
        )

        # Track source for the new adapter
        adapter_sources = dict(self.adapter_sources)
        adapter_sources[adapter_name] = adapter

        # Update all internal state
        self.components = components
        self.all_loadings = all_loadings
        self.adapter_configs = all_configs
        self.all_classifier_heads = all_classifier_heads
        self.adapter_sources = adapter_sources
        self.config.num_components = k
        self.config.adapter_names = list(all_loadings.keys())
        self.config.num_adapters = len(all_loadings)
        self.config.layer_keys = layer_keys
        self.config.compression_stats = stats

        return {
            "method": "recompute",
            "adapter_name": adapter_name,
        }

    def remove_adapter(self, adapter_name: str) -> None:
        """Remove an adapter from the compressed model.

        Removes the adapter's loadings, config, and classifier heads.
        Shared components are kept as-is (still valid for remaining adapters).

        Args:
            adapter_name: Name of the adapter to remove.

        Raises:
            KeyError: If adapter_name not found.
            ValueError: If removing would leave zero adapters.
        """
        if adapter_name not in self.all_loadings:
            raise KeyError(
                f"Adapter {adapter_name!r} not found. "
                f"Available: {list(self.all_loadings.keys())}"
            )

        if len(self.all_loadings) <= 1:
            raise ValueError(
                "Cannot remove the last adapter. "
                "A SHARE model requires at least one adapter."
            )

        del self.all_loadings[adapter_name]
        self.adapter_configs.pop(adapter_name, None)
        self.all_classifier_heads.pop(adapter_name, None)
        self.adapter_sources.pop(adapter_name, None)

        self.config.adapter_names = list(self.all_loadings.keys())
        self.config.num_adapters = len(self.all_loadings)
