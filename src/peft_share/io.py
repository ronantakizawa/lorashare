"""I/O for loading PEFT adapters and saving/loading SHARE format."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from peft_share.config import SHAREConfig

logger = logging.getLogger(__name__)


# ── Load PEFT Adapters ───────────────────────────────────────────────────────


def _is_hub_id(path: str) -> bool:
    """Check if path looks like a HuggingFace Hub repo ID (org/repo format)."""
    # Hub IDs have exactly one slash and no path separators typical of local paths
    if os.path.exists(path):
        return False
    parts = path.split("/")
    return len(parts) == 2 and all(parts)


def _derive_adapter_name(path: str) -> str:
    """Derive adapter name from path or Hub ID."""
    if _is_hub_id(path):
        return path.split("/")[-1]
    return Path(path).name


def load_peft_adapter(
    adapter_path: str,
    adapter_name: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    """Load a PEFT LoRA adapter from local path or HuggingFace Hub.

    Args:
        adapter_path: Local directory path or HuggingFace Hub repo ID.
        adapter_name: Name for this adapter. If None, derived from path.

    Returns:
        Tuple of (state_dict, config_dict, resolved_name).

    Raises:
        FileNotFoundError: If adapter weights or config not found.
        ValueError: If peft_type is not LORA.
    """
    resolved_name = adapter_name or _derive_adapter_name(adapter_path)

    if _is_hub_id(adapter_path):
        config_dict, weights_path = _download_from_hub(adapter_path)
    else:
        config_dict, weights_path = _load_local(adapter_path)

    # Validate it's a LoRA adapter
    peft_type = config_dict.get("peft_type", "")
    if peft_type != "LORA":
        raise ValueError(
            f"Adapter at {adapter_path} has peft_type={peft_type!r}, expected 'LORA'"
        )

    # Load weights
    if weights_path.endswith(".safetensors"):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Filter out classifier/task-head weights
    lora_keys = {k: v for k, v in state_dict.items() if "lora_" in k}
    non_lora_keys = [k for k in state_dict if "lora_" not in k]
    if non_lora_keys:
        logger.info(
            "Adapter %s: skipping %d non-LoRA keys (classifier/head weights)",
            resolved_name,
            len(non_lora_keys),
        )

    return lora_keys, config_dict, resolved_name


def _load_local(adapter_path: str) -> tuple[dict[str, Any], str]:
    """Load config and find weights from a local adapter directory."""
    adapter_dir = Path(adapter_path)

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No adapter_config.json found in {adapter_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Find weights file (prefer safetensors)
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safetensors_path.exists():
        weights_path = str(safetensors_path)
    elif bin_path.exists():
        weights_path = str(bin_path)
    else:
        raise FileNotFoundError(
            f"No adapter_model.safetensors or .bin found in {adapter_path}"
        )

    return config_dict, weights_path


def _download_from_hub(repo_id: str) -> tuple[dict[str, Any], str]:
    """Download adapter config and weights from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id, "adapter_config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    # Try safetensors first, fall back to bin
    try:
        weights_path = hf_hub_download(repo_id, "adapter_model.safetensors")
    except Exception:
        weights_path = hf_hub_download(repo_id, "adapter_model.bin")

    return config_dict, weights_path


# ── Validation ───────────────────────────────────────────────────────────────


def validate_adapters(adapter_configs: dict[str, dict[str, Any]]) -> None:
    """Validate that all adapters are compatible for SHARE compression.

    Checks: all LORA type, same base_model, same rank, same target_modules,
    at least 2 adapters.

    Raises:
        ValueError: With descriptive message about incompatibility.
    """
    if len(adapter_configs) < 2:
        raise ValueError(
            f"Need at least 2 adapters for SHARE compression, got {len(adapter_configs)}"
        )

    names = list(adapter_configs.keys())
    ref_name = names[0]
    ref = adapter_configs[ref_name]

    for name in names[1:]:
        cfg = adapter_configs[name]

        if cfg.get("r") != ref.get("r"):
            raise ValueError(
                f"Rank mismatch: {ref_name} has r={ref.get('r')}, "
                f"{name} has r={cfg.get('r')}"
            )

        if cfg.get("base_model_name_or_path") != ref.get("base_model_name_or_path"):
            raise ValueError(
                f"Base model mismatch: {ref_name} uses "
                f"{ref.get('base_model_name_or_path')!r}, "
                f"{name} uses {cfg.get('base_model_name_or_path')!r}"
            )

        ref_modules = set(ref.get("target_modules", []))
        cfg_modules = set(cfg.get("target_modules", []))
        if ref_modules != cfg_modules:
            raise ValueError(
                f"Target modules mismatch: {ref_name} targets {ref_modules}, "
                f"{name} targets {cfg_modules}"
            )


# ── SHARE Checkpoint I/O ─────────────────────────────────────────────────────


def save_share_checkpoint(
    output_dir: str | Path,
    config: SHAREConfig,
    components: dict[str, torch.Tensor],
    all_loadings: dict[str, dict[str, torch.Tensor]],
    adapter_configs: dict[str, dict[str, Any]],
    adapter_sources: dict[str, str] | None = None,
) -> None:
    """Save SHARE compressed format to disk.

    Layout::

        output_dir/
        ├── share_config.json
        ├── shared_components.safetensors
        └── adapters/
            ├── cola/
            │   ├── loadings.safetensors
            │   └── adapter_meta.json
            └── ...
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(output_dir)

    # Save shared components
    save_file(components, str(out / "shared_components.safetensors"))

    # Save per-adapter loadings
    for adapter_name, loadings in all_loadings.items():
        adapter_dir = out / "adapters" / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        save_file(loadings, str(adapter_dir / "loadings.safetensors"))

        # Save adapter metadata
        meta = {
            "adapter_name": adapter_name,
            "original_config": adapter_configs.get(adapter_name, {}),
        }
        if adapter_sources and adapter_name in adapter_sources:
            meta["original_adapter_path"] = adapter_sources[adapter_name]
        with open(adapter_dir / "adapter_meta.json", "w") as f:
            json.dump(meta, f, indent=2)


def load_share_checkpoint(
    checkpoint_dir: str | Path,
) -> tuple[
    SHAREConfig,
    dict[str, torch.Tensor],
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, Any]],
]:
    """Load SHARE compressed format from disk.

    Returns:
        Tuple of (config, components, all_loadings, adapter_configs).
    """
    ckpt = Path(checkpoint_dir)

    config = SHAREConfig.load(ckpt)
    components = load_file(str(ckpt / "shared_components.safetensors"))

    all_loadings: dict[str, dict[str, torch.Tensor]] = {}
    adapter_configs: dict[str, dict[str, Any]] = {}

    adapters_dir = ckpt / "adapters"
    if adapters_dir.exists():
        for adapter_dir in sorted(adapters_dir.iterdir()):
            if not adapter_dir.is_dir():
                continue
            name = adapter_dir.name
            loadings_path = adapter_dir / "loadings.safetensors"
            if loadings_path.exists():
                all_loadings[name] = load_file(str(loadings_path))

            meta_path = adapter_dir / "adapter_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                adapter_configs[name] = meta.get("original_config", {})

    return config, components, all_loadings, adapter_configs


# ── Reconstructed Adapter I/O ────────────────────────────────────────────────


def save_reconstructed_adapter(
    output_dir: str | Path,
    weights: dict[str, torch.Tensor],
    original_config: dict[str, Any],
) -> None:
    """Save reconstructed LoRA adapter in standard PEFT format.

    Creates ``adapter_config.json`` and ``adapter_model.safetensors``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(weights, str(out / "adapter_model.safetensors"))

    with open(out / "adapter_config.json", "w") as f:
        json.dump(original_config, f, indent=2)


# ── Hub ──────────────────────────────────────────────────────────────────────


def push_share_to_hub(
    checkpoint_dir: str | Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Push SHARE checkpoint to HuggingFace Hub.

    Returns:
        URL of the uploaded repository.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
    )
    return f"https://huggingface.co/{repo_id}"
