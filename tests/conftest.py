"""Shared test fixtures for peft-share tests."""

import json

import pytest
import torch
from safetensors.torch import save_file


# Small dimensions for fast tests
D = 64   # feature dim (real: 768)
R = 4    # LoRA rank (real: 8)
NUM_LAYERS = 2
MODULES = ["query", "value"]


def _layer_keys() -> list[str]:
    return [
        f"encoder.layer.{i}.attention.self.{m}"
        for i in range(NUM_LAYERS)
        for m in MODULES
    ]


def _make_adapter_weights(seed: int) -> dict[str, torch.Tensor]:
    """Create synthetic LoRA weights for all layers."""
    torch.manual_seed(seed)
    weights = {}
    for lk in _layer_keys():
        weights[f"base_model.model.{lk}.lora_A.weight"] = torch.randn(R, D)
        weights[f"base_model.model.{lk}.lora_B.weight"] = torch.randn(D, R)
    return weights


@pytest.fixture
def layer_keys():
    return _layer_keys()


@pytest.fixture
def synthetic_adapters() -> dict[str, dict[str, torch.Tensor]]:
    """Three synthetic LoRA adapters with different seeds."""
    return {
        "cola": _make_adapter_weights(0),
        "mrpc": _make_adapter_weights(1),
        "rte": _make_adapter_weights(2),
    }


@pytest.fixture
def synthetic_lora_config() -> dict:
    """Standard PEFT LoRA config dict."""
    return {
        "peft_type": "LORA",
        "r": R,
        "lora_alpha": 8,
        "target_modules": MODULES,
        "base_model_name_or_path": "roberta-base",
        "task_type": "SEQ_CLS",
        "bias": "none",
        "inference_mode": False,
    }


@pytest.fixture
def saved_adapters(tmp_path, synthetic_adapters, synthetic_lora_config):
    """Save synthetic adapters to disk in standard PEFT format."""
    adapter_dirs = {}
    for name, weights in synthetic_adapters.items():
        adapter_dir = tmp_path / name
        adapter_dir.mkdir()
        save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(synthetic_lora_config, f)
        adapter_dirs[name] = str(adapter_dir)
    return adapter_dirs
