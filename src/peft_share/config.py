from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SHAREConfig:
    """Configuration for a SHARE compressed adapter set."""

    share_version: str = "0.1.0"
    num_components: int = 32
    component_selection: str = "fixed"  # "fixed" or "auto"
    variance_threshold: float = 0.95
    base_model_name_or_path: str = ""
    target_modules: list[str] = field(default_factory=list)
    lora_rank: int = 0
    lora_alpha: int = 0
    adapter_names: list[str] = field(default_factory=list)
    num_adapters: int = 0
    layer_keys: list[str] = field(default_factory=list)
    compression_stats: dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir) / "share_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> SHAREConfig:
        path = Path(checkpoint_dir) / "share_config.json"
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
