"""Shared fixtures for GPU-based lorashare tests.

These tests require:
1. A CUDA-capable GPU
2. Pre-trained adapters from train_adapters.py
"""

import json
from pathlib import Path

import pytest
import torch

ADAPTER_DIR = Path(__file__).parent / "adapters"
BASELINES_PATH = ADAPTER_DIR / "baselines.json"
BASE_MODEL = "roberta-base"
TASK_NAMES = ["sst2", "cola", "mrpc", "rte"]

# GLUE task metadata for evaluation
GLUE_TASKS = {
    "sst2": {
        "dataset": ("glue", "sst2"),
        "text_key": "sentence",
        "text_key_b": None,
        "metric": "accuracy",
        "split": "validation",
    },
    "cola": {
        "dataset": ("glue", "cola"),
        "text_key": "sentence",
        "text_key_b": None,
        "metric": "matthews_correlation",
        "split": "validation",
    },
    "mrpc": {
        "dataset": ("glue", "mrpc"),
        "text_key": "sentence1",
        "text_key_b": "sentence2",
        "metric": "f1",
        "split": "validation",
    },
    "rte": {
        "dataset": ("glue", "rte"),
        "text_key": "sentence1",
        "text_key_b": "sentence2",
        "metric": "accuracy",
        "split": "validation",
    },
}


def _check_adapters_exist():
    """Check if pre-trained adapters exist."""
    for task in TASK_NAMES:
        adapter_dir = ADAPTER_DIR / task
        if not (adapter_dir / "adapter_model.safetensors").exists():
            return False
    if not BASELINES_PATH.exists():
        return False
    return True


requires_adapters = pytest.mark.skipif(
    not _check_adapters_exist(),
    reason="Pre-trained adapters not found. Run: python tests/gpu/train_adapters.py",
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU not available",
)


@pytest.fixture(scope="session")
def adapter_paths() -> dict[str, str]:
    """Paths to pre-trained GLUE adapters."""
    if not _check_adapters_exist():
        pytest.skip("Pre-trained adapters not found")
    return {task: str(ADAPTER_DIR / task) for task in TASK_NAMES}


@pytest.fixture(scope="session")
def baselines() -> dict:
    """Baseline metrics from training."""
    if not BASELINES_PATH.exists():
        pytest.skip("Baselines file not found")
    with open(BASELINES_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def share_model(adapter_paths):
    """Compressed SHARE model from all 4 adapters."""
    from lorashare import SHAREModel

    return SHAREModel.from_adapters(
        adapter_paths,
        num_components=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@pytest.fixture(scope="session")
def tokenizer():
    """RoBERTa tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL)


@pytest.fixture(scope="session")
def base_model():
    """RoBERTa base model (not wrapped with PEFT)."""
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2
    )


def evaluate_adapter(model, tokenizer, task_name: str, max_samples: int = 256) -> dict:
    """Evaluate a model on a GLUE task and return metrics.

    Args:
        model: A model with a forward() method (base + adapter applied).
        tokenizer: The tokenizer.
        task_name: One of sst2, cola, mrpc, rte.
        max_samples: Max validation samples to evaluate (for speed).

    Returns:
        Dict with metric name and value.
    """
    import numpy as np
    from datasets import load_dataset
    from evaluate import load as load_metric

    task_config = GLUE_TASKS[task_name]
    ds_name, ds_config = task_config["dataset"]
    dataset = load_dataset(ds_name, ds_config, split=task_config["split"])

    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    metric = load_metric("glue", ds_config)
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for example in dataset:
            if task_config["text_key_b"]:
                inputs = tokenizer(
                    example[task_config["text_key"]],
                    example[task_config["text_key_b"]],
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(device)
            else:
                inputs = tokenizer(
                    example[task_config["text_key"]],
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(device)

            outputs = model(**inputs)
            pred = outputs.logits.argmax(dim=-1).item()
            all_preds.append(pred)
            all_labels.append(example["label"])

    results = metric.compute(
        predictions=np.array(all_preds),
        references=np.array(all_labels),
    )
    return results


def get_original_model(adapter_path: str, device: str = "cuda"):
    """Load original PEFT model for comparison."""
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()
    return model
