"""
Train 4 LoRA adapters on roberta-base for GLUE tasks.

Run once before GPU tests to create ground-truth adapters:
    python tests/gpu/train_adapters.py

Produces:
    tests/gpu/adapters/{sst2,cola,mrpc,rte}/
    tests/gpu/adapters/baselines.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load as load_metric
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ADAPTER_DIR = Path(__file__).parent / "adapters"
BASE_MODEL = "roberta-base"

TASKS = {
    "sst2": {
        "dataset": ("glue", "sst2"),
        "text_key": "sentence",
        "text_key_b": None,
        "num_labels": 2,
        "metric": "accuracy",
    },
    "cola": {
        "dataset": ("glue", "cola"),
        "text_key": "sentence",
        "text_key_b": None,
        "num_labels": 2,
        "metric": "matthews_correlation",
    },
    "mrpc": {
        "dataset": ("glue", "mrpc"),
        "text_key": "sentence1",
        "text_key_b": "sentence2",
        "num_labels": 2,
        "metric": "f1",
    },
    "rte": {
        "dataset": ("glue", "rte"),
        "text_key": "sentence1",
        "text_key_b": "sentence2",
        "num_labels": 2,
        "metric": "accuracy",
    },
}

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["query", "value"],
    "lora_dropout": 0.1,
    "bias": "none",
    "modules_to_save": ["classifier"],
}

TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "logging_steps": 50,
    "eval_strategy": "epoch",
    "save_strategy": "no",
    "fp16": torch.cuda.is_available(),
    "report_to": "none",
}


def train_adapter(task_name: str, task_config: dict) -> dict:
    """Train a single LoRA adapter and return evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"Training {task_name.upper()}")
    print(f"{'='*60}")

    output_dir = ADAPTER_DIR / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=task_config["num_labels"]
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        **LORA_CONFIG,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds_name, ds_config = task_config["dataset"]
    dataset = load_dataset(ds_name, ds_config)

    def tokenize(examples):
        if task_config["text_key_b"]:
            return tokenizer(
                examples[task_config["text_key"]],
                examples[task_config["text_key_b"]],
                truncation=True,
                max_length=128,
            )
        return tokenizer(
            examples[task_config["text_key"]],
            truncation=True,
            max_length=128,
        )

    tokenized = dataset.map(tokenize, batched=True)

    metric_loader = load_metric("glue", ds_config)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric_loader.compute(predictions=preds, references=labels)

    tmp_dir = output_dir / "training_tmp"
    training_args = TrainingArguments(
        output_dir=str(tmp_dir),
        **TRAINING_ARGS,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Clean up training tmp dir
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    metric_key = task_config["metric"]
    metric_value = eval_results.get(f"eval_{metric_key}", None)

    print(f"\n{task_name.upper()} — {metric_key}: {metric_value}")
    return {
        "task": task_name,
        "metric_name": metric_key,
        "metric_value": metric_value,
        "eval_results": {k: v for k, v in eval_results.items() if isinstance(v, (int, float))},
    }


def main():
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    baselines = {}

    for task_name, task_config in TASKS.items():
        result = train_adapter(task_name, task_config)
        baselines[task_name] = result

    baselines_path = ADAPTER_DIR / "baselines.json"
    with open(baselines_path, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"\n{'='*60}")
    print("ALL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Adapters saved to: {ADAPTER_DIR}")
    print(f"Baselines saved to: {baselines_path}")
    for task_name, result in baselines.items():
        print(f"  {task_name}: {result['metric_name']} = {result['metric_value']}")


if __name__ == "__main__":
    main()
