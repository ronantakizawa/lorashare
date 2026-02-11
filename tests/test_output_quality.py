"""Output quality tests: compare standard PEFT LoRA vs SHARE-reconstructed adapters.

These tests verify that SHARE compression preserves model output quality by
comparing logits, predictions, and distributional metrics between standard
PEFT adapter inference and SHARE-reconstructed adapter inference.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lorashare import SHAREModel


def create_roberta_lora_adapter(
    path: Path,
    hidden_size: int = 768,
    num_labels: int = 2,
    rank: int = 8,
    num_layers: int = 12,
    seed: int = 42,
):
    """Create a PEFT LoRA adapter compatible with roberta-base.

    Uses seeded random weights so adapters are reproducible.
    """
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    weights = {}
    for layer_idx in range(num_layers):
        for module in ["query", "value"]:
            weights[
                f"base_model.model.roberta.encoder.layer.{layer_idx}"
                f".attention.self.{module}.lora_A.default.weight"
            ] = torch.randn(rank, hidden_size) * 0.01
            weights[
                f"base_model.model.roberta.encoder.layer.{layer_idx}"
                f".attention.self.{module}.lora_B.default.weight"
            ] = torch.randn(hidden_size, rank) * 0.01

    # Classifier head
    weights["base_model.model.classifier.dense.weight"] = (
        torch.randn(hidden_size, hidden_size) * 0.1
    )
    weights["base_model.model.classifier.dense.bias"] = (
        torch.randn(hidden_size) * 0.1
    )
    weights["base_model.model.classifier.out_proj.weight"] = (
        torch.randn(num_labels, hidden_size) * 0.1
    )
    weights["base_model.model.classifier.out_proj.bias"] = (
        torch.randn(num_labels) * 0.1
    )

    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "roberta-base",
        "r": rank,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["query", "value"],
        "task_type": "SEQ_CLS",
        "inference_mode": False,
        "modules_to_save": ["classifier"],
    }

    save_file(weights, str(path / "adapter_model.safetensors"))
    with open(path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return weights


def _load_base_model(num_labels: int = 2):
    """Load roberta-base for sequence classification, or skip if unavailable."""
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        pytest.skip("transformers not installed")

    try:
        return AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=num_labels
        )
    except Exception as e:
        pytest.skip(f"Could not load roberta-base: {e}")


def _load_tokenizer():
    """Load roberta-base tokenizer, or skip if unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    try:
        return AutoTokenizer.from_pretrained("roberta-base")
    except Exception as e:
        pytest.skip(f"Could not load tokenizer: {e}")


def _get_standard_logits(base_model, adapter_path: str, input_ids, attention_mask):
    """Get logits from a standard PEFT LoRA adapter applied to a fresh base model."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


def _get_share_logits(share_model, adapter_name, base_model, input_ids, attention_mask):
    """Get logits from a SHARE-reconstructed adapter applied to a fresh base model."""
    model = share_model.apply(base_model, adapter_name)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened)."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def _kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL divergence from distribution P (standard) to Q (SHARE)."""
    p = torch.softmax(logits_p.float(), dim=-1)
    q = torch.softmax(logits_q.float(), dim=-1)
    # Clamp to avoid log(0)
    q = q.clamp(min=1e-8)
    p = p.clamp(min=1e-8)
    return torch.nn.functional.kl_div(
        q.log(), p, reduction="batchmean"
    ).item()


SAMPLE_TEXTS = [
    "The movie was fantastic and I really enjoyed it!",
    "This is a terrible product that broke after one day.",
    "The weather is nice today, perfect for a walk in the park.",
    "I can't believe how disappointing this restaurant was.",
    "She completed the assignment on time and received full marks.",
]


@pytest.mark.slow
class TestOutputQuality:
    """Compare standard PEFT LoRA outputs vs SHARE-reconstructed outputs."""

    @pytest.fixture(autouse=True)
    def _skip_if_deps_missing(self):
        try:
            import transformers  # noqa: F401
            import peft  # noqa: F401
        except ImportError:
            pytest.skip("transformers and peft required for output quality tests")

    @pytest.fixture
    def adapter_paths(self, tmp_path):
        """Create 3 reproducible adapters on disk."""
        paths = {}
        for i, name in enumerate(["adapter_a", "adapter_b", "adapter_c"]):
            p = tmp_path / name
            create_roberta_lora_adapter(p, seed=100 + i)
            paths[name] = str(p)
        return paths

    @pytest.fixture
    def share_model(self, adapter_paths):
        """Compress adapters into a SHARE model with k=16."""
        return SHAREModel.from_adapters(
            list(adapter_paths.values()), num_components=16
        )

    @pytest.fixture
    def tokenizer(self):
        return _load_tokenizer()

    @pytest.fixture
    def sample_inputs(self, tokenizer):
        """Tokenize sample texts into model inputs."""
        return tokenizer(
            SAMPLE_TEXTS,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

    def test_logit_closeness(self, adapter_paths, share_model, sample_inputs):
        """Logits from SHARE adapter should be close to standard PEFT adapter."""
        input_ids = sample_inputs["input_ids"]
        attention_mask = sample_inputs["attention_mask"]

        for adapter_name, adapter_path in adapter_paths.items():
            base_std = _load_base_model()
            base_share = _load_base_model()

            std_logits = _get_standard_logits(
                base_std, adapter_path, input_ids, attention_mask
            )
            share_logits = _get_share_logits(
                share_model, adapter_name, base_share, input_ids, attention_mask
            )

            max_diff = (std_logits - share_logits).abs().max().item()
            assert max_diff < 1.0, (
                f"{adapter_name}: max logit difference {max_diff:.4f} exceeds "
                f"threshold 1.0"
            )

    def test_prediction_agreement(self, adapter_paths, share_model, sample_inputs):
        """Argmax predictions should agree between standard and SHARE adapters."""
        input_ids = sample_inputs["input_ids"]
        attention_mask = sample_inputs["attention_mask"]

        for adapter_name, adapter_path in adapter_paths.items():
            base_std = _load_base_model()
            base_share = _load_base_model()

            std_logits = _get_standard_logits(
                base_std, adapter_path, input_ids, attention_mask
            )
            share_logits = _get_share_logits(
                share_model, adapter_name, base_share, input_ids, attention_mask
            )

            std_preds = std_logits.argmax(dim=-1)
            share_preds = share_logits.argmax(dim=-1)

            agreement = (std_preds == share_preds).float().mean().item()
            assert agreement >= 0.8, (
                f"{adapter_name}: prediction agreement {agreement:.2%} is below "
                f"80% threshold"
            )

    def test_cosine_similarity(self, adapter_paths, share_model, sample_inputs):
        """Cosine similarity between standard and SHARE logits should be high."""
        input_ids = sample_inputs["input_ids"]
        attention_mask = sample_inputs["attention_mask"]

        for adapter_name, adapter_path in adapter_paths.items():
            base_std = _load_base_model()
            base_share = _load_base_model()

            std_logits = _get_standard_logits(
                base_std, adapter_path, input_ids, attention_mask
            )
            share_logits = _get_share_logits(
                share_model, adapter_name, base_share, input_ids, attention_mask
            )

            sim = _cosine_similarity(std_logits, share_logits)
            assert sim > 0.9, (
                f"{adapter_name}: cosine similarity {sim:.4f} is below 0.9"
            )

    def test_kl_divergence(self, adapter_paths, share_model, sample_inputs):
        """KL divergence between standard and SHARE output distributions should be small."""
        input_ids = sample_inputs["input_ids"]
        attention_mask = sample_inputs["attention_mask"]

        for adapter_name, adapter_path in adapter_paths.items():
            base_std = _load_base_model()
            base_share = _load_base_model()

            std_logits = _get_standard_logits(
                base_std, adapter_path, input_ids, attention_mask
            )
            share_logits = _get_share_logits(
                share_model, adapter_name, base_share, input_ids, attention_mask
            )

            kl = _kl_divergence(std_logits, share_logits)
            assert kl < 0.5, (
                f"{adapter_name}: KL divergence {kl:.4f} exceeds threshold 0.5"
            )

    def test_quality_improves_with_more_components(self, adapter_paths):
        """Higher num_components should produce lower weight reconstruction error."""
        adapter_name = "adapter_a"
        adapter_path = adapter_paths[adapter_name]

        # Compare weight-level reconstruction error at different k values.
        # Weight-level error is guaranteed to decrease with more components
        # because PCA captures more variance. (Logit-level comparisons are
        # unreliable here because the classifier head is stored exactly and
        # dominates the output, masking the LoRA reconstruction error.)
        errors = {}
        for k in [2, 16]:
            share = SHAREModel.from_adapters(
                list(adapter_paths.values()), num_components=k
            )
            error = share.reconstruction_error(
                adapter_name, original_path=adapter_path
            )
            errors[k] = error["mean"]

        assert errors[16] <= errors[2], (
            f"Higher k should give lower weight reconstruction error: "
            f"k=2 error={errors[2]:.6f}, k=16 error={errors[16]:.6f}"
        )

    def test_output_shape_preserved(
        self, adapter_paths, share_model, sample_inputs
    ):
        """SHARE adapter outputs should have the same shape as standard outputs."""
        input_ids = sample_inputs["input_ids"]
        attention_mask = sample_inputs["attention_mask"]

        for adapter_name, adapter_path in adapter_paths.items():
            base_std = _load_base_model()
            base_share = _load_base_model()

            std_logits = _get_standard_logits(
                base_std, adapter_path, input_ids, attention_mask
            )
            share_logits = _get_share_logits(
                share_model, adapter_name, base_share, input_ids, attention_mask
            )

            assert std_logits.shape == share_logits.shape, (
                f"{adapter_name}: shape mismatch — standard {std_logits.shape} "
                f"vs SHARE {share_logits.shape}"
            )

    def test_softmax_distribution_close(
        self, adapter_paths, share_model, sample_inputs
    ):
        """Softmax probability distributions should be close."""
        input_ids = sample_inputs["input_ids"]
        attention_mask = sample_inputs["attention_mask"]

        for adapter_name, adapter_path in adapter_paths.items():
            base_std = _load_base_model()
            base_share = _load_base_model()

            std_logits = _get_standard_logits(
                base_std, adapter_path, input_ids, attention_mask
            )
            share_logits = _get_share_logits(
                share_model, adapter_name, base_share, input_ids, attention_mask
            )

            std_probs = torch.softmax(std_logits.float(), dim=-1)
            share_probs = torch.softmax(share_logits.float(), dim=-1)

            max_prob_diff = (std_probs - share_probs).abs().max().item()
            assert max_prob_diff < 0.3, (
                f"{adapter_name}: max probability difference {max_prob_diff:.4f} "
                f"exceeds 0.3"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
