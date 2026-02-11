# lorashare

A Python library that lets you use multiple LoRA adapters with **100x memory savings**. 

Based on the [SHARE paper](https://arxiv.org/abs/2602.06043) (Kaushik et al., 2026).

## The Key Insight

LoRA adapters trained on different tasks share a common low-rank subspace. 

Instead of storing N separate adapters, you can extract the shared principal components through PCA and keep only tiny per-adapter coefficients.

**lorashare** lets you train your LoRAs and then compress them so you can store several task-specific models with the memory size of one adapter. 

## Use Cases

Use **lorashare** if you have:
  - Multiple LoRA adapters (2+)
  - Same base model 
  - Same rank
  - Same target modules
  - Want to store them together

## Install

```bash
pip install lorashare
```

## Quick Start

### Python API

```python
from lorashare import SHAREModel

# Compress multiple LoRA adapters into shared subspace
share = SHAREModel.from_adapters(
    ["path/to/cola_lora", "path/to/mrpc_lora", "path/to/rte_lora"],
    num_components=32,  # or "auto" for explained-variance selection
)

# See compression stats
share.summary()

# Reconstruct any adapter as standard PEFT LoRA
share.reconstruct("cola_lora", output_dir="./reconstructed/cola")

# Apply to base model for inference (returns standard PeftModel)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = share.apply(base_model, adapter_name="cola_lora")

# Run inference with the reconstructed adapter
text = "The movie was fantastic and I really enjoyed it!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    print(f"Predictions: {predictions}")

# Save / load
share.save_pretrained("./my_share_checkpoint")
share = SHAREModel.from_pretrained("./my_share_checkpoint")

# Push to HuggingFace Hub
share.push_to_hub("username/my-share-model")
```

### CLI

```bash
# Compress
lorashare compress adapter1/ adapter2/ adapter3/ -o ./compressed -k 32

# With GPU acceleration (10-100x faster)
lorashare compress adapter1/ adapter2/ adapter3/ -o ./compressed -k 32 --device cuda

# For large models (reduced memory)
lorashare compress adapter1/ adapter2/ adapter3/ -o ./compressed -k 32 --layer-by-layer

# For 100+ adapters (chunked processing)
lorashare compress adapter1/ ... adapter100/ -o ./compressed -k 32 --chunk-size 10

# Inspect
lorashare info ./compressed

# Reconstruct single adapter
lorashare reconstruct ./compressed --adapter cola -o ./reconstructed

# Reconstruct all
lorashare reconstruct ./compressed --all -o ./reconstructed
```

### Scalability Features (v0.2.0)

```python
# GPU Acceleration (10-100x speedup)
share = SHAREModel.from_adapters(
    adapters,
    num_components=32,
    device="cuda",  # or "cpu" or None for auto
)

# Layer-by-layer processing (70% less memory)
share = SHAREModel.from_adapters(
    adapters,
    num_components=32,
    layer_by_layer=True,  # Process one layer at a time
)

# Chunked loading (enables 100+ adapters)
share = SHAREModel.from_adapters(
    many_adapters,  # 100+ adapters
    num_components=32,
    chunk_size=10,  # Process 10 at a time
)

# Combine features
share = SHAREModel.from_adapters(
    many_adapters,
    num_components=32,
    device="cuda",
    chunk_size=10,
)
```

### From HuggingFace Hub

```python
share = SHAREModel.from_adapters(
    ["org/cola_lora", "org/mrpc_lora", "org/rte_lora"],
    num_components="auto",
    variance_threshold=0.95,
)
```

## How It Works

For each layer and side (A/B) across all adapters:

1. **Stack** all adapters' LoRA weights into a matrix
2. **Center** and compute covariance
3. **Eigendecompose** via `torch.linalg.eigh` to find principal components
4. **Project** each adapter onto the top-k components to get compact loadings
5. **Reconstruct** on demand: `original ~= components @ loadings`

**Classifier Heads**: Task-specific output layers (e.g., `classifier.weight`, `classifier.bias`) are automatically separated during compression and stored alongside each adapter's loadings. During reconstruction, they're merged back in, so reconstructed adapters work immediately for inference without additional setup.

### Memory Savings

| | 6 LoRA adapters | SHARE (k=32) |
|---|---|---|
| Storage | 6 x full adapter | 1 shared basis + 6 tiny loadings |
| Params per layer | 6 x (r x d) | 1 x (k x d) + 6 x (k x r) |
| Example (d=768, r=8) | 73,728 | 52,224 (1.4x) |
| Example (d=768, r=8, N=100) | 1,228,800 | 74,752 (16x) |

Savings increase with more adapters. The paper reports 281x savings with 6 GLUE tasks.

## Save Format

```
checkpoint/
  share_config.json              # Compression metadata
  shared_components.safetensors  # Shared PCA basis vectors
  adapters/
    cola/
      loadings.safetensors         # Per-adapter projections (tiny)
      classifier_head.safetensors  # Task-specific classifier heads (if present)
      adapter_meta.json            # Original PEFT config
    mrpc/
      loadings.safetensors
      classifier_head.safetensors
      adapter_meta.json
```

**Note:** Task-specific classifier heads (like `classifier.weight`) are automatically preserved and merged back during reconstruction, ensuring reconstructed adapters can do inference immediately.

## Requirements

- Python >= 3.9
- PyTorch >= 1.13
- peft >= 0.6.0
- transformers >= 4.30.0
- safetensors >= 0.3.0

## Logging

Enable progress logging to see what's happening:

```python
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from lorashare import SHAREModel
share = SHAREModel.from_adapters(adapters, num_components=32)
# INFO: Loading 3 adapters...
# INFO: Validating adapter compatibility...
# INFO: Grouping weights by layer...
# INFO: Computing shared components (k=32)...
# INFO: Selected 32 components
# INFO: Computing per-adapter loadings...
```

Use `logging.DEBUG` for more detailed output (per-adapter progress).

## API Reference

`SHAREModel.from_adapters(adapters, num_components=32, variance_threshold=0.95)`

Compress multiple PEFT LoRA adapters. Accepts local paths or HuggingFace Hub IDs.

- `adapters`: `list[str]` or `dict[str, str]` (name -> path mapping)
- `num_components`: `int` or `"auto"` for explained-variance selection
- `variance_threshold`: target explained variance when using `"auto"` (default 0.95)

`SHAREModel.from_pretrained(path)`

Load a saved SHARE checkpoint.

`share.reconstruct(adapter_name, output_dir=None)`

Reconstruct a single adapter's LoRA weights. Optionally save as standard PEFT format.

`share.apply(base_model, adapter_name)`

Reconstruct and apply adapter to a base model. Returns a standard `peft.PeftModel`.

`share.save_pretrained(output_dir)`

Save SHARE checkpoint to disk.

 `share.summary()`

Print compression statistics.

 `share.reconstruction_error(adapter_name, original_weights=None, original_path=None)`

Compute per-layer reconstruction error (relative Frobenius norm).

## Citation

```bibtex
@article{kaushik2026share,
  title={Shared LoRA Subspaces for Almost Strict Continual Learning},
  author={Kaushik, Prakhar and Vaidya, Ankit and Chaudhari, Shravan and Chellappa, Rama and Yuille, Alan},
  journal={arXiv preprint arXiv:2602.06043},
  year={2026}
}
```

## License

MIT
