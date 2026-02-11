# lorashare Examples

This directory contains examples demonstrating how to use lorashare for compressing and managing multiple LoRA adapters.

## Quick Start

### 1. Basic Usage (`quickstart.py`)

Shows the complete workflow:
- Load multiple LoRA adapters
- Compress into shared subspace
- Save and load checkpoints
- Reconstruct adapters for inference

```bash
python examples/quickstart.py
```

### 2. Compression Comparison (`compare_compression.py`)

Demonstrates memory savings and quality:
- Creates synthetic adapters
- Compares original vs compressed sizes
- Measures reconstruction quality
- Compares inference outputs

```bash
python examples/compare_compression.py
```

**Example output:**
```
Original size: 73.73 MB
Compressed size: 24.58 MB
Compression ratio: 3.0x
Space saved: 66.7%
```

### 3. HuggingFace Hub Integration (`hub_usage.py`)

Working with the Hub:
- Load adapters from HuggingFace Hub
- Push compressed checkpoints to Hub
- Mix local and Hub adapters
- Reconstruct and re-upload

```bash
python examples/hub_usage.py
```

## Requirements

All examples require the base installation:
```bash
pip install lorashare
```

For real model inference examples, also install:
```bash
pip install transformers torch
```

## Creating Your Own Adapters

To use these examples with your own adapters:

1. **Train LoRA adapters** using PEFT:
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"])
model = get_peft_model(base_model, lora_config)

# Train your model...
model.save_pretrained("./my_adapter")
```

2. **Compress multiple adapters**:
```python
from lorashare import SHAREModel

share = SHAREModel.from_adapters(
    ["./adapter1", "./adapter2", "./adapter3"],
    num_components=32,
)
share.save_pretrained("./compressed")
```

3. **Use for inference**:
```python
share = SHAREModel.from_pretrained("./compressed")
model = share.apply(base_model, "adapter1")
# Run inference...
```

## Example Scenarios

### Scenario 1: Multi-Task Learning
```python
# Compress adapters for different GLUE tasks
tasks = ["cola", "mrpc", "rte", "sst2", "wnli"]
adapters = [f"./glue_adapters/{task}" for task in tasks]

share = SHAREModel.from_adapters(adapters, num_components="auto")
# Use 100x less memory than storing all adapters separately
```

### Scenario 2: Continual Learning
```python
# Compress adapters learned sequentially
time_periods = ["2023-q1", "2023-q2", "2023-q3", "2023-q4"]
adapters = [f"./temporal_adapters/{period}" for period in time_periods]

share = SHAREModel.from_adapters(adapters, variance_threshold=0.95)
# Maintain performance across all time periods
```

### Scenario 3: Deployment Optimization
```python
# Compress for edge deployment
share = SHAREModel.from_adapters(
    ["./prod_adapter_v1", "./prod_adapter_v2"],
    num_components=16,  # Aggressive compression for mobile
)

# Deploy compressed checkpoint (much smaller)
share.push_to_hub("company/prod-adapters-compressed")
```

## Tips

1. **Component Selection**:
   - Use `num_components="auto"` for quality-focused compression
   - Use specific `num_components=16` for size-focused compression
   - Higher k = better quality, larger size

2. **Memory Savings**:
   - Savings increase with more adapters
   - 3 adapters: ~1.5-2x compression
   - 10 adapters: ~5-8x compression
   - 100 adapters: ~50-100x compression

3. **Quality Control**:
   ```python
   # Check reconstruction error
   error = share.reconstruction_error("my_adapter", original_path="./my_adapter")
   if error['mean'] < 0.1:
       print("Good quality!")
   ```

4. **Inference Performance**:
   - Reconstructed adapters have same inference speed as originals
   - Only compression/decompression has overhead
   - Compress once, use many times

## Troubleshooting

**Error: "Need at least 2 adapters"**
- SHARE requires minimum 2 adapters to find shared subspace
- For single adapter, use standard PEFT

**Error: "Rank mismatch"**
- All adapters must have same LoRA rank (r)
- All adapters must target same modules
- All adapters must use same base model

**Poor reconstruction quality**
- Increase `num_components`
- Use `num_components="auto"` with higher `variance_threshold`
- Check adapter compatibility

## More Resources

- [Main README](../README.md) - Full documentation
- [SHARE Paper](https://arxiv.org/abs/2602.06043) - Original research
- [PEFT Documentation](https://huggingface.co/docs/peft) - Training LoRA adapters

## Contributing

Have a useful example? Submit a PR with:
- Clear documentation
- Error handling
- Example output
- Requirements listed
