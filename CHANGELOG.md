# Changelog

All notable changes to lorashare will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI/CD testing workflow with automated tests on PRs
- Coverage reporting via Codecov
- Comprehensive examples directory with quickstart, comparison, and Hub usage
- Real model inference tests with roberta-base

## [0.1.0] - 2025-02-11

### Added
- Initial release of lorashare
- Core compression algorithm using PCA-based shared subspace extraction
- Support for compressing multiple PEFT LoRA adapters
- `SHAREModel` class with full API:
  - `from_adapters()` - Compress multiple adapters
  - `from_pretrained()` - Load compressed checkpoint
  - `save_pretrained()` - Save compressed checkpoint
  - `reconstruct()` - Reconstruct individual adapter
  - `apply()` - Apply reconstructed adapter to base model
  - `summary()` - Display compression statistics
  - `push_to_hub()` - Upload to HuggingFace Hub
  - `reconstruction_error()` - Measure reconstruction quality
- CLI tool with three commands:
  - `lorashare compress` - Compress adapters
  - `lorashare info` - Display checkpoint info
  - `lorashare reconstruct` - Reconstruct adapters
- Automatic classifier head preservation
  - Task-specific heads (classifier.weight, etc.) stored separately
  - Merged back during reconstruction for immediate inference
- Component selection modes:
  - Fixed: Specify exact number of components
  - Auto: Select based on explained variance threshold
- HuggingFace Hub integration:
  - Load adapters from Hub IDs
  - Push compressed checkpoints to Hub
  - Mix local and Hub adapters
- Comprehensive test suite:
  - 28 compression algorithm tests
  - 16 I/O operation tests
  - 13 model API tests
  - 7 synthetic inference tests
  - 3 real model inference tests (67 total tests)
- SHARE checkpoint format:
  - `share_config.json` - Compression metadata
  - `shared_components.safetensors` - Shared PCA basis
  - `adapters/{name}/loadings.safetensors` - Per-adapter projections
  - `adapters/{name}/classifier_head.safetensors` - Task heads
  - `adapters/{name}/adapter_meta.json` - Original config
- Logging support:
  - INFO level for compression progress
  - DEBUG level for detailed per-adapter progress
- Full type hints and docstrings
- MIT License

### Features
- **Post-hoc compression**: Works with any pre-trained PEFT LoRA adapters
- **100x+ memory savings**: With 100 adapters (paper reports 281x with 6 tasks)
- **Zero retraining**: No custom training required
- **PEFT compatible**: Reconstructed adapters work as standard PEFT format
- **Quality preservation**: Configurable variance threshold (default 95%)
- **Platform support**: Linux, macOS, Windows
- **Python support**: Python 3.9, 3.10, 3.11+

### Requirements
- Python >= 3.9
- torch >= 1.13
- safetensors >= 0.3.0
- huggingface-hub >= 0.19.0
- peft >= 0.6.0
- transformers >= 4.30.0

### Documentation
- README with quickstart, API examples, and algorithm explanation
- Comprehensive examples directory
- Inline docstrings for all public APIs
- Type hints throughout codebase

### Known Limitations
- Requires minimum 2 adapters for compression
- All adapters must have:
  - Same LoRA rank (r)
  - Same target modules
  - Same base model
- Loading SHARE checkpoints from HuggingFace Hub not yet implemented (can push but not load)
- No per-layer component selection (uniform k across all layers)

### Performance
- Compression time: ~1-2 seconds for 3 adapters (12 layers, r=8)
- Reconstruction time: <100ms per adapter
- Memory overhead during compression: ~2x adapter size
- Inference speed: Same as original adapters (no runtime overhead)

### Breaking Changes
- N/A (initial release)

## Migration Guides

### From Training-Time SHARE (paper)

The paper's implementation requires training with EigenFlux from scratch. This library works differently:

**Paper approach (training-time):**
```python
# Requires custom training loop
model = EigenFluxModel(...)
train_with_eigenflux(model)  # Custom training
```

**lorashare approach (post-hoc):**
```python
# 1. Train normally with standard PEFT
from peft import LoraConfig, get_peft_model
model = get_peft_model(base_model, LoraConfig(r=8))
# ... train normally ...
model.save_pretrained("./adapter")

# 2. Compress after training
from lorashare import SHAREModel
share = SHAREModel.from_adapters(["./adapter1", "./adapter2"])
```

**Benefits of post-hoc:**
- Use any existing LoRA adapters
- No custom training code
- Compatible with all PEFT training workflows
- Can compress adapters trained months apart

**Trade-off:**
- Paper's training-time approach may achieve slightly better compression
- lorashare prioritizes ease of use and compatibility

## [0.2.0] - Planned

### Planned Features
- Load SHARE checkpoints from HuggingFace Hub
- Per-layer component selection (adaptive k)
- Incremental compression (add adapters to existing checkpoint)
- Progress bars (tqdm) for long operations
- Quality metrics dashboard
- Benchmark suite

### Planned Improvements
- Better error messages with suggestions
- Automatic k selection validation
- Memory-efficient compression for large adapters
- Partial loading (load only specific layers)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Acknowledgments

Based on the SHARE paper:
```bibtex
@article{kaushik2026share,
  title={Shared LoRA Subspaces for Almost Strict Continual Learning},
  author={Kaushik, Prakhar and Vaidya, Ankit and Chaudhari, Shravan and Chellappa, Rama and Yuille, Alan},
  journal={arXiv preprint arXiv:2602.06043},
  year={2026}
}
```
