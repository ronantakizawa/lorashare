#!/bin/bash
set -e

echo "=== Setting up RunPod environment for lorashare GPU tests ==="

# Install the library in editable mode
pip install -e /workspace/peft-share

# Install test dependencies
pip install datasets evaluate accelerate scikit-learn
pip install pytest pytest-cov

# Verify GPU is available
python -c "import torch; assert torch.cuda.is_available(), 'No GPU found!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify key packages
python -c "import lorashare; print(f'lorashare version: {lorashare.__version__}')"
python -c "import peft; print(f'peft version: {peft.__version__}')"
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"

echo "=== Setup complete ==="
