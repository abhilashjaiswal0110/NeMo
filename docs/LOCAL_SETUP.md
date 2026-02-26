# Local Setup Guide

Complete guide for setting up NVIDIA NeMo Framework locally for development and experimentation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
  - [Option 1: Pip / Conda (Recommended for ASR & TTS)](#option-1-pip--conda-recommended-for-asr--tts)
  - [Option 2: From Source](#option-2-from-source)
  - [Option 3: Docker Container](#option-3-docker-container)
- [Verify Installation](#verify-installation)
- [Environment Configuration](#environment-configuration)
- [IDE Setup](#ide-setup)
- [Running Tests Locally](#running-tests-locally)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.10 – 3.12 | 3.10.12 recommended for conda |
| **PyTorch** | 2.6+ | Install before NeMo |
| **CUDA** | 12.x | Required for GPU training |
| **NVIDIA GPU** | Volta+ (sm_70+) | Required for training; inference works on CPU |
| **Git** | Latest | For source installs |
| **conda / mamba** | Latest | Recommended for environment isolation |

### Check System Requirements

```bash
# Check Python version
python --version

# Check NVIDIA GPU availability
nvidia-smi

# Check CUDA version
nvcc --version
```

---

## Installation Methods

### Option 1: Pip / Conda (Recommended for ASR & TTS)

Best for experimenting with pre-trained models and quick development.

```bash
# 1. Create isolated conda environment
conda create --name nemo python=3.10.12
conda activate nemo

# 2. Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install NeMo (choose extras based on your use case)
pip install "nemo_toolkit[all]"        # All collections
pip install "nemo_toolkit[asr]"        # Speech Recognition only
pip install "nemo_toolkit[tts]"        # Text-to-Speech only
pip install "nemo_toolkit[audio]"      # Audio processing only
```

### Option 2: From Source

Best for contributing to NeMo or using the latest development features.

```bash
# 1. Clone this repository
git clone https://github.com/abhilashjaiswal0110/NeMo.git
cd NeMo

# 2. Create and activate environment
conda create --name nemo-dev python=3.10.12
conda activate nemo-dev

# 3. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install NeMo in editable mode with desired extras
pip install -e '.[all]'         # All collections
pip install -e '.[asr]'         # ASR only
pip install -e '.[tts]'         # TTS only
pip install -e '.[asr,tts]'     # ASR + TTS

# 5. Install development dependencies
pip install -r requirements/requirements_dev.txt
pip install pre-commit
pre-commit install
```

### Option 3: Docker Container

Best for production-like environments and maximum GPU performance.

```bash
# Option A: Pre-built NeMo container (recommended)
docker pull nvcr.io/nvidia/nemo:25.11.01
docker run \
  --gpus all \
  -it \
  --rm \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/nemo:25.11.01

# Option B: NGC PyTorch base container + source install
docker run \
  --gpus all \
  -it \
  --rm \
  --shm-size=16g \
  nvcr.io/nvidia/pytorch:25.09-py3 \
  bash

# Inside container:
git clone https://github.com/abhilashjaiswal0110/NeMo.git /opt/NeMo
cd /opt/NeMo
pip install -e '.[all]'
```

---

## Verify Installation

```python
# Quick sanity check
import nemo
print(f"NeMo version: {nemo.__version__}")

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Test ASR import
from nemo.collections.asr.models import EncDecCTCModel
print("ASR import: OK")

# Test TTS import
from nemo.collections.tts.models import FastPitchModel
print("TTS import: OK")
```

---

## Environment Configuration

### Environment Variables

Create a `.env` file at the project root (already in `.gitignore`):

```bash
# .env (do NOT commit this file)

# NeMo cache directory for downloaded models
NEMO_CACHE_DIR=~/.cache/nemo

# NGC credentials (for accessing private NGC models)
NGC_API_KEY=your_ngc_api_key_here

# HuggingFace token (for gated models)
HF_TOKEN=your_hf_token_here

# Weights & Biases (optional, for experiment tracking)
WANDB_API_KEY=your_wandb_key_here

# Disable tokenizer parallelism warnings
TOKENIZERS_PARALLELISM=false

# PyTorch: allow loading trusted checkpoints
# TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # Uncomment only when needed for trusted files
```

### Model Cache

NeMo downloads pre-trained models to a local cache. Configure the cache location:

```bash
# Default location: ~/.cache/torch/NeMo
export NEMO_CACHE_DIR=/path/to/your/cache

# Or programmatically
import nemo.utils
nemo.utils.nemo_logging.Logger.set_verbosity(nemo.utils.logging.DEBUG)
```

---

## IDE Setup

### VS Code

Install recommended extensions (`.vscode/extensions.json` if present):

```bash
# Python
ext install ms-python.python
ext install ms-python.vscode-pylance

# Jupyter (for notebooks)
ext install ms-toolsai.jupyter
```

Configure Python interpreter to use your conda environment:
1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose your `nemo` or `nemo-dev` conda environment

### PyCharm

1. **File → Settings → Project → Python Interpreter**
2. Add Interpreter → Conda Environment → Select `nemo` environment

---

## Running Tests Locally

```bash
# Run a quick unit test (CPU, no GPU required)
pytest tests/unit/ -m "not pleasefixme" --cpu -x -v

# Run ASR unit tests
pytest tests/collections/asr/ -m "not pleasefixme" --cpu -x

# Run TTS unit tests
pytest tests/collections/tts/ -m "not pleasefixme" --cpu -x

# Run with GPU
pytest tests/collections/asr/ -m "not pleasefixme" -x

# Run with pre-trained model downloads (takes longer)
pytest -m "not pleasefixme" --with_downloads tests/collections/asr/
```

---

## Troubleshooting

### Common Issues

#### `ImportError: No module named 'nemo'`
```bash
# Ensure environment is activated
conda activate nemo

# Verify install
pip show nemo-toolkit
```

#### CUDA out of memory
```python
# Reduce batch size in your config
trainer.batch_size = 4  # reduce from default

# Or enable gradient checkpointing
model.config.use_gradient_checkpointing = True
```

#### `torch.load` security warning with checkpoints
```bash
# For trusted checkpoints only:
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

#### Pre-commit hooks failing
```bash
# Auto-fix formatting issues
pre-commit run --all-files

# If flake8/pylint errors persist, check specific files
flake8 path/to/file.py
pylint path/to/file.py
```

#### Slow model downloads
```bash
# Set a persistent cache directory to avoid re-downloading
export NEMO_CACHE_DIR=/path/with/enough/space
```

---

## Next Steps

- 📖 Read the [Use Cases Guide](USECASES.md) for practical examples
- 🎯 Explore the [Prompts Guide](PROMPTS.md) for NeMo model prompting
- 🤖 Use the [Agents](../agents/README.md) for automated NeMo workflows
- 🧪 Check [tutorials/](../tutorials/) for Jupyter notebook examples
