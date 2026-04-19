# Environment Notes

## Python: 3.14.4
## Conda env name: realtime-tracker

## PyTorch setup
- torch\==2.11.0+cu130, torchvision\==0.26.0+cu130
- CUDA 13.0, GPU: NVIDIA RTX 4070 Laptop
- Install command:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

## Constraints
- setuptools MUST stay at 70.2.0
- Reason: torch 2.11 requires setuptools<82, newer versions break wheel metadata
- Do NOT run: pip install --upgrade setuptools

## Reproducing this environment
1. conda create -n realtime-tracker python=3.14.4
2. conda activate realtime-tracker
3. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
4. pip install -e ".[dev,api]"

## Windows encoding
- System locale: Chinese (GBK/CP936)
- Always use: open(path, encoding='utf-8') for all file I/O
- Always save source files as UTF-8 without BOM
- Avoid Unicode box-drawing characters in source files and configs