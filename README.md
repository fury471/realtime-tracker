# Realtime Object Tracker

[![CI](https://github.com/fury471/realtime-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/fury471/realtime-tracker/actions/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Real-time object tracking pipeline using YOLOv8 + SORT algorithm, served via FastAPI with a live WebSocket stream.

## Project structure

```bash
realtime-tracker/
├── src/tracker/
│   ├── capture/        ← camera abstraction
│   ├── detection/      ← YOLO detector
│   ├── tracking/       ← Kalman filter + SORT
│   ├── processing/     ← image filters and edges
│   ├── visualization/  ← renderer and heatmap
│   ├── config.py       ← Pydantic config validation
│   └── logger.py       ← structured logging
├── api/                ← FastAPI endpoints
├── tests/              ← unit and integration tests
├── configs/            ← yaml configuration files
└── docker/             ← Dockerfile and compose
```

## Quick start

```bash
# 1. Clone
git clone git@github.com:YOUR_USERNAME/realtime-tracker.git
cd realtime-tracker

# 2. Create environment
conda create -n realtime-tracker python=3.11
conda activate realtime-tracker

# 3. Install torch with CUDA (see NOTES.md for details)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 4. Install project
pip install -e ".[dev,api]"

# 5. Run tests
python -m pytest tests/
```

## Development

```bash
python -m pytest tests/        # run all tests
python -m black src/ tests/    # format code
python -m ruff check src/      # lint
```

## Architecture

```bash
Camera → YOLODetector → SORTTracker → Renderer → FastAPI → Browser
```

Each component is independently tested and replaceable.

The pipeline never calls cv2 directly — only Camera does.
