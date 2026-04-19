"""
Tests for tracker/config.py

What we test and why:
- Happy path: valid yaml loads correctly
- Validation: invalid values raise ValidationError with clear messages
- Defaults: missing sections fall back to sensible defaults
- Edge cases: file not found, empty file
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from tracker.config import (
    CameraConfig,
    Config,
    DetectorConfig,
    LoggingConfig,
    TrackerConfig,
    VisualizationConfig,
    load_config,
)


# ------ Fixtures ------


@pytest.fixture
def default_config(tmp_path: Path) -> Path:
    """Write a minimal valid config yaml to a temp file and return the path."""
    content = textwrap.dedent("""
        camera:
          device_id: 0
          width: 1280
          height: 720
          fps: 30
        detector:
          model: "yolov8n.pt"
          confidence: 0.5
          iou_threshold: 0.45
          device: "cuda"
          detect_every: 1
        tracker:
          max_age: 30
          min_hits: 3
          iou_threshold: 0.3
        visualization:
          show_confidence: true
          show_track_id: true
          show_class_name: true
          box_thickness: 2
          font_scale: 0.6
          heatmap_alpha: 0.4
          heatmap_decay: 0.99
        logging:
          level: "INFO"
          file: "logs/tracker.log"
          rotation: "10 MB"
          retention: "7 days"
    """)
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(content, encoding="utf-8")
    return config_file


# ------ Happy path ------


def test_load_config_returns_config_object(default_config: Path) -> None:
    """load_config() must return a Config instance, not a dict."""
    cfg = load_config(default_config)
    assert isinstance(cfg, Config)


def test_load_config_camera_values(default_config: Path) -> None:
    """Camera section values must match what was written in yaml."""
    cfg = load_config(default_config)
    assert cfg.camera.device_id == 0
    assert cfg.camera.width == 1280
    assert cfg.camera.height == 720
    assert cfg.camera.fps == 30


def test_load_config_detector_values(default_config: Path) -> None:
    """Detector section values must match yaml."""
    cfg = load_config(default_config)
    assert cfg.detector.confidence == 0.5
    assert cfg.detector.iou_threshold == 0.45
    assert cfg.detector.device == "cuda"


def test_load_config_tracker_values(default_config: Path) -> None:
    cfg = load_config(default_config)
    assert cfg.tracker.max_age == 30
    assert cfg.tracker.min_hits == 3


# ------ Validation errors ------


def test_confidence_above_1_raises(tmp_path: Path) -> None:
    """confidence=1.7 must fail — this is the core value of Pydantic validation."""
    with pytest.raises(ValidationError) as exc_info:
        DetectorConfig(confidence=1.7)
    assert "less_than_equal" in str(exc_info.value)


def test_confidence_below_0_raises() -> None:
    with pytest.raises(ValidationError):
        DetectorConfig(confidence=-0.1)


def test_iou_threshold_above_1_raises() -> None:
    with pytest.raises(ValidationError):
        DetectorConfig(iou_threshold=1.1)


def test_camera_negative_device_id_raises() -> None:
    with pytest.raises(ValidationError):
        CameraConfig(device_id=-1)


def test_camera_zero_width_raises() -> None:
    """Width must be > 0 — zero-pixel camera makes no sense."""
    with pytest.raises(ValidationError):
        CameraConfig(width=0)


def test_tracker_max_age_zero_raises() -> None:
    """max_age=0 would delete tracks immediately — nonsensical."""
    with pytest.raises(ValidationError):
        TrackerConfig(max_age=0)


# ------ Defaults ------


def test_empty_config_uses_defaults(tmp_path: Path) -> None:
    """An empty yaml file must produce a Config with all default values."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}", encoding="utf-8")
    cfg = load_config(empty)
    assert cfg.camera.device_id == 0
    assert cfg.detector.confidence == 0.5
    assert cfg.detector.device == "cuda"


def test_partial_config_uses_defaults(tmp_path: Path) -> None:
    """Only overriding camera should leave detector at defaults."""
    partial = tmp_path / "partial.yaml"
    partial.write_text("camera:\n  device_id: 1\n", encoding="utf-8")
    cfg = load_config(partial)
    assert cfg.camera.device_id == 1
    assert cfg.detector.confidence == 0.5  # default untouched


# ------ Error handling ------


def test_missing_file_raises_file_not_found() -> None:
    """A clear FileNotFoundError is better than a cryptic yaml crash."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_config("configs/nonexistent.yaml")
    assert "nonexistent.yaml" in str(exc_info.value)