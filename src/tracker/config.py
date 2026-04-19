"""
tracker/config.py

Central configuration module. All runtime parameters live in
configs/default.yaml and are validated here at startup.

Design decisions:
- Pydantic v2 BaseModel: validates types AND value ranges at load time
- Each section is its own class: easy to pass CameraConfig to Camera,
  DetectorConfig to YOLODetector - no one gets the whole Config object
- load_config() is the single entry point - call it once at startup
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# ------ Section models ------
# Each class maps to one top-level key in default.yaml.
# Field(description=...) documents what the value means.
# Field(ge=0, le=1) enforces 0.0-1.0 range - ge = greater-or-equal, le = less-or-equal.


class CameraConfig(BaseModel):
    device_id: int = Field(default=0, ge=0, description="Webcam index")
    width: int = Field(default=1280, gt=0, description="Capture width in pixels")
    height: int = Field(default=720, gt=0, description="Capture height in pixels")
    fps: int = Field(default=30, gt=0, description="Target frames per second")


class DetectorConfig(BaseModel):
    model: str = Field(default="yolov8n.pt", description="Model filename in models/")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Min detection confidence"
    )
    iou_threshold: float = Field(
        default=0.45, ge=0.0, le=1.0, description="NMS IoU threshold"
    )
    device: str = Field(default="cuda", description="'cuda' or 'cpu'")
    detect_every: int = Field(
        default=1, ge=1, description="Run detector every N frames"
    )


class TrackerConfig(BaseModel):
    max_age: int = Field(
        default=30, ge=1, description="Frames to keep a track alive without detection"
    )
    min_hits: int = Field(
        default=3, ge=1, description="Detections needed before track is reported"
    )
    iou_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Min IoU to match detection to track"
    )


class VisualizationConfig(BaseModel):
    show_confidence: bool = True
    show_track_id: bool = True
    show_class_name: bool = True
    box_thickness: int = Field(default=2, ge=1, le=10)
    font_scale: float = Field(default=0.6, ge=0.1, le=3.0)
    heatmap_alpha: float = Field(default=0.4, ge=0.0, le=1.0)
    heatmap_decay: float = Field(default=0.99, ge=0.0, le=1.0)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO", description="DEBUG/INFO/WARNING/ERROR")
    file: str = Field(default="logs/tracker.log")
    rotation: str = Field(default="10 MB")
    retention: str = Field(default="7 days")


# ------ Root config ------


class Config(BaseModel):
    """Root configuration object. Holds all section configs as attributes."""

    camera: CameraConfig = CameraConfig()
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    logging: LoggingConfig = LoggingConfig()


# ------ Loader ------


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """
    Load and validate configuration from a yaml file.

    Args:
        path: Path to the yaml config file.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If any value fails validation.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            f"Expected location: configs/default.yaml"
        )

    # encoding='utf-8' is explicit — never rely on Windows system default (GBK)
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Pydantic validates every field and raises ValidationError if anything is wrong
    return Config(**raw)
