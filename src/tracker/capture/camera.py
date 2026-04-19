"""
tracker/capture/camera.py

Camera abstraction layer — the ONLY place in the codebase that
calls cv2.VideoCapture().

Design decisions:
- Context manager (__enter__/__exit__): guarantees camera release
  even when downstream code throws an exception
- Generator (frames()): lazy evaluation — one frame at a time,
  caller controls the loop, no frames held in memory
- CameraError: typed exception so callers can catch camera-specific
  failures separately from other errors
- No cv2 calls outside this file — abstraction boundary is strict

Usage:
    with Camera(config.camera) as cam:
        for frame in cam.frames():
            detections = detector.detect(frame)
"""

from __future__ import annotations

from collections.abc import Generator

import cv2
import numpy as np

from tracker.config import CameraConfig
from tracker.logger import get_logger

logger = get_logger(__name__)


class CameraError(Exception):
    """
    Raised when the camera cannot be opened or a frame cannot be read.

    Separate from generic Exception so callers can handle camera
    failures specifically — e.g. retry logic or fallback to video file.
    """


class Camera:
    """
    Webcam abstraction with context manager and frame generator.

    Args:
        config: CameraConfig section from the main Config object.

    Example:
        cfg = load_config('configs/default.yaml')
        with Camera(cfg.camera) as cam:
            for frame in cam.frames():
                # frame is a numpy array: shape (H, W, 3), dtype uint8, BGR
                process(frame)
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> Camera:
        """Open the camera device and configure resolution."""
        logger.info(
            f"Opening camera device {self._config.device_id} "
            f"({self._config.width}x{self._config.height} @ {self._config.fps}fps)"
        )

        self._cap = cv2.VideoCapture(self._config.device_id)

        if not self._cap.isOpened():
            raise CameraError(
                f"Cannot open camera device {self._config.device_id}. "
                f"Check that the webcam is connected and not in use by another app."
            )

        # Request resolution — note: camera may not honor these exactly
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        # Log what the camera actually gave us (may differ from requested)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            f"Camera opened — actual resolution: {actual_w}x{actual_h} @ {actual_fps:.1f}fps"
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the camera — always called, even if an exception occurred."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

        # Return False — do not suppress exceptions
        # If the caller's code threw an error, let it propagate
        return False

    # ── Frame generator ───────────────────────────────────────────────────────

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Yield frames from the camera indefinitely.

        Yields:
            numpy array of shape (H, W, 3), dtype uint8, BGR color order.
            OpenCV always returns BGR — convert to RGB if your model needs it.

        Raises:
            CameraError: If the camera is not open or a frame read fails.
            RuntimeError: If called outside of the context manager.
        """
        if self._cap is None:
            raise RuntimeError(
                "Camera is not open. Use 'with Camera(config) as cam:' "
                "before calling cam.frames()"
            )

        frame_count = 0
        while True:
            ret, frame = self._cap.read()

            if not ret:
                raise CameraError(
                    f"Failed to read frame {frame_count} from device "
                    f"{self._config.device_id}. "
                    f"Camera may have been disconnected."
                )

            frame_count += 1
            yield frame

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        """True if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Actual camera resolution as (width, height).

        Raises:
            RuntimeError: If called before opening the camera.
        """
        if self._cap is None:
            raise RuntimeError("Camera is not open")
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
