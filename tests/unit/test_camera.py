"""
tests/unit/test_camera.py

Tests for tracker/capture/camera.py

Key technique: mocking cv2.VideoCapture so tests run without
a physical webcam — required for CI and clean unit testing.

What we test:
- Happy path: camera opens, yields frames, releases cleanly
- Error path: camera fails to open, frame read fails
- Context manager: __exit__ always releases, even on exception
- Properties: is_open, resolution behave correctly
- Guard: frames() outside context manager raises RuntimeError
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tracker.capture.camera import Camera, CameraError
from tracker.config import CameraConfig

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_mock_cap(
    is_opened: bool = True,
    frame: np.ndarray | None = None,
    read_success: bool = True,
) -> MagicMock:
    """
    Build a mock cv2.VideoCapture with controllable behavior.

    Args:
        is_opened: What isOpened() returns.
        frame: The frame returned by read(). Defaults to black 480x640 BGR.
        read_success: What the first return value of read() is.

    Returns:
        Configured MagicMock that mimics cv2.VideoCapture.
    """
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = is_opened
    mock_cap.read.return_value = (read_success, frame)

    # Mock property getters — called after set() to log actual resolution
    mock_cap.get.side_effect = lambda prop: {
        cv2_prop_width(): 640.0,
        cv2_prop_height(): 480.0,
        cv2_prop_fps(): 30.0,
    }.get(prop, 0.0)

    return mock_cap


def cv2_prop_width() -> int:
    import cv2

    return cv2.CAP_PROP_FRAME_WIDTH


def cv2_prop_height() -> int:
    import cv2

    return cv2.CAP_PROP_FRAME_HEIGHT


def cv2_prop_fps() -> int:
    import cv2

    return cv2.CAP_PROP_FPS


# ── Happy path ────────────────────────────────────────────────────────────────


def test_camera_opens_successfully() -> None:
    """Camera.__enter__ must succeed when VideoCapture.isOpened() is True."""
    mock_cap = make_mock_cap(is_opened=True)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with Camera(CameraConfig()) as cam:
            assert cam.is_open is True


def test_camera_yields_correct_frame_shape() -> None:
    """frames() must yield numpy arrays with the correct shape and dtype."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = make_mock_cap(frame=fake_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with Camera(CameraConfig()) as cam:
            frame = next(cam.frames())

    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_camera_yields_multiple_frames() -> None:
    """frames() must yield continuously — it is an infinite generator."""
    mock_cap = make_mock_cap()

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with Camera(CameraConfig()) as cam:
            frames = [next(cam.frames()) for _ in range(5)]

    assert len(frames) == 5


def test_camera_sets_requested_resolution() -> None:
    """Camera must call cap.set() with the configured width and height."""
    mock_cap = make_mock_cap()
    config = CameraConfig(width=1280, height=720, fps=30)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with Camera(config):
            pass

    import cv2

    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ── Resource management ───────────────────────────────────────────────────────


def test_camera_releases_on_clean_exit() -> None:
    """cap.release() must be called when the context manager exits normally."""
    mock_cap = make_mock_cap()

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with Camera(CameraConfig()):
            pass

    mock_cap.release.assert_called_once()


def test_camera_releases_on_exception() -> None:
    """
    cap.release() must be called even when an exception occurs inside
    the with block. This prevents webcam lockup after crashes.
    """
    mock_cap = make_mock_cap()

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(ValueError):
            with Camera(CameraConfig()):
                raise ValueError("simulated crash inside pipeline")

    # Camera must still be released despite the exception
    mock_cap.release.assert_called_once()


def test_camera_is_not_open_after_exit() -> None:
    """is_open must return False after the context manager exits."""
    mock_cap = make_mock_cap()

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with Camera(CameraConfig()) as cam:
            assert cam.is_open is True
        # Outside the with block now
        assert cam.is_open is False


# ── Error handling ────────────────────────────────────────────────────────────


def test_camera_raises_on_device_not_found() -> None:
    """CameraError must be raised when the device cannot be opened."""
    mock_cap = make_mock_cap(is_opened=False)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(CameraError) as exc_info:
            with Camera(CameraConfig(device_id=99)):
                pass

    assert "99" in str(exc_info.value)


def test_camera_raises_on_failed_frame_read() -> None:
    """CameraError must be raised when read() returns (False, ...)."""
    mock_cap = make_mock_cap(read_success=False)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(CameraError) as exc_info:
            with Camera(CameraConfig()) as cam:
                next(cam.frames())

    assert "Failed to read frame" in str(exc_info.value)


def test_frames_outside_context_raises_runtime_error() -> None:
    """
    Calling frames() before __enter__ must raise RuntimeError.
    This guards against misuse of the API.
    """
    cam = Camera(CameraConfig())
    with pytest.raises(RuntimeError) as exc_info:
        next(cam.frames())

    assert "with Camera" in str(exc_info.value)


# ── Properties ────────────────────────────────────────────────────────────────


def test_resolution_returns_actual_camera_values() -> None:
    """resolution property must return what the camera actually reports."""
    mock_cap = make_mock_cap()

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with Camera(CameraConfig()) as cam:
            w, h = cam.resolution

    assert w == 640
    assert h == 480


def test_resolution_outside_context_raises_runtime_error() -> None:
    """resolution must raise RuntimeError if camera is not open."""
    cam = Camera(CameraConfig())
    with pytest.raises(RuntimeError):
        _ = cam.resolution
