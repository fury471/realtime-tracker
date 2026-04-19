"""
tests/unit/test_color.py

Tests for tracker/processing/color.py

Testing strategy:
- Use synthetic frames with known pixel values so expected outputs
  are mathematically predictable — no need for real images
- Test the silent bug explicitly: BGR/RGB confusion must be detectable
- Test all validation paths: wrong dtype, wrong dims, wrong channels
- Test roundtrips: BGR → RGB → BGR must equal original
"""

from __future__ import annotations

import numpy as np
import pytest

from tracker.processing.color import ColorSpaceConverter, ColorSpaceError

# ── Fixtures ──────────────────────────────────────────────────────────────────


def bgr_red() -> np.ndarray:
    """100x100 frame of pure red in BGR: [0, 0, 255]."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [0, 0, 255]
    return frame


def bgr_green() -> np.ndarray:
    """100x100 frame of pure green in BGR: [0, 255, 0]."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [0, 255, 0]
    return frame


def bgr_black() -> np.ndarray:
    """100x100 frame of pure black: [0, 0, 0]."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def bgr_white() -> np.ndarray:
    """100x100 frame of pure white: [255, 255, 255]."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)


# ── bgr_to_rgb ────────────────────────────────────────────────────────────────


def test_bgr_to_rgb_reverses_channels() -> None:
    """BGR [0,0,255] must become RGB [255,0,0]."""
    rgb = ColorSpaceConverter.bgr_to_rgb(bgr_red())
    np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])


def test_bgr_to_rgb_preserves_shape() -> None:
    """Shape must be unchanged after BGR→RGB conversion."""
    frame = bgr_red()
    rgb = ColorSpaceConverter.bgr_to_rgb(frame)
    assert rgb.shape == frame.shape


def test_bgr_to_rgb_preserves_dtype() -> None:
    """dtype must remain uint8 after conversion."""
    rgb = ColorSpaceConverter.bgr_to_rgb(bgr_red())
    assert rgb.dtype == np.uint8


def test_bgr_to_rgb_roundtrip() -> None:
    """
    BGR → RGB → BGR must equal original.
    This is a mathematical identity — any deviation is a bug.
    """
    import cv2

    frame = bgr_red()
    rgb = ColorSpaceConverter.bgr_to_rgb(frame)
    back = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    np.testing.assert_array_equal(frame, back)


def test_bgr_to_rgb_does_not_mutate_input() -> None:
    """Conversion must return a new array, never modify the input."""
    frame = bgr_red()
    original = frame.copy()
    ColorSpaceConverter.bgr_to_rgb(frame)
    np.testing.assert_array_equal(frame, original)


# ── bgr_to_gray ───────────────────────────────────────────────────────────────


def test_bgr_to_gray_returns_2d_array() -> None:
    """Grayscale output must be 2D (H, W), not 3D."""
    gray = ColorSpaceConverter.bgr_to_gray(bgr_red())
    assert gray.ndim == 2
    assert gray.shape == (100, 100)


def test_bgr_to_gray_red_luminance() -> None:
    """
    Pure red BGR [0,0,255] grayscale = 76.
    Formula: Y = 0.299×R + 0.587×G + 0.114×B = 0.299×255 = 76.245 → 76
    """
    gray = ColorSpaceConverter.bgr_to_gray(bgr_red())
    assert gray[0, 0] == 76


def test_bgr_to_gray_black_is_zero() -> None:
    """Black frame must produce all-zero grayscale."""
    gray = ColorSpaceConverter.bgr_to_gray(bgr_black())
    assert gray.max() == 0


def test_bgr_to_gray_white_is_255() -> None:
    """White frame must produce all-255 grayscale."""
    gray = ColorSpaceConverter.bgr_to_gray(bgr_white())
    assert gray.min() == 255


# ── bgr_to_hsv ────────────────────────────────────────────────────────────────


def test_bgr_to_hsv_red_hue_is_zero() -> None:
    """
    Pure red HSV hue = 0.
    Red sits at 0° on the color wheel.
    OpenCV H range is 0-179 (divided by 2 to fit uint8).
    """
    hsv = ColorSpaceConverter.bgr_to_hsv(bgr_red())
    assert hsv[0, 0, 0] == 0  # H = 0 for red


def test_bgr_to_hsv_red_saturation_is_max() -> None:
    """Pure red must have maximum saturation (255)."""
    hsv = ColorSpaceConverter.bgr_to_hsv(bgr_red())
    assert hsv[0, 0, 1] == 255  # S = 255


def test_bgr_to_hsv_red_value_is_max() -> None:
    """Pure red must have maximum value/brightness (255)."""
    hsv = ColorSpaceConverter.bgr_to_hsv(bgr_red())
    assert hsv[0, 0, 2] == 255  # V = 255


def test_bgr_to_hsv_black_value_is_zero() -> None:
    """Black must have V=0 — zero brightness regardless of hue."""
    hsv = ColorSpaceConverter.bgr_to_hsv(bgr_black())
    assert hsv[0, 0, 2] == 0


def test_bgr_rgb_confusion_is_detectable() -> None:
    """
    This test explicitly documents the silent bug.
    Feeding RGB data to a BGR→HSV function gives wrong hue.
    BGR red [0,0,255] fed as if it were BGR gives H=0 (correct).
    RGB red [255,0,0] fed to BGR→HSV gives H=120 (wrong — that's green!).
    """
    import cv2

    bgr_input = bgr_red()  # [0, 0, 255] — correct BGR red
    rgb_input = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_input[:, :] = [255, 0, 0]  # [255, 0, 0] — RGB red, wrong for BGR fn

    correct_hsv = ColorSpaceConverter.bgr_to_hsv(bgr_input)
    wrong_hsv = cv2.cvtColor(rgb_input, cv2.COLOR_BGR2HSV)

    assert correct_hsv[0, 0, 0] == 0  # H=0, correct red
    assert wrong_hsv[0, 0, 0] == 120  # H=120, wrong — interpreted as green!
    assert correct_hsv[0, 0, 0] != wrong_hsv[0, 0, 0]


# ── gray_to_bgr ───────────────────────────────────────────────────────────────


def test_gray_to_bgr_returns_3_channels() -> None:
    """gray_to_bgr must return a 3-channel array."""
    gray = ColorSpaceConverter.bgr_to_gray(bgr_red())
    bgr = ColorSpaceConverter.gray_to_bgr(gray)
    assert bgr.ndim == 3
    assert bgr.shape[2] == 3


def test_gray_to_bgr_all_channels_equal() -> None:
    """
    Grayscale→BGR: all 3 channels must have identical values.
    There is no color information to recover — channels are duplicated.
    """
    gray = ColorSpaceConverter.bgr_to_gray(bgr_red())
    bgr = ColorSpaceConverter.gray_to_bgr(gray)
    np.testing.assert_array_equal(bgr[:, :, 0], bgr[:, :, 1])
    np.testing.assert_array_equal(bgr[:, :, 1], bgr[:, :, 2])


# ── Histogram ─────────────────────────────────────────────────────────────────


def test_histogram_returns_three_channels() -> None:
    """compute_histogram must return dict with blue/green/red keys."""
    hist = ColorSpaceConverter.compute_histogram(bgr_red())
    assert set(hist.keys()) == {"blue", "green", "red"}


def test_histogram_red_frame_peaks_in_red_channel() -> None:
    """
    Pure red frame: all pixels have R=255, G=0, B=0.
    Red channel histogram bin 255 must be 100×100=10000.
    Blue and green channel bin 255 must be 0.
    """
    hist = ColorSpaceConverter.compute_histogram(bgr_red())
    assert hist["red"][255] == 10000  # all 10000 pixels at max red
    assert hist["blue"][255] == 0  # no pixels at max blue
    assert hist["green"][255] == 0  # no pixels at max green


def test_histogram_bin_count_matches_request() -> None:
    """Histogram length must match requested bin count."""
    hist = ColorSpaceConverter.compute_histogram(bgr_red(), bins=128)
    assert len(hist["red"]) == 128


# ── Validation ────────────────────────────────────────────────────────────────


def test_wrong_dtype_raises() -> None:
    """float32 input must raise ColorSpaceError — not a silent wrong result."""
    frame = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(ColorSpaceError) as exc_info:
        ColorSpaceConverter.bgr_to_rgb(frame)
    assert "uint8" in str(exc_info.value)


def test_wrong_dimensions_raises() -> None:
    """2D input to bgr_to_rgb must raise ColorSpaceError."""
    frame = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ColorSpaceError) as exc_info:
        ColorSpaceConverter.bgr_to_rgb(frame)
    assert "3D" in str(exc_info.value)


def test_wrong_channel_count_raises() -> None:
    """4-channel BGRA input must raise ColorSpaceError."""
    frame = np.zeros((100, 100, 4), dtype=np.uint8)
    with pytest.raises(ColorSpaceError) as exc_info:
        ColorSpaceConverter.bgr_to_rgb(frame)
    assert "3 channels" in str(exc_info.value)


def test_non_array_input_raises() -> None:
    """Passing a Python list instead of numpy array must raise ColorSpaceError."""
    with pytest.raises(ColorSpaceError) as exc_info:
        ColorSpaceConverter.bgr_to_rgb([[[0, 0, 255]]])
    assert "numpy array" in str(exc_info.value)
