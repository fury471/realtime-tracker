"""
tracker/processing/color.py

Color space conversion utilities.

Design decisions:
- Static methods only: no state, no instantiation needed
  ColorSpaceConverter.bgr_to_rgb(frame) reads clearly at call sites
- All methods validate input dtype and channels before converting
  Silent wrong results are worse than loud early errors
- BGR is the canonical internal format (OpenCV default)
  All methods document their expected input format explicitly
- No cv2 calls outside this file in the processing module

Color space reference:
    BGR  — OpenCV default. Blue-Green-Red channel order.
           Shape: (H, W, 3), dtype: uint8
    RGB  — Standard for neural networks and display.
           Same shape, reversed channel order.
    HSV  — Hue-Saturation-Value. Good for color-based filtering.
           H: 0-179 (OpenCV), S: 0-255, V: 0-255
    LAB  — Perceptually uniform. L: lightness, A/B: color axes.
           Good for illumination-invariant processing.
    GRAY — Single channel luminance.
           Shape: (H, W), dtype: uint8
"""

from __future__ import annotations

import cv2
import numpy as np


class ColorSpaceError(Exception):
    """Raised when an invalid frame is passed to a color conversion."""


class ColorSpaceConverter:
    """
    Static utility class for color space conversions.

    All methods:
    - Accept BGR input (OpenCV default from Camera.frames())
    - Validate shape and dtype before converting
    - Return a new array (never mutate the input)

    Example:
        frame = next(camera.frames())          # BGR, uint8
        rgb   = ColorSpaceConverter.bgr_to_rgb(frame)   # for YOLO
        gray  = ColorSpaceConverter.bgr_to_gray(frame)  # for edge detection
        hsv   = ColorSpaceConverter.bgr_to_hsv(frame)   # for color filtering
    """

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_bgr(frame: np.ndarray) -> None:
        """
        Validate that a frame is a proper BGR image.

        Args:
            frame: Array to validate.

        Raises:
            ColorSpaceError: If frame is not a valid BGR image.
        """
        if not isinstance(frame, np.ndarray):
            raise ColorSpaceError(f"Expected numpy array, got {type(frame).__name__}")
        if frame.dtype != np.uint8:
            raise ColorSpaceError(
                f"Expected uint8 dtype, got {frame.dtype}. "
                f"Pixel values must be in range [0, 255]."
            )
        if frame.ndim != 3:
            raise ColorSpaceError(
                f"Expected 3D array (H, W, C), got {frame.ndim}D array. "
                f"Use bgr_to_gray() result for single-channel operations."
            )
        if frame.shape[2] != 3:
            raise ColorSpaceError(
                f"Expected 3 channels (BGR), got {frame.shape[2]} channels."
            )

    @staticmethod
    def _validate_gray(frame: np.ndarray) -> None:
        """Validate that a frame is a proper grayscale image."""
        if not isinstance(frame, np.ndarray):
            raise ColorSpaceError(f"Expected numpy array, got {type(frame).__name__}")
        if frame.dtype != np.uint8:
            raise ColorSpaceError(f"Expected uint8 dtype, got {frame.dtype}.")
        if frame.ndim != 2:
            raise ColorSpaceError(
                f"Expected 2D grayscale array (H, W), got {frame.ndim}D array."
            )

    # ── Conversions ───────────────────────────────────────────────────────────

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR to RGB.

        Use this before passing frames to neural networks (YOLO, etc.)
        which expect RGB input.

        Args:
            frame: BGR image, shape (H, W, 3), dtype uint8.

        Returns:
            RGB image, shape (H, W, 3), dtype uint8.
        """
        ColorSpaceConverter._validate_bgr(frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def bgr_to_gray(frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR to grayscale.

        Use this before edge detection, convolution, and morphological ops.
        Grayscale uses luminance weighting: Y = 0.299R + 0.587G + 0.114B

        Args:
            frame: BGR image, shape (H, W, 3), dtype uint8.

        Returns:
            Grayscale image, shape (H, W), dtype uint8.
        """
        ColorSpaceConverter._validate_bgr(frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def bgr_to_hsv(frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR to HSV.

        Use this for color-based filtering (e.g. detect red objects).
        Note: OpenCV H range is 0-179 (not 0-360) to fit in uint8.

        Args:
            frame: BGR image, shape (H, W, 3), dtype uint8.

        Returns:
            HSV image, shape (H, W, 3), dtype uint8.
            H: 0-179, S: 0-255, V: 0-255
        """
        ColorSpaceConverter._validate_bgr(frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    @staticmethod
    def bgr_to_lab(frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR to LAB (CIELAB) color space.

        LAB is perceptually uniform — equal numeric distances correspond
        to equal perceived color differences. Useful for illumination-
        invariant processing and color distance calculations.

        Args:
            frame: BGR image, shape (H, W, 3), dtype uint8.

        Returns:
            LAB image, shape (H, W, 3), dtype uint8.
            L: 0-255, A: 0-255, B: 0-255
        """
        ColorSpaceConverter._validate_bgr(frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    @staticmethod
    def gray_to_bgr(frame: np.ndarray) -> np.ndarray:
        """
        Convert grayscale back to BGR (3-channel).

        Use this when you need to draw colored annotations on a
        grayscale-processed frame.

        Args:
            frame: Grayscale image, shape (H, W), dtype uint8.

        Returns:
            BGR image, shape (H, W, 3), dtype uint8.
        """
        ColorSpaceConverter._validate_gray(frame)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # ── Analysis ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_histogram(
        frame: np.ndarray,
        bins: int = 256,
    ) -> dict[str, np.ndarray]:
        """
        Compute per-channel histogram for a BGR image.

        Useful for understanding image exposure and color distribution.
        Each value in the returned arrays = count of pixels with that intensity.

        Args:
            frame: BGR image, shape (H, W, 3), dtype uint8.
            bins: Number of histogram bins (default 256 = one per intensity).

        Returns:
            Dict with keys 'blue', 'green', 'red', each mapping to a
            1D numpy array of length `bins`.
        """
        ColorSpaceConverter._validate_bgr(frame)
        hist_range = [0, 256]
        return {
            "blue": cv2.calcHist([frame], [0], None, [bins], hist_range).flatten(),
            "green": cv2.calcHist([frame], [1], None, [bins], hist_range).flatten(),
            "red": cv2.calcHist([frame], [2], None, [bins], hist_range).flatten(),
        }

    @staticmethod
    def mean_color(frame: np.ndarray) -> tuple[float, float, float]:
        """
        Compute mean BGR color of the entire frame.

        Args:
            frame: BGR image, shape (H, W, 3), dtype uint8.

        Returns:
            Tuple of (mean_blue, mean_green, mean_red) as floats.
        """
        ColorSpaceConverter._validate_bgr(frame)
        means = cv2.mean(frame)
        return (means[0], means[1], means[2])
