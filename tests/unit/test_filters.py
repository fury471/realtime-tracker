"""
tests/unit/test_filters.py

Tests for tracker/processing/filters.py

Testing strategy:
- Known kernels have mathematically guaranteed outputs — use these
  as ground truth instead of hardcoded magic numbers
- Compare scratch implementation against cv2 — if they match,
  our implementation is correct
- Test invariants: shape, dtype, variance properties
- Test all error paths: wrong dimensions, even kernel, bad padding
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tracker.processing.filters import (
    BOX_BLUR,
    GAUSSIAN_3X3,
    IDENTITY,
    SOBEL_X,
    SOBEL_Y,
    convolve2d,
    convolve2d_fast,
    gaussian_blur,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_gradient_image(h: int = 50, w: int = 50) -> np.ndarray:
    """
    Create a grayscale image with a smooth horizontal gradient.
    Pixel value = column index × 5, clamped to [0, 255].
    Useful for testing edge detection — strong vertical edge at center.
    """
    image = np.zeros((h, w), dtype=np.uint8)
    for j in range(w):
        image[:, j] = min(j * 5, 255)
    return image


def make_uniform_image(value: int = 128, h: int = 50, w: int = 50) -> np.ndarray:
    """Uniform image — all pixels same value. Useful for blur testing."""
    return np.full((h, w), value, dtype=np.uint8)


def make_random_image(h: int = 50, w: int = 50, seed: int = 42) -> np.ndarray:
    """Random image with fixed seed for reproducibility."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


# ── Identity kernel ───────────────────────────────────────────────────────────


def test_identity_kernel_preserves_center_pixel() -> None:
    """
    Identity kernel must leave every pixel unchanged.
    This is the mathematical definition of an identity operation.
    """
    image = make_random_image()
    result = convolve2d(image, IDENTITY)
    # Center pixels (away from border) must be exactly preserved
    np.testing.assert_allclose(
        result[1:-1, 1:-1],
        image[1:-1, 1:-1].astype(np.float32),
        atol=1e-4,
    )


def test_identity_kernel_preserves_shape() -> None:
    """Output shape must equal input shape for any kernel."""
    image = make_random_image(h=37, w=53)
    result = convolve2d(image, IDENTITY)
    assert result.shape == image.shape


def test_identity_kernel_output_is_float32() -> None:
    """convolve2d always returns float32 — caller clips/casts as needed."""
    result = convolve2d(make_random_image(), IDENTITY)
    assert result.dtype == np.float32


# ── Box blur ──────────────────────────────────────────────────────────────────


def test_box_blur_reduces_variance() -> None:
    """
    Blur must reduce pixel variance — this is the definition of smoothing.
    A uniform image has zero variance; blur moves any image toward uniform.
    """
    image = make_random_image()
    blurred = convolve2d(image, BOX_BLUR)
    assert blurred.var() < image.astype(float).var()


def test_box_blur_uniform_image_unchanged() -> None:
    """
    Applying any blur to a uniform image must leave values unchanged.
    Average of identical values = same value.
    """
    image = make_uniform_image(value=128)
    blurred = convolve2d(image, BOX_BLUR)
    np.testing.assert_allclose(
        blurred[1:-1, 1:-1],
        128.0,
        atol=1e-3,
    )


def test_box_blur_output_in_valid_range() -> None:
    """Blur of uint8 input must stay within [0, 255]."""
    image = make_random_image()
    blurred = convolve2d(image, BOX_BLUR)
    assert blurred.min() >= 0.0
    assert blurred.max() <= 255.0


# ── Sobel kernels ─────────────────────────────────────────────────────────────


def test_sobel_x_detects_vertical_edges() -> None:
    """
    Sobel X must give high response where intensity changes left-to-right.
    Gradient image has strong vertical edges — Sobel X should fire strongly.
    """
    image = make_gradient_image()
    edges = convolve2d(image, SOBEL_X)
    # Interior of gradient image: most pixels should have positive response
    interior = edges[1:-1, 1:-1]
    assert interior.mean() > 0, "Sobel X should detect gradient direction"


def test_sobel_x_uniform_image_gives_zero() -> None:
    """
    Sobel X on a uniform image must give exactly zero everywhere.
    No intensity change = no edge = zero gradient.
    """
    image = make_uniform_image(value=128)
    edges = convolve2d(image, SOBEL_X)
    np.testing.assert_allclose(edges[1:-1, 1:-1], 0.0, atol=1e-3)


def test_sobel_y_uniform_image_gives_zero() -> None:
    """Same invariant for Sobel Y — uniform image has no horizontal edges."""
    image = make_uniform_image(value=128)
    edges = convolve2d(image, SOBEL_Y)
    np.testing.assert_allclose(edges[1:-1, 1:-1], 0.0, atol=1e-3)


def test_sobel_x_and_y_are_orthogonal() -> None:
    """
    Sobel X detects vertical edges, Sobel Y detects horizontal edges.
    On a horizontal gradient: Sobel X fires, Sobel Y does not.
    """
    image = make_gradient_image()
    edges_x = convolve2d(image, SOBEL_X)
    edges_y = convolve2d(image, SOBEL_Y)
    # X response much stronger than Y on horizontal gradient
    assert edges_x[1:-1, 1:-1].mean() > edges_y[1:-1, 1:-1].mean()


# ── Match cv2 ─────────────────────────────────────────────────────────────────


def test_scratch_matches_cv2_identity() -> None:
    """
    Our convolve2d must match cv2.filter2D for identity kernel.
    If they match, our implementation is correct — cv2 is ground truth.
    """
    image = make_random_image()
    ours = convolve2d(image, IDENTITY)
    cv2_result = convolve2d_fast(image, IDENTITY)
    np.testing.assert_allclose(ours, cv2_result, atol=1.0)


def test_scratch_matches_cv2_box_blur() -> None:
    """Our box blur must match cv2.filter2D to within 1 intensity unit."""
    image = make_random_image()
    ours = convolve2d(image, BOX_BLUR)
    cv2_result = convolve2d_fast(image, BOX_BLUR)
    np.testing.assert_allclose(ours, cv2_result, atol=1.0)


def test_scratch_matches_cv2_sobel_x() -> None:
    """Our Sobel X must match cv2.filter2D."""
    image = make_random_image()
    ours = convolve2d(image, SOBEL_X)
    cv2_result = convolve2d_fast(image, SOBEL_X)
    np.testing.assert_allclose(ours, cv2_result, atol=1.0)


def test_scratch_matches_cv2_gaussian() -> None:
    """Our Gaussian kernel must match cv2.filter2D."""
    image = make_random_image()
    ours = convolve2d(image, GAUSSIAN_3X3)
    cv2_result = convolve2d_fast(image, GAUSSIAN_3X3)
    np.testing.assert_allclose(ours, cv2_result, atol=1.0)


# ── Speed comparison ──────────────────────────────────────────────────────────


def test_cv2_faster_than_scratch() -> None:
    """
    cv2.filter2D must be faster than our Python implementation.
    This is not just a performance test — it documents WHY we have
    two implementations: scratch for understanding, cv2 for production.
    """
    image = make_random_image(h=100, w=100)

    start = time.perf_counter()
    for _ in range(3):
        convolve2d(image, BOX_BLUR)
    scratch_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(3):
        convolve2d_fast(image, BOX_BLUR)
    cv2_time = time.perf_counter() - start

    speedup = scratch_time / cv2_time
    print(f"\nSpeedup: cv2 is {speedup:.0f}x faster than scratch implementation")
    assert (
        cv2_time < scratch_time
    ), f"cv2 ({cv2_time:.4f}s) should be faster than scratch ({scratch_time:.4f}s)"


# ── Gaussian blur ─────────────────────────────────────────────────────────────


def test_gaussian_blur_reduces_variance() -> None:
    """Gaussian blur must smooth the image — reduce variance."""
    image = make_random_image()
    blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)
    assert blurred.var() < image.astype(float).var()


def test_gaussian_blur_preserves_shape() -> None:
    """Gaussian blur must preserve image dimensions."""
    image = make_random_image(h=48, w=64)
    blurred = gaussian_blur(image)
    assert blurred.shape == image.shape


def test_gaussian_blur_odd_kernel_only() -> None:
    """Even kernel size must raise ValueError."""
    image = make_random_image()
    with pytest.raises(ValueError) as exc_info:
        gaussian_blur(image, kernel_size=4)
    assert "odd" in str(exc_info.value)


# ── Validation ────────────────────────────────────────────────────────────────


def test_3d_input_raises_value_error() -> None:
    """3D input (BGR image) must raise ValueError — use grayscale."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        convolve2d(image, IDENTITY)
    assert "2D" in str(exc_info.value)


def test_even_kernel_raises_value_error() -> None:
    """Even-sized kernel must raise ValueError."""
    image = make_random_image()
    even_kernel = np.ones((4, 4), dtype=np.float32) / 16.0
    with pytest.raises(ValueError) as exc_info:
        convolve2d(image, even_kernel)
    assert "odd" in str(exc_info.value)


def test_invalid_padding_raises_value_error() -> None:
    """Unknown padding mode must raise ValueError."""
    image = make_random_image()
    with pytest.raises(ValueError) as exc_info:
        convolve2d(image, IDENTITY, padding="same")
    assert "padding" in str(exc_info.value).lower()


def test_2d_kernel_required() -> None:
    """1D kernel must raise ValueError."""
    image = make_random_image()
    with pytest.raises(ValueError):
        convolve2d(image, np.ones(3, dtype=np.float32))
