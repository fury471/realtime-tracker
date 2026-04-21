"""
tests/unit/test_edges.py

Tests for tracker/processing/edges.py

Testing strategy:
- Synthetic images with known edge locations — no real images needed
- Test each pipeline step independently before testing the full pipeline
- Invariants: shape, dtype, value range
- Physics-based checks: uniform image has no edges, step edge has edges
- Compare fast hysteresis against slow reference implementation
"""

from __future__ import annotations

import numpy as np
import pytest

from tracker.processing.edges import (
    SobelEdgeDetector,
    compute_gradients,
    hysteresis_threshold,
    hysteresis_threshold_fast,
    non_maximum_suppression,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def vertical_step_edge(h: int = 60, w: int = 60) -> np.ndarray:
    """
    Grayscale image with a sharp vertical edge at the center column.
    Left half = 0, right half = 255.
    Expected: strong edges at center column after detection.
    """
    image = np.zeros((h, w), dtype=np.uint8)
    image[:, w // 2 :] = 255
    return image


def horizontal_step_edge(h: int = 60, w: int = 60) -> np.ndarray:
    """
    Grayscale image with a sharp horizontal edge at the center row.
    Top half = 0, bottom half = 255.
    """
    image = np.zeros((h, w), dtype=np.uint8)
    image[h // 2 :, :] = 255
    return image


def uniform_image(value: int = 128, h: int = 60, w: int = 60) -> np.ndarray:
    """Uniform image — no edges anywhere."""
    return np.full((h, w), value, dtype=np.uint8)


def random_image(h: int = 60, w: int = 60, seed: int = 42) -> np.ndarray:
    """Random noise image with fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


# ── compute_gradients ─────────────────────────────────────────────────────────


def test_gradients_return_two_arrays() -> None:
    """compute_gradients must return (magnitude, direction) tuple."""
    mag, direction = compute_gradients(uniform_image())
    assert mag.shape == (60, 60)
    assert direction.shape == (60, 60)


def test_gradients_output_is_float32() -> None:
    """Both magnitude and direction must be float32."""
    mag, direction = compute_gradients(uniform_image())
    assert mag.dtype == np.float32
    assert direction.dtype == np.float32


def test_gradients_uniform_image_is_zero() -> None:
    """
    Uniform image has no intensity changes anywhere.
    Both Gx and Gy must be zero → magnitude must be zero.
    """
    mag, _ = compute_gradients(uniform_image())
    np.testing.assert_allclose(mag[1:-1, 1:-1], 0.0, atol=1e-3)


def test_gradients_step_edge_has_high_magnitude() -> None:
    """
    Step edge must produce high gradient magnitude at the edge location.
    A 0→255 step is the strongest possible edge.
    """
    mag, _ = compute_gradients(vertical_step_edge())
    # Maximum magnitude must be significantly above zero
    assert mag.max() > 100.0


def test_gradients_vertical_edge_direction() -> None:
    """
    Vertical step edge (left-right intensity change) must produce
    gradient direction close to 0° (pointing left-to-right).
    arctan2(0, positive) = 0 radians.
    """
    image = vertical_step_edge()
    _, direction = compute_gradients(image)
    center_col = image.shape[1] // 2
    # Gradient direction at the edge should be near 0 (horizontal gradient)
    edge_directions = direction[10:-10, center_col]
    assert np.abs(edge_directions).mean() < 1.0  # close to 0 radians


def test_gradients_preserves_shape() -> None:
    """Output shape must match input shape."""
    image = random_image(h=37, w=53)
    mag, direction = compute_gradients(image)
    assert mag.shape == image.shape
    assert direction.shape == image.shape


# ── non_maximum_suppression ───────────────────────────────────────────────────


def test_nms_output_shape_preserved() -> None:
    """NMS must preserve image dimensions."""
    mag, direction = compute_gradients(vertical_step_edge())
    suppressed = non_maximum_suppression(mag, direction)
    assert suppressed.shape == mag.shape


def test_nms_output_is_float32() -> None:
    """NMS output must be float32."""
    mag, direction = compute_gradients(vertical_step_edge())
    suppressed = non_maximum_suppression(mag, direction)
    assert suppressed.dtype == np.float32


def test_nms_thins_edge() -> None:
    """
    NMS must reduce the number of edge pixels — thinning thick edges.
    Before NMS: many pixels fire near the edge.
    After NMS: only the ridge peak pixels remain.
    """
    image = vertical_step_edge()
    mag, direction = compute_gradients(image)

    # Count pixels above threshold before and after NMS
    threshold = mag.max() * 0.3
    pixels_before = (mag > threshold).sum()
    suppressed = non_maximum_suppression(mag, direction)
    pixels_after = (suppressed > threshold).sum()

    assert (
        pixels_after < pixels_before
    ), "NMS must reduce the number of above-threshold pixels"


def test_nms_uniform_image_stays_zero() -> None:
    """NMS on uniform image must produce all zeros — no edges to thin."""
    mag, direction = compute_gradients(uniform_image())
    suppressed = non_maximum_suppression(mag, direction)
    assert suppressed[1:-1, 1:-1].max() == 0.0


def test_nms_does_not_create_new_pixels() -> None:
    """NMS can only suppress pixels — it must never increase magnitude."""
    mag, direction = compute_gradients(random_image())
    suppressed = non_maximum_suppression(mag, direction)
    assert suppressed.max() <= mag.max()


# ── hysteresis_threshold ──────────────────────────────────────────────────────


def test_hysteresis_output_is_binary() -> None:
    """
    Hysteresis output must be binary — only 0 and 255.
    No intermediate values should remain after thresholding.
    """
    mag, direction = compute_gradients(vertical_step_edge())
    suppressed = non_maximum_suppression(mag, direction)
    edges = hysteresis_threshold_fast(suppressed, low=50, high=150)
    unique_values = np.unique(edges)
    for v in unique_values:
        assert v in (0, 255), f"Unexpected value {v} in edge map"


def test_hysteresis_output_dtype_is_uint8() -> None:
    """Edge map must be uint8."""
    mag, direction = compute_gradients(vertical_step_edge())
    suppressed = non_maximum_suppression(mag, direction)
    edges = hysteresis_threshold_fast(suppressed, low=50, high=150)
    assert edges.dtype == np.uint8


def test_hysteresis_uniform_produces_no_edges() -> None:
    """Uniform image must produce zero edge pixels after full pipeline."""
    mag, direction = compute_gradients(uniform_image())
    suppressed = non_maximum_suppression(mag, direction)
    edges = hysteresis_threshold_fast(suppressed, low=50, high=150)
    assert edges.max() == 0, "Uniform image should have no edges"


def test_hysteresis_step_edge_produces_edges() -> None:
    """Step edge must produce edge pixels after full pipeline."""
    mag, direction = compute_gradients(vertical_step_edge())
    suppressed = non_maximum_suppression(mag, direction)
    edges = hysteresis_threshold_fast(suppressed, low=30, high=100)
    assert (edges == 255).sum() > 0, "Step edge should produce edge pixels"


def test_hysteresis_fast_matches_reference() -> None:
    """
    Fast hysteresis (cv2 connected components) must produce the
    same result as the reference Python implementation.
    """
    mag, direction = compute_gradients(vertical_step_edge())
    suppressed = non_maximum_suppression(mag, direction)

    ref = hysteresis_threshold(suppressed, low=30, high=100)
    fast = hysteresis_threshold_fast(suppressed, low=30, high=100)

    # Both must agree on strong edges (above high threshold)
    strong_mask = suppressed >= 100
    np.testing.assert_array_equal(
        ref[strong_mask],
        fast[strong_mask],
        err_msg="Both implementations must agree on strong edge pixels",
    )


# ── SobelEdgeDetector ─────────────────────────────────────────────────────────


def test_detector_output_shape_preserved() -> None:
    """detect() must return same shape as input."""
    detector = SobelEdgeDetector()
    image = vertical_step_edge()
    edges = detector.detect(image)
    assert edges.shape == image.shape


def test_detector_output_is_binary_uint8() -> None:
    """detect() must return binary uint8 map."""
    detector = SobelEdgeDetector()
    edges = detector.detect(vertical_step_edge())
    assert edges.dtype == np.uint8
    unique = np.unique(edges)
    for v in unique:
        assert v in (0, 255)


def test_detector_finds_vertical_edge() -> None:
    """Detector must find edges in the correct column range."""
    detector = SobelEdgeDetector(low_threshold=30, high_threshold=100)
    image = vertical_step_edge(h=60, w=60)
    edges = detector.detect(image)

    # Edge pixels must be concentrated around center column (col 30)
    edge_cols = np.where((edges == 255).any(axis=0))[0]
    assert len(edge_cols) > 0, "No edges found"
    assert edge_cols.min() >= 25, "Edge too far left"
    assert edge_cols.max() <= 35, "Edge too far right"


def test_detector_no_edges_on_uniform() -> None:
    """Uniform image must produce zero edges."""
    detector = SobelEdgeDetector()
    edges = detector.detect(uniform_image())
    assert edges.max() == 0


def test_detector_steps_returns_all_keys() -> None:
    """detect_steps() must return all intermediate results."""
    detector = SobelEdgeDetector()
    steps = detector.detect_steps(vertical_step_edge())
    required_keys = {"blurred", "magnitude", "direction", "suppressed", "edges"}
    assert set(steps.keys()) == required_keys


def test_detector_steps_consistent_with_detect() -> None:
    """
    detect_steps()['edges'] must equal detect() output.
    Both run the same pipeline — results must be identical.
    """
    detector = SobelEdgeDetector()
    image = vertical_step_edge()
    edges_direct = detector.detect(image)
    edges_steps = detector.detect_steps(image)["edges"]
    np.testing.assert_array_equal(edges_direct, edges_steps)


# ── SobelEdgeDetector validation ──────────────────────────────────────────────


def test_detector_even_blur_kernel_raises() -> None:
    """Even blur kernel size must raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        SobelEdgeDetector(blur_kernel_size=4)
    assert "odd" in str(exc_info.value)


def test_detector_low_above_high_raises() -> None:
    """low_threshold >= high_threshold must raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        SobelEdgeDetector(low_threshold=150, high_threshold=50)
    assert "low_threshold" in str(exc_info.value)


def test_detector_equal_thresholds_raises() -> None:
    """Equal thresholds must also raise ValueError."""
    with pytest.raises(ValueError):
        SobelEdgeDetector(low_threshold=100, high_threshold=100)
