"""
tracker/processing/edges.py

Edge detection implemented from scratch using NumPy and the
filters module. No cv2.Canny() until verification.

Pipeline:
    grayscale image
        → gaussian blur        (remove noise)
        → sobel gradients      (find intensity changes)
        → non-max suppression  (thin edges to 1px)
        → hysteresis threshold (remove weak isolated edges)
        → binary edge map

Design decisions:
- Each step is a separate function: testable, debuggable, replaceable
- compute_gradients() returns both magnitude AND direction —
  direction is needed for non-max suppression
- nms() and hysteresis() operate on float32 magnitude maps —
  no uint8 clamping until the final output
- SobelEdgeDetector wraps all steps: use this in production
- Individual step functions exposed for testing and education
"""

from __future__ import annotations

import cv2
import numpy as np

from tracker.processing.filters import (
    SOBEL_X,
    SOBEL_Y,
    convolve2d_fast,
    gaussian_blur,
)

# ── Step 2: Sobel gradients ───────────────────────────────────────────────────


def compute_gradients(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient magnitude and direction using Sobel kernels.

    Applies Sobel X and Y kernels, then computes:
        magnitude = sqrt(Gx² + Gy²)
        direction = arctan2(Gy, Gx)  in radians

    Args:
        image: Grayscale image, shape (H, W), dtype uint8.

    Returns:
        magnitude: Float32 array, shape (H, W). Pixel-wise edge strength.
        direction: Float32 array, shape (H, W). Gradient angle in radians.
    """
    gx = convolve2d_fast(image, SOBEL_X)
    gy = convolve2d_fast(image, SOBEL_Y)

    magnitude = np.sqrt(gx**2 + gy**2).astype(np.float32)
    direction = np.arctan2(gy, gx).astype(np.float32)

    return magnitude, direction


# ── Step 3: Non-maximum suppression ──────────────────────────────────────────


def non_maximum_suppression(
    magnitude: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Thin edges to 1 pixel by suppressing non-local-maxima.

    For each pixel, look at its two neighbors along the gradient direction.
    If the pixel is NOT the strongest of the three, suppress it (set to 0).

    Why this works:
        An edge is a ridge in the magnitude image.
        The ridge peak is the true edge location.
        NMS keeps only ridge peaks, discarding the slopes.

    Args:
        magnitude: Gradient magnitude, shape (H, W), float32.
        direction: Gradient direction in radians, shape (H, W), float32.

    Returns:
        Suppressed magnitude, shape (H, W), float32.
        Non-maxima are set to 0.
    """
    h, w = magnitude.shape
    output = np.zeros((h, w), dtype=np.float32)

    # Convert direction to degrees and snap to 4 angles: 0, 45, 90, 135
    # This discretization maps continuous gradient direction to
    # one of 4 neighbor pairs to compare against
    angle = np.degrees(direction) % 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            mag = magnitude[i, j]

            a = angle[i, j]

            # Determine which two neighbors to compare based on gradient angle
            # 0°   → compare left/right neighbors    (horizontal edge)
            # 45°  → compare diagonal neighbors      (diagonal edge)
            # 90°  → compare top/bottom neighbors    (vertical edge)
            # 135° → compare other diagonal          (other diagonal)
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1 = magnitude[i, j + 1]
                n2 = magnitude[i, j - 1]
            elif 22.5 <= a < 67.5:
                n1 = magnitude[i + 1, j - 1]
                n2 = magnitude[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                n1 = magnitude[i + 1, j]
                n2 = magnitude[i - 1, j]
            else:
                n1 = magnitude[i - 1, j - 1]
                n2 = magnitude[i + 1, j + 1]

            # Keep pixel only if it is the local maximum
            if mag >= n1 and mag >= n2:
                output[i, j] = mag

    return output


# ── Step 4: Hysteresis thresholding ──────────────────────────────────────────


def hysteresis_threshold(
    image: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """
    Apply double thresholding and edge tracking by hysteresis.

    Three pixel classes:
        Strong (>= high):  definitely an edge — always kept
        Weak (low..high):  maybe an edge — kept only if connected to strong
        Suppressed (< low): definitely not an edge — always discarded

    Why two thresholds beat one:
        Single threshold: too high = broken edges, too low = noisy edges.
        Double threshold: keeps weak pixels that are part of real edges
        while discarding isolated weak pixels (noise).

    Args:
        image: NMS-suppressed magnitude, shape (H, W), float32.
        low: Lower threshold. Pixels below this are discarded.
        high: Upper threshold. Pixels above this are strong edges.

    Returns:
        Binary edge map, shape (H, W), dtype uint8.
        Edge pixels = 255, background = 0.
    """
    strong = np.zeros_like(image, dtype=np.uint8)
    weak = np.zeros_like(image, dtype=np.uint8)

    strong[image >= high] = 255
    weak[(image >= low) & (image < high)] = 128

    # Edge tracking: promote weak pixels connected to strong pixels
    # Use cv2.connectedComponentsWithStats for efficient connected components
    output = strong.copy()

    # Check 8-connectivity: for each weak pixel, if any of its 8 neighbors
    # is strong, promote it to strong
    # Simple approach: iterate until no more promotions
    h, w = image.shape
    changed = True
    while changed:
        changed = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak[i, j] == 128 and output[i, j] == 0:
                    # Check 8 neighbors
                    neighborhood = output[i - 1 : i + 2, j - 1 : j + 2]
                    if neighborhood.max() == 255:
                        output[i, j] = 255
                        changed = True

    return output


# ── Vectorized hysteresis (fast version) ─────────────────────────────────────


def hysteresis_threshold_fast(
    image: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """
    Fast hysteresis using cv2's connected components.

    Same result as hysteresis_threshold() but uses cv2 for
    connected component labeling — much faster on large images.

    Args:
        image: NMS-suppressed magnitude, shape (H, W), float32.
        low: Lower threshold.
        high: Upper threshold.

    Returns:
        Binary edge map, shape (H, W), dtype uint8. Edge = 255.
    """
    strong = (image >= high).astype(np.uint8) * 255

    # Find connected components in the combined strong+weak map
    combined = np.where(image >= low, 255, 0).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(combined)

    # Keep only components that contain at least one strong pixel
    output = np.zeros_like(image, dtype=np.uint8)
    for label in range(1, n_labels):
        component_mask = labels == label
        if strong[component_mask].max() == 255:
            output[component_mask] = 255

    return output


# ── Complete edge detector ────────────────────────────────────────────────────


class SobelEdgeDetector:
    """
    Complete edge detector: blur → gradients → NMS → hysteresis.

    Use detect() for production.
    Use detect_steps() to inspect intermediate results for debugging.

    Args:
        blur_kernel_size: Gaussian blur kernel size (must be odd).
        blur_sigma: Gaussian blur sigma.
        low_threshold: Hysteresis lower threshold.
        high_threshold: Hysteresis upper threshold.

    Example:
        detector = SobelEdgeDetector(low_threshold=50, high_threshold=150)
        edges = detector.detect(gray_frame)
    """

    def __init__(
        self,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
        low_threshold: float = 50.0,
        high_threshold: float = 150.0,
    ) -> None:
        if blur_kernel_size % 2 == 0:
            raise ValueError(f"blur_kernel_size must be odd, got {blur_kernel_size}.")
        if low_threshold >= high_threshold:
            raise ValueError(
                f"low_threshold ({low_threshold}) must be less than "
                f"high_threshold ({high_threshold})."
            )
        self._blur_kernel_size = blur_kernel_size
        self._blur_sigma = blur_sigma
        self._low = low_threshold
        self._high = high_threshold

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Run full edge detection pipeline.

        Args:
            image: Grayscale image, shape (H, W), dtype uint8.

        Returns:
            Binary edge map, shape (H, W), dtype uint8. Edge = 255.
        """
        blurred = gaussian_blur(image, self._blur_kernel_size, self._blur_sigma)
        magnitude, direction = compute_gradients(blurred)
        suppressed = non_maximum_suppression(magnitude, direction)
        return hysteresis_threshold_fast(suppressed, self._low, self._high)

    def detect_steps(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run pipeline and return all intermediate results.

        Use this for debugging and visualization — inspect what each
        step produces to understand where the pipeline might be failing.

        Returns:
            Dict with keys: 'blurred', 'magnitude', 'direction',
            'suppressed', 'edges'
        """
        blurred = gaussian_blur(image, self._blur_kernel_size, self._blur_sigma)
        magnitude, direction = compute_gradients(blurred)
        suppressed = non_maximum_suppression(magnitude, direction)
        edges = hysteresis_threshold_fast(suppressed, self._low, self._high)

        return {
            "blurred": blurred,
            "magnitude": magnitude,
            "direction": direction,
            "suppressed": suppressed,
            "edges": edges,
        }
