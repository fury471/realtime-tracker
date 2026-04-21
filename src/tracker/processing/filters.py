"""
tracker/processing/filters.py

Image filtering and convolution primitives implemented from scratch
using only NumPy. No cv2, no scipy — pure array math.

Why implement from scratch:
- Understanding what cv2.filter2D() actually computes
- CNN intuition: every conv layer does exactly this, with learned kernels
- Interview readiness: "explain convolution" becomes trivial after this

Design decisions:
- convolve2d() operates on single-channel (grayscale) images only
  Multi-channel convolution = apply independently to each channel
- 'reflect' padding mirrors the border pixels outward
  This avoids the dark border artifact that 'zero' padding produces
- Output dtype is float32 internally, caller clips/casts as needed
  Intermediate values can exceed [0,255] during computation

Kernel reference:
    Identity  = [[0,0,0],[0,1,0],[0,0,0]]  → no change
    Box blur  = ones(3,3)/9                → average blur
    Gaussian  = approximated with binomial → smooth blur
    Sobel X   = [[-1,0,1],[-2,0,2],[-1,0,1]] → vertical edges
    Sobel Y   = [[-1,-2,-1],[0,0,0],[1,2,1]] → horizontal edges
    Sharpen   = [[0,-1,0],[-1,5,-1],[0,-1,0]] → enhance edges
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Kernels ───────────────────────────────────────────────────────────────────
# Pre-built kernels as module-level constants.
# Import and use directly: from tracker.processing.filters import SOBEL_X


IDENTITY = np.array(
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    dtype=np.float32,
)

BOX_BLUR = np.ones((3, 3), dtype=np.float32) / 9.0

GAUSSIAN_3X3 = (
    np.array(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        dtype=np.float32,
    )
    / 16.0
)

GAUSSIAN_5X5 = (
    np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=np.float32,
    )
    / 256.0
)

SOBEL_X = np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    dtype=np.float32,
)

SOBEL_Y = np.array(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    dtype=np.float32,
)

SHARPEN = np.array(
    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    dtype=np.float32,
)


# ── Core convolution ──────────────────────────────────────────────────────────


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: str = "reflect",
) -> np.ndarray:
    """
    Apply a 2D convolution kernel to a grayscale image.

    Implemented from scratch using NumPy — no cv2 or scipy.
    Equivalent to cv2.filter2D(image, -1, kernel) for 2D input.

    How it works:
        1. Pad the image so border pixels have neighbors
        2. Slide the kernel over every pixel position
        3. At each position: element-wise multiply patch × kernel, sum result
        4. Store the sum as the output pixel

    Args:
        image: Grayscale image, shape (H, W), dtype uint8 or float32.
        kernel: 2D convolution kernel, shape (kH, kW).
                Must have odd dimensions (3x3, 5x5, etc.)
        padding: Border handling strategy.
                 'reflect' — mirror pixels at border (recommended)
                 'zero'    — pad with zeros (causes dark border artifacts)

    Returns:
        Convolved image, shape (H, W), dtype float32.
        Call .clip(0, 255).astype(np.uint8) to convert back to uint8.

    Raises:
        ValueError: If kernel dimensions are not odd.
        ValueError: If image is not 2D.
        ValueError: If padding mode is not supported.

    Example:
        gray = ColorSpaceConverter.bgr_to_gray(frame)
        blurred = convolve2d(gray, GAUSSIAN_3X3)
        result = blurred.clip(0, 255).astype(np.uint8)
    """
    # ── Validation ────────────────────────────────────────────────────────────
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got {image.ndim}D array. "
            f"Convert to grayscale first with ColorSpaceConverter.bgr_to_gray()."
        )

    if kernel.ndim != 2:
        raise ValueError(f"Expected 2D kernel, got {kernel.ndim}D array.")

    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(
            f"Kernel dimensions must be odd (3x3, 5x5, etc.), got {kh}x{kw}."
        )

    if padding not in ("reflect", "zero"):
        raise ValueError(
            f"Unsupported padding mode '{padding}'. Use 'reflect' or 'zero'."
        )

    # ── Padding ───────────────────────────────────────────────────────────────
    # How much to pad on each side = kernel half-size
    # A 3x3 kernel needs 1 pixel of padding on each side
    # A 5x5 kernel needs 2 pixels of padding on each side
    pad_h = kh // 2
    pad_w = kw // 2

    if padding == "reflect":
        # Mirror: edge pixel [a,b,c,...] becomes [...,b,a | a,b,c,... | ...,c,b]
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    else:
        # Zero: pad with zeros — simpler but causes dark borders
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    # ── Convolution ───────────────────────────────────────────────────────────
    # Work in float32 — intermediate values can exceed [0,255]
    img_float = padded.astype(np.float32)
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float32)

    # Slide kernel over every output pixel
    for i in range(h):
        for j in range(w):
            # Extract the patch centered at (i, j)
            patch = img_float[i : i + kh, j : j + kw]
            # Element-wise multiply and sum = one convolution step
            output[i, j] = (patch * kernel).sum()

    return output


def convolve2d_fast(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: str = "reflect",
) -> np.ndarray:
    """
    Fast convolution using cv2.filter2D — same result as convolve2d().

    Use this in production. Use convolve2d() to understand what it does.
    cv2.filter2D is implemented in C++ and is ~1000x faster.

    Args:
        image: Grayscale image, shape (H, W), dtype uint8.
        kernel: 2D kernel, shape (kH, kW), dtype float32.
        padding: Border handling ('reflect' or 'zero').

    Returns:
        Convolved image, shape (H, W), dtype float32.
    """
    border = cv2.BORDER_REFLECT_101 if padding == "reflect" else cv2.BORDER_CONSTANT
    return cv2.filter2D(
        image.astype(np.float32),
        ddepth=-1,
        kernel=kernel,
        borderType=border,
    )


# ── Gaussian blur ─────────────────────────────────────────────────────────────


def gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian blur using cv2 (production-ready, fast).

    Use this before edge detection to reduce noise.
    Gaussian blur is separable — cv2 applies it as two 1D passes,
    which is much faster than a single 2D convolution.

    Args:
        image: Grayscale image, shape (H, W), dtype uint8.
        kernel_size: Size of Gaussian kernel (must be odd).
        sigma: Standard deviation of the Gaussian.
               Larger sigma = stronger blur = more noise reduction.

    Returns:
        Blurred image, shape (H, W), dtype uint8.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
