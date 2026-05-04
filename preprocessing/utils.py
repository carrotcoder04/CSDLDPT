"""
utils.py — Utility functions for image I/O and visualization
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and return as RGB numpy array.

    Args:
        image_path: Path to the image file (jpg / png / webp …)

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8, RGB order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file cannot be decoded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use Pillow so that EXIF orientation is respected and transparency is handled
    pil_img = Image.open(image_path).convert("RGB")
    img = np.array(pil_img)
    return img


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save a numpy RGB image to disk.

    Args:
        image:       np.ndarray (H, W, 3) uint8, RGB order.
        output_path: Destination file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """Save a binary mask (0/255) to disk."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, mask)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_pipeline(
    stages: dict,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (22, 5),
) -> None:
    """
    Render a grid of pipeline stage images side-by-side.

    Args:
        stages:      OrderedDict / dict  {stage_name: np.ndarray (RGB)} 
                     Mask arrays (2-D) are converted to 3-channel grey for display.
        output_path: If given, saves the figure to this path.
        figsize:     Matplotlib figure size.
    """
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, stages.items()):
        if img is None:
            ax.axis("off")
            ax.set_title(name, fontsize=9, pad=4)
            continue

        display = img
        # If mask / single-channel → convert to 3-ch for uniform display
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)

        ax.imshow(display)
        ax.set_title(name, fontsize=9, pad=4)
        ax.axis("off")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved pipeline visualization → {output_path}")
    plt.close(fig)


def draw_bbox_on_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box on a copy of the image.

    Args:
        image:     RGB image.
        bbox:      (x, y, w, h) in pixel coordinates.
        color:     RGB color.
        thickness: Line thickness.

    Returns:
        New image with bounding box drawn (RGB).
    """
    vis = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), color[::-1], thickness)  # OpenCV BGR
    return vis
