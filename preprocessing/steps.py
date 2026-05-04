"""
steps.py — Individual preprocessing steps (each step is a pure function)
=========================================================================

Pipeline steps (in order):
  1. validate_image      – filter invalid / non-tree images
  2. fix_orientation     – correct EXIF / rotation
  3. normalize_image     – resize + pixel normalization
  4. denoise_image       – light denoising / blur
  5. segment_tree        – background removal (rembg)
  6. crop_to_tree        – tight crop around the tree mask
  7. center_and_scale    – center tree + uniform output size
  8. augment_image       – optional data augmentation
"""

import random
import warnings
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags

# rembg for background removal (ONNX-based U2-Net)
try:
    from rembg import remove as rembg_remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    warnings.warn("rembg not installed — segment_tree will use GrabCut fallback.")


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

RGBImage = np.ndarray   # shape (H, W, 3), dtype uint8
Mask     = np.ndarray   # shape (H, W),    dtype uint8  (0 or 255)
BBox     = Tuple[int, int, int, int]  # (x, y, w, h)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Validate Image
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationResult:
    """Container returned by validate_image()."""

    __slots__ = ("is_valid", "reason", "issues")

    def __init__(self, is_valid: bool, reason: str = "", issues: list = None):
        self.is_valid = is_valid
        self.reason   = reason
        self.issues   = issues or []

    def __bool__(self):
        return self.is_valid

    def __repr__(self):
        return f"ValidationResult(valid={self.is_valid}, reason='{self.reason}')"


def validate_image(
    image: RGBImage,
    min_size: int = 64,
    max_size: int = 8000,
    green_ratio_threshold: float = 0.04,
    edge_density_threshold: float = 0.02,
    illustration_sat_threshold: float = 0.75,
) -> ValidationResult:
    """
    Validate whether the image is a suitable full-tree photograph.

    Checks performed
    ────────────────
    1. Minimum / maximum dimensions  → reject tiny / huge images
    2. Green channel dominance       → ensure some vegetation is present
    3. Edge density                  → reject flat / vector-like illustrations
    4. Colour saturation heuristic   → detect synthetic / illustrated images
    5. Aspect-ratio check            → flag extreme close-up crops

    Args:
        image:                     RGB image (H, W, 3) uint8.
        min_size:                  Minimum dimension in pixels.
        max_size:                  Maximum dimension in pixels.
        green_ratio_threshold:     Minimum fraction of pixels that appear green.
        edge_density_threshold:    Minimum Canny edge density expected.
        illustration_sat_threshold: If mean HSV-S > this in Lab synthetic image
                                    heuristic, flag as possible illustration.

    Returns:
        ValidationResult with is_valid, reason, issues list.
    """
    issues = []

    h, w = image.shape[:2]

    # ── Check 1: size ────────────────────────────────────────────────────────
    if min(h, w) < min_size:
        return ValidationResult(False, f"Image too small ({w}×{h} < {min_size}px)", issues)
    if max(h, w) > max_size:
        issues.append(f"Very large image ({w}×{h}), will be resized")

    # ── Check 2: green vegetation presence ───────────────────────────────────
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Hue 35–85 °  covers yellow-green → green → cyan-green
    green_mask = (
        (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 85) &
        (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 30)
    )
    green_ratio = green_mask.mean()
    if green_ratio < green_ratio_threshold:
        issues.append(
            f"Low green ratio ({green_ratio:.3f} < {green_ratio_threshold}) – "
            "image may not contain a tree"
        )

    # ── Check 3: edge density (distinguish photo vs. illustration) ───────────
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    if edge_density < edge_density_threshold:
        issues.append(
            f"Very low edge density ({edge_density:.4f}) – "
            "possible blank/solid illustration"
        )

    # ── Check 4: illustration / vector detection ──────────────────────────────
    # Illustrations tend to have very few unique colours per block
    small = cv2.resize(image, (64, 64))
    unique_colours = len(np.unique(small.reshape(-1, 3), axis=0))
    if unique_colours < 80:
        issues.append(
            f"Very few unique colours ({unique_colours}) – "
            "image may be a diagram or illustration"
        )

    # ── Check 5: extremely tall / narrow (likely a cropped leaf shot) ─────────
    aspect = max(h, w) / max(min(h, w), 1)
    if aspect > 5.0:
        issues.append(
            f"Extreme aspect ratio ({aspect:.1f}:1) – "
            "possible close-up / non-full-tree image"
        )

    # ── Final decision ────────────────────────────────────────────────────────
    # Hard-fail only on critical checks
    critical = [i for i in issues if "may not contain a tree" in i or
                "diagram or illustration" in i]
    if len(critical) >= 2:
        return ValidationResult(False, "; ".join(critical), issues)

    reason = "; ".join(issues) if issues else "OK"
    return ValidationResult(True, reason, issues)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Fix Orientation
# ═══════════════════════════════════════════════════════════════════════════════

_EXIF_ORIENTATION_TAG = next(
    (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
)

_ORIENTATION_TO_ROTATE = {
    3: cv2.ROTATE_180,
    6: cv2.ROTATE_90_COUNTERCLOCKWISE,   # EXIF 6 = camera rotated 90° CW → correct CCW
    8: cv2.ROTATE_90_CLOCKWISE,
}


def fix_orientation(image: RGBImage, pil_image: Optional[Image.Image] = None) -> RGBImage:
    """
    Auto-correct image orientation using EXIF metadata.

    If no EXIF data is present the image is returned unchanged.

    Args:
        image:     RGB numpy array.
        pil_image: Corresponding PIL image (used to read EXIF). If None,
                   EXIF correction is skipped (image already decoded).

    Returns:
        Orientation-corrected RGB numpy array.
    """
    if pil_image is None or _EXIF_ORIENTATION_TAG is None:
        return image

    try:
        exif = pil_image._getexif()  # type: ignore[attr-defined]
        if exif is None:
            return image
        orientation = exif.get(_EXIF_ORIENTATION_TAG, 1)
        rotate_code = _ORIENTATION_TO_ROTATE.get(orientation)
        if rotate_code is not None:
            image = cv2.rotate(image, rotate_code)
    except Exception:
        pass  # Non-critical: ignore EXIF errors silently

    return image


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Normalize Image
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_image(
    image: RGBImage,
    target_size: Tuple[int, int] = (512, 512),
    normalize_pixels: bool = False,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225),
    correct_exposure: bool = True,
) -> RGBImage:
    """
    Resize image and optionally normalize pixel values.

    Steps
    ─────
    1. Resize to *target_size* (keeps content, pads with neutral grey if needed).
    2. Histogram equalization / CLAHE for exposure correction.
    3. Pixel normalization to [0, 1] or ImageNet mean/std (optional).

    Note: The function always returns uint8 unless normalize_pixels=True,
          in which case it returns float32.

    Args:
        image:            RGB uint8 array.
        target_size:      (width, height) of the output.
        normalize_pixels: If True, output is float32 normalized by mean/std.
        mean:             Per-channel mean (used only when normalize_pixels=True).
        std:              Per-channel std  (used only when normalize_pixels=True).
        correct_exposure: Apply CLAHE on L channel to reduce lighting variance.

    Returns:
        Resized (and optionally normalized) image.
    """
    h, w = image.shape[:2]
    tw, th = target_size

    # ── Letterbox resize (preserve aspect, pad with grey 128) ────────────────
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    pad_top  = (th - nh) // 2
    pad_left = (tw - nw) // 2
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
    image = canvas

    # ── CLAHE on luminance (reduce harsh lighting) ────────────────────────────
    if correct_exposure:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ── Optional pixel normalization ──────────────────────────────────────────
    if normalize_pixels:
        img_f = image.astype(np.float32) / 255.0
        img_f = (img_f - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        return img_f

    return image


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – Denoise Image
# ═══════════════════════════════════════════════════════════════════════════════

def denoise_image(
    image: RGBImage,
    method: str = "nlmeans",
    gaussian_ksize: int = 3,
    nlmeans_h: float = 6.0,
    nlmeans_template_ws: int = 7,
    nlmeans_search_ws: int = 21,
    remove_watermark: bool = True,
) -> RGBImage:
    """
    Reduce noise and optionally suppress light watermarks.

    Methods
    ───────
    * ``"gaussian"``  – Fast Gaussian blur (σ proportional to ksize).
    * ``"bilateral"`` – Edge-preserving bilateral filter.
    * ``"nlmeans"``   – Non-local means (slow but high quality).
    * ``"none"``      – Skip denoising (pass-through).

    Args:
        image:               RGB uint8 image.
        method:              One of ``"gaussian"``, ``"bilateral"``,
                             ``"nlmeans"``, ``"none"``.
        gaussian_ksize:      Kernel size for Gaussian blur (odd integer).
        nlmeans_h:           Filter strength for NL-means (higher = smoother).
        nlmeans_template_ws: Template window size (NL-means).
        nlmeans_search_ws:   Search window size (NL-means).
        remove_watermark:    Apply a light unsharp-mask AFTER denoising to
                             reduce near-transparent overlays.

    Returns:
        Denoised RGB image (uint8).
    """
    method = method.lower()

    if method == "none":
        out = image.copy()

    elif method == "gaussian":
        k = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
        out = cv2.GaussianBlur(image, (k, k), 0)

    elif method == "bilateral":
        out = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    elif method == "nlmeans":
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bgr = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            h=nlmeans_h,
            hColor=nlmeans_h,
            templateWindowSize=nlmeans_template_ws,
            searchWindowSize=nlmeans_search_ws,
        )
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    else:
        raise ValueError(f"Unknown denoising method: '{method}'. "
                         "Choose 'gaussian', 'bilateral', 'nlmeans', or 'none'.")

    # ── Light watermark suppression ───────────────────────────────────────────
    # Watermarks usually appear as semi-transparent bright text/logos.
    # A simple high-frequency subtraction attenuates them slightly.
    if remove_watermark:
        blurred = cv2.GaussianBlur(out, (21, 21), 0)
        # Unsharp masking: original - 0.15 * (original - blurred)
        # This is a mild approach; aggressive removal needs a dedicated model.
        alpha = 0.15
        out = np.clip(
            out.astype(np.float32) - alpha * (out.astype(np.float32) - blurred),
            0, 255
        ).astype(np.uint8)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Segment Tree (Background Removal)
# ═══════════════════════════════════════════════════════════════════════════════

def _grabcut_segment(image: RGBImage) -> Mask:
    """
    GrabCut-based tree segmentation fallback (no deep learning required).

    Uses a conservative central rectangle as the initial foreground hint.
    Post-processes the mask with green-channel boosting and morphological ops.

    Returns:
        Binary mask uint8 (255 = tree, 0 = background).
    """
    h, w = image.shape[:2]
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initial rectangle: occupy central 70% of the image
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.10)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    gc_mask  = np.zeros((h, w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(bgr, gc_mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # GrabCut can fail on very small images; return full mask
        return np.full((h, w), 255, dtype=np.uint8)

    # Pixels marked definite-FG or probable-FG
    binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # ── Boost with green-pixel heuristic ─────────────────────────────────────
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green_mask = (
        (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 90) &
        (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 25)
    ).astype(np.uint8) * 255
    # Add green pixels that GrabCut might have mis-classified as BG
    binary = cv2.bitwise_or(binary, green_mask)

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Keep only the largest connected component
    binary = _keep_largest_component(binary)

    return binary


def _rembg_segment(image: RGBImage, session=None) -> Mask:
    """
    Run rembg (U2-Net) to remove background and return a binary mask.

    Args:
        image:   RGB uint8 array.
        session: Pre-loaded rembg session (reuse for speed). If None a new
                 session is created each call.

    Returns:
        Binary mask uint8 (255 = foreground / tree, 0 = background).
    """
    pil_in = Image.fromarray(image, mode="RGB")
    pil_out = rembg_remove(pil_in, session=session)          # → RGBA PIL
    alpha = np.array(pil_out)[:, :, 3]                       # alpha channel
    binary = (alpha > 127).astype(np.uint8) * 255
    # Post-process
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = _keep_largest_component(binary)
    return binary


def _keep_largest_component(mask: Mask) -> Mask:
    """Retain only the largest connected component in a binary mask."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask
    # stats[0] is the background component; skip it
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_idx).astype(np.uint8) * 255


def segment_tree(
    image: RGBImage,
    method: str = "rembg",
    rembg_session=None,
    white_background: bool = True,
) -> Tuple[Mask, RGBImage]:
    """
    Separate the tree from the background.

    Args:
        image:            RGB uint8 array.
        method:           ``"rembg"`` (U2-Net, best quality) or
                          ``"grabcut"`` (fast, no GPU).
        rembg_session:    Pre-loaded rembg session for batch efficiency.
        white_background: If True, background pixels are set to white (255);
                          otherwise they remain as-is (original background,
                          useful for debugging).

    Returns:
        (mask, segmented_image)
            mask             – binary Mask (255 = tree).
            segmented_image  – RGB image with background replaced.
    """
    if method == "rembg":
        if not REMBG_AVAILABLE:
            warnings.warn("rembg not available, falling back to GrabCut.")
            method = "grabcut"
        else:
            try:
                mask = _rembg_segment(image, session=rembg_session)
            except Exception as e:
                warnings.warn(f"rembg failed ({e}), falling back to GrabCut.")
                method = "grabcut"

    if method == "grabcut":
        mask = _grabcut_segment(image)

    # ── Apply mask to image ───────────────────────────────────────────────────
    mask_3ch = np.stack([mask] * 3, axis=-1)
    bg_value  = 255 if white_background else None

    if bg_value is not None:
        bg = np.full_like(image, bg_value)
        segmented = np.where(mask_3ch > 0, image, bg)
    else:
        segmented = image.copy()

    return mask, segmented.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 – Crop to Tree
# ═══════════════════════════════════════════════════════════════════════════════

def crop_to_tree(
    image: RGBImage,
    mask: Mask,
    padding_ratio: float = 0.07,
) -> Tuple[RGBImage, Mask, BBox]:
    """
    Crop the image tightly around the tree bounding box.

    Args:
        image:         RGB image (H, W, 3).
        mask:          Binary mask (H, W) — 255 = tree.
        padding_ratio: Fractional padding added around the bounding box
                       (relative to the bounding-box dimension).

    Returns:
        (cropped_image, cropped_mask, bbox)
            cropped_image – Cropped RGB image.
            cropped_mask  – Corresponding cropped mask.
            bbox          – (x, y, w, h) bounding box in the ORIGINAL image.
    """
    h, w = image.shape[:2]

    # Find bounding box from mask
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        # No foreground found → return original
        return image, mask, (0, 0, w, h)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Ensure bounding box has at least 1px in each dimension
    if x_max <= x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1

    bw = x_max - x_min
    bh = y_max - y_min
    bbox = (int(x_min), int(y_min), int(bw), int(bh))

    # Add padding (minimum 2px so crop is never 0-sized)
    pad_x = max(2, int(bw * padding_ratio))
    pad_y = max(2, int(bh * padding_ratio))
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    # Final safety: ensure non-empty slice
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    cropped_image = image[y1:y2, x1:x2]
    cropped_mask  = mask[y1:y2, x1:x2]

    return cropped_image, cropped_mask, bbox


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 – Center and Scale
# ═══════════════════════════════════════════════════════════════════════════════

def center_and_scale(
    image: RGBImage,
    mask:  Mask,
    output_size: Tuple[int, int] = (512, 512),
    background_value: int = 255,
) -> Tuple[RGBImage, Mask]:
    """
    Place the cropped tree in the centre of a square canvas with uniform scale.

    The tree is scaled so that its longest dimension fills `scale_fill` fraction
    of the output canvas, ensuring:
      * The tree occupies most of the canvas (not tiny).
      * A consistent visual scale across all images (key for retrieval).

    Args:
        image:            Cropped RGB image containing the tree.
        mask:             Corresponding binary mask.
        output_size:      (width, height) of the square output canvas.
        background_value: Pixel fill value for the empty canvas area (white=255).

    Returns:
        (centered_image, centered_mask) both of shape output_size.
    """
    SCALE_FILL = 0.88          # tree should occupy 88 % of the output edge
    ow, oh = output_size
    h,  w  = image.shape[:2]

    # Guard against empty image (e.g. from a degenerate mask crop)
    if h == 0 or w == 0:
        canvas_img  = np.full((oh, ow, 3), background_value, dtype=np.uint8)
        canvas_mask = np.zeros((oh, ow), dtype=np.uint8)
        return canvas_img, canvas_mask

    scale = (SCALE_FILL * min(ow, oh)) / max(h, w, 1)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    img_r  = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask,  (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas_img  = np.full((oh, ow, 3), background_value, dtype=np.uint8)
    canvas_mask = np.zeros((oh, ow), dtype=np.uint8)

    pad_top  = (oh - nh) // 2
    pad_left = (ow - nw) // 2
    canvas_img[pad_top:pad_top + nh, pad_left:pad_left + nw]   = img_r
    canvas_mask[pad_top:pad_top + nh, pad_left:pad_left + nw]  = mask_r

    return canvas_img, canvas_mask


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 – Augment Image (optional)
# ═══════════════════════════════════════════════════════════════════════════════

def augment_image(
    image: RGBImage,
    mask:  Optional[Mask] = None,
    flip_horizontal: bool = True,
    flip_vertical:   bool = False,
    rotate_range:    Tuple[float, float] = (-10, 10),
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    seed: Optional[int] = None,
) -> Tuple[RGBImage, Optional[Mask]]:
    """
    Apply random data-augmentation transforms.

    All transforms are applied to both the image and its mask (if given)
    to maintain correspondence.

    Args:
        image:            RGB uint8 image.
        mask:             Optional binary mask.
        flip_horizontal:  50 % chance of horizontal flip.
        flip_vertical:    50 % chance of vertical flip.
        rotate_range:     (min_deg, max_deg) range for random rotation.
        brightness_range: (min_factor, max_factor) multiplicative brightness.
        seed:             Optional random seed for reproducibility.

    Returns:
        (augmented_image, augmented_mask)
    """
    rng = random.Random(seed)

    out_img  = image.copy()
    out_mask = mask.copy() if mask is not None else None

    h, w = out_img.shape[:2]

    # ── Horizontal flip ───────────────────────────────────────────────────────
    if flip_horizontal and rng.random() < 0.5:
        out_img = cv2.flip(out_img, 1)
        if out_mask is not None:
            out_mask = cv2.flip(out_mask, 1)

    # ── Vertical flip (rare, but possible for top-down / aerial shots) ────────
    if flip_vertical and rng.random() < 0.5:
        out_img = cv2.flip(out_img, 0)
        if out_mask is not None:
            out_mask = cv2.flip(out_mask, 0)

    # ── Random rotation ───────────────────────────────────────────────────────
    angle = rng.uniform(*rotate_range)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out_img = cv2.warpAffine(
        out_img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if out_mask is not None:
        out_mask = cv2.warpAffine(
            out_mask, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # ── Brightness jitter ─────────────────────────────────────────────────────
    factor = rng.uniform(*brightness_range)
    out_img = np.clip(out_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return out_img, out_mask
