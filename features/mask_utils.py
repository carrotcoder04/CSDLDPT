"""
mask_utils.py  [v2.1 – Improved]
----------------------------------
Tạo mặt nạ nhị phân (tree mask) bằng GrabCut + multi-method fusion.

Cải tiến so với v1 (Otsu + flood-fill):
    - GrabCut: semi-automatic foreground extraction, tốt hơn Otsu trên nền phức tạp.
    - Multi-method fusion: kết hợp Otsu, GrabCut, và HSV green mask → vote.
    - Center-biased prior: cây thường ở trung tâm ảnh → ưu tiên vùng giữa.

Kết quả: mask chính xác hơn → mọi feature module đều được hưởng lợi.
"""

import cv2
import numpy as np


def create_tree_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Tạo mask vùng cây bằng multi-method fusion.

    Pipeline:
        1. GrabCut (3 iterations) với initial rect = vùng giữa 80% ảnh.
        2. Otsu + flood-fill (phương pháp cũ) làm backup.
        3. HSV color mask: vùng xanh lá (H ∈ [20°, 90°], S > 30) = likely tree.
        4. Fusion: majority vote (≥ 2/3 methods agree) → final mask.
        5. Morphology cleanup.

    Args:
        image_bgr: Ảnh BGR (H×W×3).

    Returns:
        mask: 0/255, pixel cây = 255.
    """
    h, w = image_bgr.shape[:2]

    # ── Method 1: GrabCut ─────────────────────────────────
    mask_gc = _grabcut_mask(image_bgr)

    # ── Method 2: Otsu + flood-fill ───────────────────────
    mask_otsu = _otsu_floodfill_mask(image_bgr)

    # ── Method 3: HSV color mask (green/brown = tree) ─────
    mask_color = _color_mask(image_bgr)

    # ── Fusion: majority vote (≥ 2/3) ────────────────────
    vote = ((mask_gc > 0).astype(np.uint8) +
            (mask_otsu > 0).astype(np.uint8) +
            (mask_color > 0).astype(np.uint8))
    fused = np.where(vote >= 2, 255, 0).astype(np.uint8)

    # ── Fallback: nếu fusion quá rỗng, dùng GrabCut alone
    if np.sum(fused == 255) < h * w * 0.05:
        fused = mask_gc.copy()

    # ── Nếu vẫn rỗng, dùng Otsu alone
    if np.sum(fused == 255) < h * w * 0.05:
        fused = mask_otsu.copy()

    # ── Nếu vẫn rỗng, dùng toàn ảnh
    if np.sum(fused == 255) < h * w * 0.01:
        fused = np.full((h, w), 255, dtype=np.uint8)

    # ── Morphology cleanup ────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel)
    fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel)

    return fused


def _grabcut_mask(image_bgr: np.ndarray) -> np.ndarray:
    """GrabCut với initial rect = vùng giữa 80% ảnh."""
    h, w = image_bgr.shape[:2]

    # Init rect: 10% margin on each side (tree usually centered)
    margin_x = max(int(w * 0.10), 1)
    margin_y = max(int(h * 0.10), 1)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model,
                    3, cv2.GC_INIT_WITH_RECT)
        # GC_FGD=1, GC_PR_FGD=3 → foreground
        result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                          255, 0).astype(np.uint8)
    except cv2.error:
        result = np.zeros((h, w), dtype=np.uint8)

    return result


def _otsu_floodfill_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Original Otsu + flood-fill method."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    canvas = otsu.copy()
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for (r, c) in corners:
        cv2.floodFill(canvas, flood_mask, (c, r), 128,
                      loDiff=10, upDiff=10,
                      flags=cv2.FLOODFILL_FIXED_RANGE)

    tree_m = np.where(canvas == 128, 0, 255).astype(np.uint8)

    if np.sum(tree_m == 255) < h * w * 0.01:
        corner_mean = float(np.mean([gray[r, c] for (r, c) in corners]))
        if corner_mean > 127:
            tree_m = cv2.bitwise_not(otsu)
        else:
            tree_m = otsu.copy()

    return tree_m


def _color_mask(image_bgr: np.ndarray) -> np.ndarray:
    """HSV-based mask: green + brown + dark regions = likely tree."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Green foliage: H ∈ [20, 90], S > 30, V > 30
    green = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([90, 255, 255]))

    # Brown bark/trunk: H ∈ [5, 25], S > 30, V ∈ [30, 200]
    brown = cv2.inRange(hsv, np.array([5, 30, 30]), np.array([25, 255, 200]))

    # Dark regions (trunk shadow): V < 80, S < 80
    dark = cv2.inRange(hsv, np.array([0, 0, 10]), np.array([180, 80, 80]))

    combined = green | brown | dark

    # Cleanup small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return combined


# Alias
create_leaf_mask = create_tree_mask
