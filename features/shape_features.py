"""
shape_features.py
-----------------
Trích rút đặc trưng hình thái cây (12 chiều).

Vector hình thái (12 chiều):
    - area_ratio, aspect_ratio, centroid_x_norm, centroid_y_norm
    - crown_ratio, extent_ratio
    - hu_0..hu_3: 4 Hu Moments đầu, log-scaled
    - solidity, symmetry
"""

import cv2
import numpy as np
from typing import Optional

from features.mask_utils import create_tree_mask

# ─────────────────────────────────────────────
#  Hằng số cấu hình
# ─────────────────────────────────────────────
HU_COUNT = 4
MIN_CONTOUR_AREA = 200


def _largest_contour(mask: np.ndarray):
    """Trả về contour lớn nhất (cây chính) hoặc None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    return max(valid, key=cv2.contourArea) if valid else None


def extract_geometry_features(contour, mask: np.ndarray) -> dict:
    """
    Tính 4 đặc trưng hình học từ contour và mask:
        aspect_ratio : W/H của bounding box.
        solidity     : Area / ConvexHullArea – tán dày đặc (gần 1) vs thưa (< 0.7).
        extent_ratio : Area / BoundingBoxArea – mức độ lấp đầy bounding box.
        crown_ratio  : Tỷ lệ pixel vùng cây thuộc nửa trên ảnh [0, 1].

    Args:
        contour: Contour lớn nhất (cv2.findContours).
        mask:    Mặt nạ nhị phân vùng cây (0/255).

    Returns:
        dict: {aspect_ratio, solidity, extent_ratio, crown_ratio}
    """
    h_img, w_img = mask.shape[:2]

    # Bounding box
    _, _, bw, bh = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    area_ratio = area / float(h_img * w_img) if h_img > 0 and w_img > 0 else 0.0
    bbox_area = float(bw * bh)
    aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0
    extent_ratio = area / bbox_area if bbox_area > 0 else 0.0

    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        centroid_x_norm = float(moments["m10"] / moments["m00"]) / float(w_img)
        centroid_y_norm = float(moments["m01"] / moments["m00"]) / float(h_img)
    else:
        centroid_x_norm = 0.5
        centroid_y_norm = 0.5

    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Crown ratio – tỷ lệ pixel vùng trên (tán)
    h = mask.shape[0]
    mid = h // 2
    top_px = float(np.sum(mask[:mid] == 255))
    total_px = float(np.sum(mask == 255))
    crown_ratio = top_px / total_px if total_px > 0 else 0.5

    return {
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "centroid_x_norm": centroid_x_norm,
        "centroid_y_norm": centroid_y_norm,
        "solidity": solidity,
        "extent_ratio": extent_ratio,
        "crown_ratio": crown_ratio,
    }


def extract_hu_moments(contour) -> dict:
    """
    Tính 3 Hu Moments đầu tiên (log-scaled).

    Hu Moments bất biến với tịnh tiến, tỷ lệ, và xoay.
    Log-transform: val = -sign(hu) × log10(|hu|) để tránh giá trị cực nhỏ.

    Args:
        contour: Contour cây.

    Returns:
        dict: {hu_0, hu_1, hu_2}
    """
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()

    result = {}
    for i in range(HU_COUNT):
        val = float(hu[i])
        if val != 0.0:
            val = -np.copysign(1.0, val) * np.log10(abs(val) + 1e-12)
        result[f"hu_{i}"] = val
    return result


def _empty_shape() -> dict:
    """Vector hình thái rỗng khi không tìm được contour."""
    d = {
        "area_ratio": 0.0,
        "aspect_ratio": 1.0,
        "centroid_x_norm": 0.5,
        "centroid_y_norm": 0.5,
        "crown_ratio": 0.5,
        "extent_ratio": 0.0,
        "solidity": 0.0,
        "symmetry": 0.0,
    }
    for i in range(HU_COUNT):
        d[f"hu_{i}"] = 0.0
    return d


def extract_symmetry(mask: np.ndarray) -> float:
    """
    Tính đối xứng trái-phải của mask theo bounding box đối tượng.
    Giá trị càng gần 1 nghĩa là cây càng đối xứng theo trục dọc.
    """
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return 0.0

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = (mask[y1:y2, x1:x2] == 255).astype(np.uint8)
    if crop.size == 0:
        return 0.0

    flipped = np.fliplr(crop)
    common_w = min(crop.shape[1], flipped.shape[1])
    crop = crop[:, :common_w]
    flipped = flipped[:, :common_w]
    union = np.logical_or(crop, flipped).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(crop, flipped).sum()
    return float(inter / union)


# ─────────────────────────────────────────────
#  Hàm tổng hợp (public API)
# ─────────────────────────────────────────────

def extract_shape_features(image_bgr: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> dict:
    """
    Trích rút toàn bộ đặc trưng hình thái cây (12 chiều).

    Args:
        image_bgr: Ảnh BGR (H×W×3).
        mask:      Mặt nạ vùng cây 0/255. None → tự tính.

    Returns:
        dict (7 khóa):
            aspect_ratio  – Tỷ lệ rộng/cao bounding box
            solidity      – Độ đặc của tán [0, 1]
            extent_ratio  – Mức lấp đầy bounding box [0, 1]
            crown_ratio   – Tỷ lệ pixel nửa trên [0, 1]
            hu_0, hu_1, hu_2 – Hu Moments (log-scaled)
    """
    if mask is None:
        mask = create_tree_mask(image_bgr)

    contour = _largest_contour(mask)
    if contour is None:
        return _empty_shape()

    geom = extract_geometry_features(contour, mask)
    hu = extract_hu_moments(contour)
    symmetry = extract_symmetry(mask)
    return {**geom, **hu, "symmetry": symmetry}


# ─────────────────────────────────────────────
#  CLI thử nghiệm
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "tree.jpg")
    if img is None:
        print("[ERROR] Khong the doc anh.")
        sys.exit(1)
    feats = extract_shape_features(img)
    print("=== DAC TRUNG HINH THAI (7 chieu) ===")
    for k, v in feats.items():
        print(f"  {k:<20}: {v:.6f}")
    print(f"Tong: {len(feats)} chieu")
