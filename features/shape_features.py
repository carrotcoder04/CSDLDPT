"""
shape_features.py  [v2 – Redesigned]
--------------------------------------
Trích rút đặc trưng hình thái cây – phiên bản tối giản (7 chiều).

Vector hình thái (7 chiều):
    - aspect_ratio  : Tỷ lệ W/H bounding box → phân biệt cây cao thẳng vs cây bụi
    - solidity      : Area / ConvexHullArea    → tán dày đặc vs tán thưa
    - extent_ratio  : Area / BoundingBoxArea   → hình dạng tổng quát
    - crown_ratio   : Tỷ lệ pixel nửa trên    → cây có tán cao hay tán thấp
    - hu_0, hu_1, hu_2 : 3 Hu Moments đầu (log-scaled, bất biến tịnh tiến/tỷ lệ/xoay)

Lý do thiết kế:
    - Loại bỏ centroid_x_norm: cây luôn gần trung tâm ảnh → không discriminative.
    - Loại bỏ centroid_y_norm: tương quan cao với crown_ratio (VIF > 5).
    - Loại bỏ symmetry: chi phí tính toán cao, giá trị phân tán không ổn định.
    - Loại bỏ area_ratio: phụ thuộc vào cách chụp ảnh (zoom), không phản ánh loài cây.
    - Hu Moments: giữ hu_0..hu_2 (3 giá trị đầu ổn định nhất); hu_3..hu_6
      thường rất nhỏ và nhạy cảm với nhiễu hơn.

Tổng: 7 chiều.
"""

import cv2
import numpy as np
from typing import Optional

from features.mask_utils import create_tree_mask

# ─────────────────────────────────────────────
#  Hằng số cấu hình
# ─────────────────────────────────────────────
HU_COUNT = 3          # Số Hu Moments sử dụng (3 ổn định nhất trong 7)
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
    # Bounding box
    _, _, bw, bh = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    bbox_area = float(bw * bh)
    aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0
    extent_ratio = area / bbox_area if bbox_area > 0 else 0.0

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
        "aspect_ratio": aspect_ratio,
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
    d = {"aspect_ratio": 1.0, "solidity": 0.0, "extent_ratio": 0.0, "crown_ratio": 0.5}
    for i in range(HU_COUNT):
        d[f"hu_{i}"] = 0.0
    return d


# ─────────────────────────────────────────────
#  Hàm tổng hợp (public API)
# ─────────────────────────────────────────────

def extract_shape_features(image_bgr: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> dict:
    """
    Trích rút toàn bộ đặc trưng hình thái cây (7 chiều).

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
    return {**geom, **hu}


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
