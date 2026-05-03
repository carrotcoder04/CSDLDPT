"""
canopy_features.py
------------------
Module trích rút đặc trưng tán cây (Canopy Features) từ ảnh cây.

Đây là module MỚI thay thế vein_features.py (gân lá).
Tán cây có các đặc trưng cấu trúc riêng biệt không tồn tại ở lá đơn lẻ.

Các đặc trưng bao gồm:
    1. Phân bố pixel theo chiều dọc (Vertical Profile):
        - Phân bố tán theo từng dải ngang → mô tả "silhouette" cây.
        - peak_row_norm   : Hàng có nhiều pixel nhất (vị trí tán dày đặc nhất)
        - top25_ratio     : Tỷ lệ pixel trong 25% trên cùng (tán phía trên)
        - bottom25_ratio  : Tỷ lệ pixel trong 25% dưới cùng (thân/rễ)

    2. Độ phức tạp đường viền tán (Canopy Contour Complexity):
        - contour_complexity  : Perimeter / sqrt(Area)
        - convexity           : HullPerimeter / ContourPerimeter

    3. Phân bố chiều ngang theo từng vùng (Horizontal Distribution):
        - width_mean      : Độ rộng tán trung bình theo từng hàng (chuẩn hóa)
        - width_std       : Độ lệch chuẩn độ rộng tán (hình nón vs hình tròn)
        - max_width_norm  : Độ rộng tối đa tán (chuẩn hóa)

    4. Số vùng phân tán (Connected Components):
        - n_components    : Số vùng cây rời rạc (cây chụp đơn vs nhiều cây)

Tài liệu tham khảo: Nhóm 6 - Báo cáo ĐPT - Hệ CSDL Đa Phương Tiện
"""

import cv2
import numpy as np
from typing import Optional

from features.mask_utils import create_tree_mask


# ─────────────────────────────────────────────
#  Hằng số cấu hình
# ─────────────────────────────────────────────
MIN_COMPONENT_AREA = 500   # Diện tích tối thiểu của một vùng cây (loại nhiễu)


def extract_vertical_profile(mask: np.ndarray) -> dict:
    """
    Tính phân bố pixel theo chiều dọc của tán cây.

    Chia ảnh thành các dải ngang mỏng → tính số pixel trong mỗi dải.
    Kết quả mô tả "silhouette" cây theo chiều dọc.

    Args:
        mask: Mặt nạ nhị phân vùng cây (0/255).

    Returns:
        dict:
            - peak_row_norm   : Hàng có pixel dày đặc nhất, chuẩn hóa [0, 1]
            - top25_ratio     : Tỷ lệ pixel ở 25% trên [0, 1]
            - bottom25_ratio  : Tỷ lệ pixel ở 25% dưới [0, 1]
    """
    h = mask.shape[0]
    row_counts = np.sum(mask == 255, axis=1).astype(np.float32)  # shape (H,)
    total = float(np.sum(row_counts))

    if total == 0:
        return {
            "peak_row_norm": 0.5,
            "top25_ratio": 0.25,
            "bottom25_ratio": 0.25,
        }

    peak_row = int(np.argmax(row_counts))
    peak_row_norm = float(peak_row) / float(h)

    q1 = h // 4
    q3 = 3 * h // 4

    top25_ratio    = float(np.sum(row_counts[:q1])) / total
    bottom25_ratio = float(np.sum(row_counts[q3:])) / total

    return {
        "peak_row_norm":    peak_row_norm,
        "top25_ratio":      top25_ratio,
        "bottom25_ratio":   bottom25_ratio,
    }


def extract_contour_complexity(mask: np.ndarray) -> dict:
    """
    Tính độ phức tạp đường viền tán cây.

    - contour_complexity : Perimeter / sqrt(Area) – viền phức tạp → cao.
    - convexity          : HullPerimeter / ContourPerimeter – tán lồi → gần 1.

    Args:
        mask: Mặt nạ nhị phân vùng cây (0/255).

    Returns:
        dict: contour_complexity, convexity
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"contour_complexity": 0.0, "convexity": 1.0}

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))

    if area <= 0:
        return {"contour_complexity": 0.0, "convexity": 1.0}

    complexity = perimeter / np.sqrt(area)

    hull = cv2.convexHull(contour)
    hull_perimeter = float(cv2.arcLength(hull, closed=True))
    convexity = hull_perimeter / perimeter if perimeter > 0 else 1.0

    return {
        "contour_complexity": float(complexity),
        "convexity": float(convexity),
    }


def extract_horizontal_distribution(mask: np.ndarray) -> dict:
    """
    Tính phân bố chiều rộng tán theo từng hàng (Horizontal Distribution).

    Với mỗi hàng, đếm số pixel liên tục → đo "độ rộng tán".
    Thống kê: mean, std, max → mô tả hình dạng tán (hình nón, tròn, trải rộng).

    Args:
        mask: Mặt nạ nhị phân vùng cây (0/255).

    Returns:
        dict: width_mean, width_std, max_width_norm
    """
    h, w = mask.shape
    row_widths = np.sum(mask == 255, axis=1).astype(np.float32)

    # Chỉ xét hàng có ít nhất 1 pixel
    active = row_widths[row_widths > 0]
    if len(active) == 0:
        return {"width_mean": 0.0, "width_std": 0.0, "max_width_norm": 0.0}

    # Chuẩn hóa theo chiều rộng ảnh
    active_norm = active / float(w)
    return {
        "width_mean":     float(np.mean(active_norm)),
        "width_std":      float(np.std(active_norm)),
        "max_width_norm": float(np.max(active_norm)),
    }


def extract_connected_components(mask: np.ndarray) -> int:
    """
    Đếm số vùng cây rời rạc (Connected Components) trong mask.

    n_components = 1 → ảnh chứa 1 cây duy nhất.
    n_components > 1 → nhiều cây, tán bị che khuất, hoặc cây có tán thưa.

    Args:
        mask: Mặt nạ nhị phân vùng cây (0/255).

    Returns:
        int: Số thành phần liên thông đáng kể (lọc nhiễu nhỏ).
    """
    # Đóng kín khoảng hở nhỏ trước khi đếm
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    # stats[:, cv2.CC_STAT_AREA] chứa diện tích từng thành phần
    # Bỏ qua label 0 (nền)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    significant = int(np.sum(component_areas >= MIN_COMPONENT_AREA))
    return max(significant, 0)


def extract_canopy_features(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Hàm chính: trích rút toàn bộ đặc trưng tán cây.

    Args:
        image_bgr: Ảnh BGR đầu vào (numpy array H x W x 3).
        mask:      Mặt nạ vùng cây (0/255). Nếu None, sẽ tự tính.

    Returns:
        dict chứa tất cả đặc trưng tán cây:
            - peak_row_norm      : Vị trí tán dày nhất theo chiều dọc [0, 1]
            - top25_ratio        : Tỷ lệ pixel trong 25% trên [0, 1]
            - bottom25_ratio     : Tỷ lệ pixel trong 25% dưới [0, 1]
            - contour_complexity : Perimeter / sqrt(Area)
            - convexity          : HullPerimeter / ContourPerimeter [0, 1]
            - width_mean         : Độ rộng tán trung bình (chuẩn hóa) [0, 1]
            - width_std          : Độ lệch chuẩn độ rộng tán [0, 1]
            - max_width_norm     : Độ rộng tán tối đa (chuẩn hóa) [0, 1]
            - n_components       : Số vùng cây rời rạc
    """
    if mask is None:
        mask = create_tree_mask(image_bgr)

    # Nếu mask quá rỗng, dùng toàn ảnh như mask mặc định
    h, w = image_bgr.shape[:2]
    if np.sum(mask == 255) < h * w * 0.02:
        mask = np.full((h, w), 255, dtype=np.uint8)

    # ── 1. Vertical Profile ───────────────────────────────
    vert = extract_vertical_profile(mask)

    # ── 2. Contour Complexity ─────────────────────────────
    compl = extract_contour_complexity(mask)

    # ── 3. Horizontal Distribution ───────────────────────
    horiz = extract_horizontal_distribution(mask)

    # ── 4. Connected Components ───────────────────────────
    n_comp = extract_connected_components(mask)

    return {
        **vert,
        **compl,
        **horiz,
        "n_components": float(n_comp),
    }


# ─────────────────────────────────────────────
#  Chạy thử nghiệm
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "tree.jpg"
    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] Khong the doc anh: {image_path}")
        sys.exit(1)

    features = extract_canopy_features(img)
    print("\n=== DAC TRUNG TAN CAY ===")
    for key, val in features.items():
        print(f"  {key:<24} : {val:.6f}")
    print(f"\nTong so dac trung tan cay: {len(features)}")
