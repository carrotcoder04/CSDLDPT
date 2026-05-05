"""
canopy_features.py  [v2 – Redesigned]
---------------------------------------
Trích rút đặc trưng tán cây – phiên bản tối giản (5 chiều).

Vector tán cây (5 chiều):
    - peak_row_norm     : Vị trí hàng có mật độ pixel cao nhất [0, 1]
    - top25_ratio       : Tỷ lệ pixel trong 25% trên cùng [0, 1]
    - contour_complexity: Perimeter / sqrt(Area) – viền phức tạp vs đơn giản
    - width_mean        : Độ rộng tán trung bình (chuẩn hóa theo chiều rộng ảnh)
    - width_std         : Độ lệch chuẩn độ rộng tán – phân biệt hình nón vs hình tròn

Lý do thiết kế:
    - Loại bỏ bottom25_ratio: tương quan cao với top25_ratio (r = -0.81).
      top25_ratio + peak_row_norm đã đại diện đầy đủ phân bố dọc.
    - Loại bỏ convexity: tương quan cao với solidity trong shape_features (r > 0.75),
      dẫn đến redundancy giữa hai nhóm đặc trưng.
    - Loại bỏ max_width_norm: tương quan cao với width_mean (r ≈ 0.90).
    - Loại bỏ n_components: không ổn định, phụ thuộc nhiều vào chất lượng mask.
      Giá trị thường = 1 cho hầu hết ảnh cây đơn.

Tổng: 5 chiều.
"""

import cv2
import numpy as np
from typing import Optional

from features.mask_utils import create_tree_mask


def extract_vertical_distribution(mask: np.ndarray) -> dict:
    """
    Tính 2 đặc trưng phân bố pixel theo chiều dọc.

    - peak_row_norm : Hàng có pixel nhiều nhất (vị trí tán dày nhất), chuẩn hóa [0, 1].
                      Giá trị thấp → tán tập trung ở phần trên ảnh (cây cao).
    - top25_ratio   : Tỷ lệ pixel trong 25% trên cùng [0, 1].
                      Cao → tán phát triển mạnh ở phần ngọn.

    Args:
        mask: Mặt nạ nhị phân vùng cây (0/255, H×W).

    Returns:
        dict: {peak_row_norm, top25_ratio}
    """
    h = mask.shape[0]
    row_counts = np.sum(mask == 255, axis=1).astype(np.float32)
    total = float(row_counts.sum())

    if total == 0:
        return {"peak_row_norm": 0.5, "top25_ratio": 0.25}

    peak_row = int(np.argmax(row_counts))
    peak_row_norm = float(peak_row) / float(h)
    top25_ratio = float(row_counts[:h // 4].sum()) / total

    return {
        "peak_row_norm": peak_row_norm,
        "top25_ratio":   top25_ratio,
    }


def extract_contour_complexity(mask: np.ndarray) -> float:
    """
    Tính độ phức tạp đường viền tán cây: Perimeter / sqrt(Area).

    Chỉ số Polsby–Popper (dạng nghịch đảo): giá trị cao → viền phức tạp, răng cưa,
    như tán cây lá kim. Giá trị thấp → tán tròn trơn, như cây bóng mát.

    Công thức: complexity = P / √A
        P: chu vi contour lớn nhất.
        A: diện tích contour.

    Args:
        mask: Mặt nạ nhị phân vùng cây (0/255).

    Returns:
        float: Độ phức tạp viền tán cây (≥ 0, thường trong khoảng [4, 30]).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return 0.0

    perimeter = float(cv2.arcLength(contour, closed=True))
    return perimeter / np.sqrt(area)


def extract_horizontal_width(mask: np.ndarray) -> dict:
    """
    Tính 2 đặc trưng phân bố chiều rộng tán theo từng hàng.

    Mỗi hàng pixel: đếm số pixel thuộc vùng cây → đo "độ rộng tán" tại hàng đó.
    Sau đó chuẩn hóa theo chiều rộng ảnh.

    - width_mean : Độ rộng tán trung bình [0, 1].
                   Cao → cây tán rộng (cây bóng mát).
    - width_std  : Độ lệch chuẩn độ rộng tán [0, 1].
                   Cao → hình dạng thay đổi nhiều theo chiều dọc (hình nón/tam giác).
                   Thấp → hình dạng đều (hình cầu/trụ).

    Args:
        mask: Mặt nạ nhị phân vùng cây (0/255).

    Returns:
        dict: {width_mean, width_std}
    """
    h, w = mask.shape
    row_widths = np.sum(mask == 255, axis=1).astype(np.float32)
    active = row_widths[row_widths > 0]

    if len(active) == 0:
        return {"width_mean": 0.0, "width_std": 0.0}

    active_norm = active / float(w)
    return {
        "width_mean": float(np.mean(active_norm)),
        "width_std":  float(np.std(active_norm)),
    }


# ─────────────────────────────────────────────
#  Hàm tổng hợp (public API)
# ─────────────────────────────────────────────

def extract_canopy_features(image_bgr: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> dict:
    """
    Trích rút toàn bộ đặc trưng tán cây (5 chiều).

    Args:
        image_bgr: Ảnh BGR (H×W×3).
        mask:      Mặt nạ vùng cây 0/255. None → tự tính.

    Returns:
        dict (5 khóa):
            peak_row_norm      – Vị trí tán dày nhất (dọc) [0, 1]
            top25_ratio        – Tỷ lệ pixel trong 25% trên [0, 1]
            contour_complexity – Perimeter / sqrt(Area)
            width_mean         – Độ rộng tán trung bình (chuẩn hóa)
            width_std          – Độ lệch chuẩn độ rộng tán
    """
    if mask is None:
        mask = create_tree_mask(image_bgr)

    # Fallback nếu mask quá rỗng
    h, w = image_bgr.shape[:2]
    if np.sum(mask == 255) < h * w * 0.05:
        mask = np.full((h, w), 255, dtype=np.uint8)

    vert = extract_vertical_distribution(mask)
    complexity = extract_contour_complexity(mask)
    horiz = extract_horizontal_width(mask)

    return {
        **vert,
        "contour_complexity": complexity,
        **horiz,
    }


# ─────────────────────────────────────────────
#  CLI thử nghiệm
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "tree.jpg")
    if img is None:
        print("[ERROR] Khong the doc anh.")
        sys.exit(1)
    feats = extract_canopy_features(img)
    print("=== DAC TRUNG TAN CAY (5 chieu) ===")
    for k, v in feats.items():
        print(f"  {k:<24}: {v:.6f}")
    print(f"Tong: {len(feats)} chieu")
