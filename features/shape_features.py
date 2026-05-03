"""
shape_features.py
-----------------
Module trích rút đặc trưng hình thái (Shape Features) từ ảnh cây.

Ảnh cây khác hoàn toàn ảnh lá: đối tượng chiếm phần lớn ảnh, có thân cây,
tán rộng, phân nhánh phức tạp. Các đặc trưng tập trung vào:

    1. Hình thái tổng thể tán cây:
        - Crown ratio    : Tỷ lệ tán/thân theo chiều dọc
        - Aspect ratio   : Tỷ lệ rộng/cao của bounding box
        - Extent ratio   : Diện tích cây / diện tích bounding box
        - Solidity       : Diện tích cây / diện tích convex hull (độ dày đặc tán)

    2. Phân bố khối lượng hình học:
        - Centroid Y     : Vị trí trọng tâm theo chiều dọc (chuẩn hóa)
        - Symmetry       : Độ đối xứng trái/phải của tán cây

    3. Đặc trưng Hu Moments (4 giá trị đầu, bất biến với phép biến đổi affine)

Tài liệu tham khảo: Nhóm 6 - Báo cáo ĐPT - Hệ CSDL Đa Phương Tiện
"""

import cv2
import numpy as np
from typing import Optional

from features.mask_utils import create_tree_mask


# ─────────────────────────────────────────────
#  Hằng số cấu hình
# ─────────────────────────────────────────────
HU_MOMENTS_COUNT = 4     # Số Hu Moments sử dụng (tổng 7, lấy 4 đầu)
MIN_CONTOUR_AREA = 200   # Diện tích tối thiểu để lọc nhiễu


def _get_largest_contour(mask: np.ndarray):
    """
    Tìm contour lớn nhất (giả định là viền tán cây chính).

    Args:
        mask: Mặt nạ nhị phân 0/255.

    Returns:
        contour: Contour lớn nhất, hoặc None nếu không tìm thấy.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not valid:
        return None
    return max(valid, key=cv2.contourArea)


def extract_bounding_box_features(contour, image_shape: tuple) -> dict:
    """
    Tính các đặc trưng từ bounding box của cây.

        - aspect_ratio   : W / H của bounding box
        - extent_ratio   : Area / (W × H) – diện tích cây so với bounding box
        - area_ratio     : Area / (img_H × img_W) – cây chiếm bao nhiêu % ảnh

    Args:
        contour:     Contour lớn nhất (cv2.findContours).
        image_shape: Tuple (H, W) ảnh gốc.

    Returns:
        dict: aspect_ratio, extent_ratio, area_ratio
    """
    img_h, img_w = image_shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    bbox_area = float(bw * bh)

    aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0
    extent_ratio = area / bbox_area if bbox_area > 0 else 0.0
    area_ratio = area / (img_h * img_w) if (img_h * img_w) > 0 else 0.0

    return {
        "aspect_ratio": aspect_ratio,
        "extent_ratio": extent_ratio,
        "area_ratio": area_ratio,
    }


def extract_solidity(contour) -> float:
    """
    Tính Độ đặc (Solidity) = Area / Convex Hull Area.

    Solidity cao → tán cây dày đặc, tán hình oval/tròn.
    Solidity thấp → tán thưa, cây có nhiều khoảng trống, cây dạng lá kim.

    Args:
        contour: Contour cây.

    Returns:
        float: Solidity trong khoảng [0, 1].
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return float(area) / float(hull_area)


def extract_centroid_features(mask: np.ndarray) -> dict:
    """
    Tính vị trí trọng tâm (centroid) của vùng cây theo chiều dọc và ngang.

    Ý nghĩa:
        - centroid_y_norm: Trọng tâm theo chiều cao (0 = đỉnh ảnh, 1 = đáy ảnh).
          Cây có tán cao sẽ có centroid_y_norm thấp; cây bụi → cao.
        - centroid_x_norm: Trọng tâm theo chiều ngang (0 = trái, 1 = phải).
          Thường ≈ 0.5 với cây đứng thẳng, lệch nếu cây nghiêng.

    Args:
        mask: Mặt nạ nhị phân vùng cây (0/255).

    Returns:
        dict: centroid_y_norm, centroid_x_norm
    """
    h, w = mask.shape[:2]
    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        return {"centroid_y_norm": 0.5, "centroid_x_norm": 0.5}

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    return {
        "centroid_y_norm": float(cy / h) if h > 0 else 0.5,
        "centroid_x_norm": float(cx / w) if w > 0 else 0.5,
    }


def extract_symmetry(mask: np.ndarray) -> float:
    """
    Tính Độ đối xứng trái/phải (Lateral Symmetry) của tán cây.

    Phương pháp:
        1. Lấy trục đối xứng dọc tại centroid_x.
        2. Flip nửa phải sang trái → so sánh với nửa trái.
        3. Symmetry = IoU(left_half, flipped_right_half).

    Giá trị gần 1.0 → cây đối xứng hoàn hảo (như cây thông, cây thánh).
    Giá trị thấp   → tán không đều, lệch một phía.

    Args:
        mask: Mặt nạ nhị phân vùng cây (0/255).

    Returns:
        float: Symmetry score [0, 1].
    """
    h, w = mask.shape[:2]
    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        return 0.0

    cx = int(moments["m10"] / moments["m00"])
    cx = max(1, min(cx, w - 2))  # Tránh biên

    left_half = mask[:, :cx].astype(np.float32)
    right_half = mask[:, cx:].astype(np.float32)

    # Điều chỉnh để 2 nửa bằng nhau
    min_w = min(left_half.shape[1], right_half.shape[1])
    left_crop = left_half[:, :min_w]
    right_crop = right_half[:, :min_w]
    right_flipped = np.fliplr(right_crop)

    # Intersection over Union
    intersection = float(np.sum((left_crop > 0) & (right_flipped > 0)))
    union = float(np.sum((left_crop > 0) | (right_flipped > 0)))

    return intersection / union if union > 0 else 0.0


def extract_crown_trunk_ratio(mask: np.ndarray) -> float:
    """
    Ước tính tỷ lệ tán/thân cây theo phân bố pixel theo chiều dọc.

    Phương pháp:
        - Chia ảnh làm 4 phần đều nhau theo chiều dọc.
        - Tán cây thường ở nửa trên → tập trung pixel ở top 50%.
        - Thân cây thường ở nửa dưới → pixel ít hơn, hẹp hơn.
        - crown_ratio = pixels_top_half / total_pixels ∈ [0, 1].
          Gần 0.5 nếu phân bố đều; > 0.6 nếu tán rộng ở trên; < 0.4 nếu tán thấp.

    Args:
        mask: Mặt nạ nhị phân vùng cây (0/255).

    Returns:
        float: Tỷ lệ pixel phần trên tán [0, 1].
    """
    h = mask.shape[0]
    mid = h // 2
    top_pixels = float(np.sum(mask[:mid] == 255))
    total_pixels = float(np.sum(mask == 255))

    if total_pixels == 0:
        return 0.5
    return top_pixels / total_pixels


def extract_hu_moments(contour) -> dict:
    """
    Tính 4 Hu Moments đầu tiên – đặc trưng hình dạng bất biến.

    Hu Moments bất biến với tịnh tiến, tỷ lệ và xoay.
    Log-transform để tránh giá trị quá nhỏ.

    Args:
        contour: Contour cây.

    Returns:
        dict: hu_0 .. hu_3 (log-scaled)
    """
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()

    result = {}
    for i in range(HU_MOMENTS_COUNT):
        val = hu[i]
        if val != 0:
            val = -np.copysign(1, val) * np.log10(abs(val))
        result[f"hu_{i}"] = float(val)
    return result


def extract_shape_features(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Hàm chính: trích rút toàn bộ đặc trưng hình thái cây.

    Args:
        image_bgr: Ảnh BGR đầu vào (numpy array H x W x 3).
        mask:      Mặt nạ vùng cây (0/255). Nếu None, sẽ tự tính.

    Returns:
        dict chứa tất cả đặc trưng hình thái:
            - aspect_ratio    : Tỷ lệ W/H bounding box
            - extent_ratio    : Area / BoundingBoxArea
            - area_ratio      : Area / ImageArea [0, 1]
            - solidity        : Area / ConvexHullArea [0, 1]
            - centroid_y_norm : Vị trí trọng tâm dọc [0, 1]
            - centroid_x_norm : Vị trí trọng tâm ngang [0, 1]
            - symmetry        : Độ đối xứng trái/phải [0, 1]
            - crown_ratio     : Tỷ lệ pixel nửa trên (tán) [0, 1]
            - hu_0 .. hu_3    : Hu Moments (log-scaled)
    """
    if mask is None:
        mask = create_tree_mask(image_bgr)

    # Centroid và symmetry tính trên mask (không cần contour)
    centroid_feats = extract_centroid_features(mask)
    symmetry = extract_symmetry(mask)
    crown_ratio = extract_crown_trunk_ratio(mask)

    contour = _get_largest_contour(mask)
    if contour is None:
        return _empty_shape_features()

    bbox_feats = extract_bounding_box_features(contour, image_bgr.shape)
    solidity = extract_solidity(contour)
    hu_dict = extract_hu_moments(contour)

    return {
        **bbox_feats,
        "solidity": solidity,
        **centroid_feats,
        "symmetry": symmetry,
        "crown_ratio": crown_ratio,
        **hu_dict,
    }


def _empty_shape_features() -> dict:
    """Trả về dict đặc trưng hình thái rỗng (tất cả = 0.0)."""
    result = {
        "aspect_ratio": 0.0,
        "extent_ratio": 0.0,
        "area_ratio": 0.0,
        "solidity": 0.0,
        "centroid_y_norm": 0.5,
        "centroid_x_norm": 0.5,
        "symmetry": 0.0,
        "crown_ratio": 0.5,
    }
    for i in range(HU_MOMENTS_COUNT):
        result[f"hu_{i}"] = 0.0
    return result


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

    features = extract_shape_features(img)
    print("\n=== DAC TRUNG HINH THAI CAY ===")
    for key, val in features.items():
        print(f"  {key:<22} : {val:.6f}")
    print(f"\nTong so dac trung hinh thai: {len(features)}")
