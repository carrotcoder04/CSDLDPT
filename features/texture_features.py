"""
texture_features.py
-------------------
Module trích rút đặc trưng kết cấu (Texture Features) từ ảnh cây.

Đây là module MỚI thay thế edge_features.py (viền lá) trong bối cảnh ảnh cây.
Kết cấu bề mặt tán cây (mịn/thô/hạt/dạng kim...) là đặc trưng phân biệt quan trọng.

Các đặc trưng bao gồm:
    1. LBP Histogram (Local Binary Pattern) – 10 bins
       Mô tả kết cấu cục bộ: tán mịn (cây lá rộng) vs tán thô (cây lá kim).
    2. GLCM Statistics (Gray-Level Co-occurrence Matrix)
       - contrast   : Độ tương phản cục bộ (cây lá kim → cao)
       - homogeneity: Độ đồng nhất (tán mịn → cao)
       - energy     : Năng lượng kết cấu (kết cấu đều → cao)
       - correlation: Tương quan pixel liền kề
    3. Gradient Statistics (Sobel)
       - grad_mean  : Cường độ cạnh trung bình (tán phức tạp → cao)
       - grad_std   : Độ lệch chuẩn cạnh
    4. Fractal Dimension (Lacunarity proxy)
       - roughness  : Độ nhám tán cây (ước tính bằng std cục bộ)

Tài liệu tham khảo: Nhóm 6 - Báo cáo ĐPT - Hệ CSDL Đa Phương Tiện
"""

import cv2
import numpy as np
from typing import Optional

from features.mask_utils import create_tree_mask


# ─────────────────────────────────────────────
#  Hằng số cấu hình
# ─────────────────────────────────────────────
LBP_BINS = 10          # Số bins cho LBP histogram (uniform LBP: 0–9)
LBP_RADIUS = 1         # Bán kính vòng LBP
LBP_NEIGHBORS = 8      # Số điểm lấy mẫu
GLCM_DISTANCES = [1]   # Khoảng cách GLCM (pixel)
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135°


def _compute_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Tính LBP (Local Binary Pattern) thủ công cho 8 điểm lân cận bán kính 1.

    Với mỗi pixel trung tâm, so sánh với 8 pixel lân cận theo chiều kim đồng hồ.
    Nếu lân cận >= trung tâm → bit 1, ngược lại → bit 0.
    Kết quả là mã nhị phân 8-bit (0–255).

    Args:
        gray: Ảnh grayscale.

    Returns:
        lbp: Ma trận LBP cùng kích thước với gray (dtype uint8).
    """
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    # 8 điểm lân cận: bắt đầu từ vị trí trên-phải, theo chiều kim đồng hồ
    offsets = [(-1, 1), (-1, 0), (-1, -1), (0, -1),
               (1, -1),  (1, 0),  (1, 1),  (0, 1)]

    center = gray[1:-1, 1:-1].astype(np.int16)

    for bit, (dr, dc) in enumerate(offsets):
        neighbor = gray[1 + dr: h - 1 + dr, 1 + dc: w - 1 + dc].astype(np.int16)
        lbp[1:-1, 1:-1] |= ((neighbor >= center).astype(np.uint8) << bit)

    return lbp


def extract_lbp_histogram(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Tính LBP Histogram từ ảnh cây.

    LBP mô tả kết cấu cục bộ mà không cần màu sắc.
    Cây lá rộng: histogram tập trung ở các pattern đồng nhất.
    Cây lá kim: histogram phân tán, nhiều pattern cạnh.

    Args:
        image_bgr: Ảnh BGR đầu vào.
        mask:      Mặt nạ vùng cây (0/255). None = dùng toàn ảnh.

    Returns:
        numpy array shape (10,): LBP histogram chuẩn hóa (10 bins đại diện).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = _compute_lbp(gray)

    if mask is not None:
        # lbp có shape (H, W) nhưng chỉ vùng [1:-1, 1:-1] được tính.
        # Phải dùng lbp[1:-1, 1:-1] để khớp kích thước với mask[1:-1, 1:-1].
        lbp_inner = lbp[1:-1, 1:-1]
        mask_inner = mask[1:-1, 1:-1]
        lbp_masked = lbp_inner[mask_inner == 255]
    else:
        lbp_masked = lbp[1:-1, 1:-1].flatten()

    if len(lbp_masked) == 0:
        return np.zeros(LBP_BINS, dtype=np.float32)

    # Gom thành LBP_BINS bins (256 giá trị → 10 bins)
    hist, _ = np.histogram(lbp_masked, bins=LBP_BINS, range=(0, 256))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _compute_glcm(gray: np.ndarray, distance: int = 1, angle: float = 0.0) -> np.ndarray:
    """
    Tính GLCM (Gray-Level Co-occurrence Matrix) với 64 mức xám.

    Args:
        gray:     Ảnh grayscale (0–255).
        distance: Khoảng cách giữa cặp pixel.
        angle:    Góc hướng (radian): 0=0°, π/4=45°, π/2=90°, 3π/4=135°.

    Returns:
        glcm: Ma trận 64×64 chuẩn hóa.
    """
    # Lượng tử hóa về 64 mức
    levels = 64
    g = (gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.int32)
    g = np.clip(g, 0, levels - 1)

    dr = int(round(-distance * np.sin(angle)))
    dc = int(round(distance * np.cos(angle)))

    h, w = g.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)

    r_start = max(0, -dr)
    r_end   = h + min(0, -dr)
    c_start = max(0, -dc)
    c_end   = w + min(0, -dc)

    i_vals = g[r_start:r_end, c_start:c_end]
    j_vals = g[r_start + dr: r_end + dr, c_start + dc: c_end + dc]

    np.add.at(glcm, (i_vals.ravel(), j_vals.ravel()), 1)
    # Đối xứng
    glcm += glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def extract_glcm_features(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """
    Tính 4 thống kê GLCM: contrast, homogeneity, energy, correlation.

    Trung bình trên 4 góc (0°, 45°, 90°, 135°) để bất biến với hướng.

    Args:
        image_bgr: Ảnh BGR đầu vào.
        mask:      Mặt nạ vùng cây. None = dùng toàn ảnh.

    Returns:
        dict: contrast, homogeneity, energy, correlation
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Nếu có mask, chỉ giữ lại vùng cây (đặt nền = giá trị trung bình)
    if mask is not None and np.sum(mask == 255) > 0:
        mean_val = int(np.mean(gray[mask == 255]))
        gray_roi = np.where(mask == 255, gray, mean_val).astype(np.uint8)
    else:
        gray_roi = gray

    levels = 64
    i_idx = np.arange(levels)
    j_idx = np.arange(levels)
    I, J = np.meshgrid(i_idx, j_idx, indexing="ij")

    stats = {"contrast": 0.0, "homogeneity": 0.0, "energy": 0.0, "correlation": 0.0}
    n_angles = len(GLCM_ANGLES)

    for angle in GLCM_ANGLES:
        glcm = _compute_glcm(gray_roi, distance=1, angle=angle)

        stats["contrast"]    += float(np.sum(glcm * (I - J) ** 2))
        stats["homogeneity"] += float(np.sum(glcm / (1.0 + np.abs(I - J) + 1e-9)))
        stats["energy"]      += float(np.sum(glcm ** 2))

        mu_i = float(np.sum(I * glcm))
        mu_j = float(np.sum(J * glcm))
        sig_i = float(np.sqrt(np.sum(glcm * (I - mu_i) ** 2) + 1e-9))
        sig_j = float(np.sqrt(np.sum(glcm * (J - mu_j) ** 2) + 1e-9))
        stats["correlation"] += float(np.sum(glcm * (I - mu_i) * (J - mu_j)) / (sig_i * sig_j))

    for k in stats:
        stats[k] /= n_angles

    return stats


def extract_gradient_features(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """
    Tính thống kê gradient (Sobel) của ảnh cây.

    Gradient cao → viền tán phức tạp, nhiều chi tiết.
    Gradient thấp → tán mịn, ít chi tiết.

    Args:
        image_bgr: Ảnh BGR đầu vào.
        mask:      Mặt nạ vùng cây. None = dùng toàn ảnh.

    Returns:
        dict: grad_mean, grad_std
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    if mask is not None and np.sum(mask == 255) > 0:
        vals = magnitude[mask == 255]
    else:
        vals = magnitude.flatten()

    if len(vals) == 0:
        return {"grad_mean": 0.0, "grad_std": 0.0}

    return {
        "grad_mean": float(np.mean(vals)),
        "grad_std":  float(np.std(vals)),
    }


def extract_roughness(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Tính Độ nhám tán cây (Roughness) bằng Local Standard Deviation.

    Phương pháp: Áp dụng cửa sổ trượt 5×5, tính std cục bộ → lấy trung bình.
    Tán cây dạng lá kim (Cedrus, Salix) → roughness cao hơn tán tròn mịn.

    Args:
        image_bgr: Ảnh BGR đầu vào.
        mask:      Mặt nạ vùng cây. None = dùng toàn ảnh.

    Returns:
        float: Roughness (độ nhám) trung bình.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Tính local mean và local mean of squares bằng boxFilter
    kernel_size = 5
    local_mean = cv2.boxFilter(gray, -1, (kernel_size, kernel_size))
    local_sq_mean = cv2.boxFilter(gray ** 2, -1, (kernel_size, kernel_size))
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)
    local_std = np.sqrt(local_var)

    if mask is not None and np.sum(mask == 255) > 0:
        vals = local_std[mask == 255]
    else:
        vals = local_std.flatten()

    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


def extract_texture_features(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Hàm chính: trích rút toàn bộ đặc trưng kết cấu từ ảnh cây.

    Args:
        image_bgr: Ảnh BGR đầu vào (numpy array H x W x 3).
        mask:      Mặt nạ vùng cây (0/255). Nếu None, sẽ tự tính.

    Returns:
        dict chứa tất cả đặc trưng kết cấu (17 chiều):
            - lbp_0 .. lbp_9   : LBP Histogram (10 bins, chuẩn hóa) → 10 chiều
            - contrast         : GLCM Contrast
            - homogeneity      : GLCM Homogeneity
            - energy           : GLCM Energy
            - correlation      : GLCM Correlation
            - grad_mean        : Cường độ cạnh trung bình (Sobel)
            - grad_std         : Độ lệch chuẩn cạnh
            - roughness        : Độ nhám bề mặt tán
            Tổng: 17 chiều
    """
    if mask is None:
        mask = create_tree_mask(image_bgr)

    h, w = image_bgr.shape[:2]
    use_mask = mask if np.sum(mask == 255) >= h * w * 0.05 else None

    # ── 1. LBP Histogram ──────────────────────────────────
    lbp_hist = extract_lbp_histogram(image_bgr, use_mask)
    lbp_dict = {f"lbp_{i}": float(lbp_hist[i]) for i in range(LBP_BINS)}

    # ── 2. GLCM Statistics ────────────────────────────────
    glcm_stats = extract_glcm_features(image_bgr, use_mask)

    # ── 3. Gradient Statistics ────────────────────────────
    grad_stats = extract_gradient_features(image_bgr, use_mask)

    # ── 4. Roughness ──────────────────────────────────────
    roughness = extract_roughness(image_bgr, use_mask)

    return {
        **lbp_dict,
        **glcm_stats,
        **grad_stats,
        "roughness": roughness,
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

    features = extract_texture_features(img)
    print("\n=== DAC TRUNG KET CAU (CAY) ===")
    for key, val in features.items():
        print(f"  {key:<22} : {val:.6f}")
    print(f"\nTong so dac trung ket cau: {len(features)}")
