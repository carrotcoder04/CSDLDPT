"""
texture_features.py
-------------------
Trích rút đặc trưng kết cấu tán cây (17 chiều).

Vector kết cấu (17 chiều):
    - contrast, correlation, energy, homogeneity: GLCM trung bình 4 hướng
    - grad_mean, grad_std: thống kê biên Sobel
    - lbp_0..lbp_9: LBP Histogram 10 bins
    - roughness: độ nhám cục bộ theo sai khác với ảnh làm mờ
"""

import cv2
import numpy as np
from typing import Optional

from features.mask_utils import create_tree_mask

# ─────────────────────────────────────────────
#  Hằng số cấu hình
# ─────────────────────────────────────────────
LBP_BINS = 10
GLCM_LEVELS = 64    # Số mức xám cho GLCM
GLCM_ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 4 góc để bất biến hướng


def _compute_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Tính LBP (Local Binary Pattern) 8-điểm bán kính 1 thủ công.

    Với mỗi pixel trung tâm, so sánh với 8 pixel hàng xóm theo chiều kim đồng hồ.
    Kết quả: mã nhị phân 8-bit trong khoảng [0, 255].

    Args:
        gray: Ảnh grayscale (H×W).

    Returns:
        lbp: Ma trận LBP (H×W, dtype uint8).
    """
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    # Thứ tự 8 hàng xóm: bắt đầu từ trên-phải, đi theo chiều kim đồng hồ
    offsets = [(-1, 1), (-1, 0), (-1, -1), (0, -1),
               (1, -1),  (1, 0),  (1, 1),  (0, 1)]

    center = gray[1:-1, 1:-1].astype(np.int16)
    for bit, (dr, dc) in enumerate(offsets):
        neighbor = gray[1 + dr: h - 1 + dr, 1 + dc: w - 1 + dc].astype(np.int16)
        lbp[1:-1, 1:-1] |= ((neighbor >= center).astype(np.uint8) << bit)

    return lbp


def extract_lbp_histogram(image_bgr: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Tính LBP Histogram (5 bins) từ ảnh cây.

    5 bins đại diện 5 nhóm kết cấu: rất mịn, mịn, trung bình, thô, rất thô.
    Histogram chuẩn hóa theo tổng số pixel.

    Args:
        image_bgr: Ảnh BGR (H×W×3).
        mask:      Mặt nạ vùng cây 0/255. None = toàn ảnh.

    Returns:
        numpy array (5,): LBP histogram chuẩn hóa.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = _compute_lbp(gray)

    if mask is not None:
        lbp_inner = lbp[1:-1, 1:-1]
        mask_inner = mask[1:-1, 1:-1]
        values = lbp_inner[mask_inner == 255]
    else:
        values = lbp[1:-1, 1:-1].flatten()

    if len(values) == 0:
        return np.zeros(LBP_BINS, dtype=np.float32)

    hist, _ = np.histogram(values, bins=LBP_BINS, range=(0, 256))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _compute_glcm(gray: np.ndarray, distance: int = 1,
                  angle: float = 0.0) -> np.ndarray:
    """
    Tính GLCM với GLCM_LEVELS mức xám cho một góc/khoảng cách cụ thể.

    Args:
        gray:     Ảnh grayscale (H×W, uint8).
        distance: Khoảng cách giữa cặp pixel (pixel).
        angle:    Góc hướng (radian).

    Returns:
        glcm: Ma trận (GLCM_LEVELS × GLCM_LEVELS), chuẩn hóa tổng = 1.
    """
    levels = GLCM_LEVELS
    g = np.clip(
        (gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.int32),
        0, levels - 1
    )
    dr = int(round(-distance * np.sin(angle)))
    dc = int(round(distance * np.cos(angle)))

    h, w = g.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)
    r0, r1 = max(0, -dr), h + min(0, -dr)
    c0, c1 = max(0, -dc), w + min(0, -dc)

    i_vals = g[r0:r1, c0:c1]
    j_vals = g[r0 + dr: r1 + dr, c0 + dc: c1 + dc]
    np.add.at(glcm, (i_vals.ravel(), j_vals.ravel()), 1)
    glcm += glcm.T   # Đối xứng → bất biến hướng
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def extract_glcm_features(image_bgr: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> dict:
    """
    Tính 4 đặc trưng GLCM: contrast, correlation, energy và homogeneity.

    Trung bình trên 4 góc (0°, 45°, 90°, 135°) → bất biến với hướng.

    Định nghĩa:
        contrast    = Σ_{i,j} (i-j)² · p(i,j)
        homogeneity = Σ_{i,j} p(i,j) / (1 + |i-j|)

    Hai chỉ số này bổ trợ nhau:
        - Tán lá kim (cây thông): contrast cao, homogeneity thấp.
        - Tán tròn mịn (cây bóng mát): contrast thấp, homogeneity cao.

    Args:
        image_bgr: Ảnh BGR (H×W×3).
        mask:      Mặt nạ vùng cây 0/255. None = toàn ảnh.

    Returns:
        dict: {contrast, homogeneity}
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if mask is not None and np.sum(mask == 255) > 0:
        mean_val = int(np.mean(gray[mask == 255]))
        gray_roi = np.where(mask == 255, gray, mean_val).astype(np.uint8)
    else:
        gray_roi = gray

    levels = GLCM_LEVELS
    i_idx = np.arange(levels)
    I, J = np.meshgrid(i_idx, i_idx, indexing="ij")
    diff = I - J

    contrast = 0.0
    correlation = 0.0
    energy = 0.0
    homogeneity = 0.0

    for angle in GLCM_ANGLES:
        glcm = _compute_glcm(gray_roi, distance=1, angle=angle)
        contrast    += float(np.sum(glcm * diff ** 2))
        energy      += float(np.sum(glcm ** 2))
        homogeneity += float(np.sum(glcm / (1.0 + np.abs(diff) + 1e-9)))

        px = glcm.sum(axis=1)
        py = glcm.sum(axis=0)
        mux = float(np.sum(i_idx * px))
        muy = float(np.sum(i_idx * py))
        sigx = float(np.sqrt(np.sum(((i_idx - mux) ** 2) * px)))
        sigy = float(np.sqrt(np.sum(((i_idx - muy) ** 2) * py)))
        if sigx > 1e-12 and sigy > 1e-12:
            correlation += float(np.sum((I - mux) * (J - muy) * glcm) / (sigx * sigy))

    n = len(GLCM_ANGLES)
    return {
        "contrast": contrast / n,
        "correlation": correlation / n,
        "energy": energy / n,
        "homogeneity": homogeneity / n,
    }


def extract_gradient_features(image_bgr: np.ndarray,
                              mask: Optional[np.ndarray] = None) -> dict:
    """Tính mean/std của gradient magnitude trong vùng cây."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    values = grad[mask == 255] if mask is not None and np.any(mask == 255) else grad.ravel()
    if len(values) == 0:
        return {"grad_mean": 0.0, "grad_std": 0.0}
    return {
        "grad_mean": float(np.mean(values)),
        "grad_std": float(np.std(values)),
    }


def extract_roughness(image_bgr: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> float:
    """Độ nhám cục bộ: trung bình sai khác tuyệt đối với ảnh Gaussian blur."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    rough = np.abs(gray - smooth) / 255.0
    values = rough[mask == 255] if mask is not None and np.any(mask == 255) else rough.ravel()
    return float(np.mean(values)) if len(values) else 0.0


# ─────────────────────────────────────────────
#  Hàm tổng hợp (public API)
# ─────────────────────────────────────────────

def extract_texture_features(image_bgr: np.ndarray,
                             mask: Optional[np.ndarray] = None) -> dict:
    """
    Trích rút toàn bộ đặc trưng kết cấu từ ảnh cây (17 chiều).

    Args:
        image_bgr: Ảnh BGR (H×W×3).
        mask:      Mặt nạ vùng cây 0/255. None → tự tính.

    Returns:
        dict (7 khóa):
            lbp_0 .. lbp_4  – LBP Histogram 5 bins (chuẩn hóa)
            contrast        – GLCM Contrast (trung bình 4 góc)
            homogeneity     – GLCM Homogeneity (trung bình 4 góc)
    """
    if mask is None:
        mask = create_tree_mask(image_bgr)

    h, w = image_bgr.shape[:2]
    use_mask = mask if np.sum(mask == 255) >= h * w * 0.05 else None

    # 1. LBP Histogram (10 bins)
    lbp_hist = extract_lbp_histogram(image_bgr, use_mask)
    lbp_dict = {f"lbp_{i}": float(lbp_hist[i]) for i in range(LBP_BINS)}

    # 2. GLCM + gradient + roughness
    glcm = extract_glcm_features(image_bgr, use_mask)
    grad = extract_gradient_features(image_bgr, use_mask)
    roughness = extract_roughness(image_bgr, use_mask)

    return {**glcm, **grad, **lbp_dict, "roughness": roughness}


# ─────────────────────────────────────────────
#  CLI thử nghiệm
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "tree.jpg")
    if img is None:
        print("[ERROR] Khong the doc anh.")
        sys.exit(1)
    feats = extract_texture_features(img)
    print("=== DAC TRUNG KET CAU (7 chieu) ===")
    for k, v in feats.items():
        print(f"  {k:<20}: {v:.6f}")
    print(f"Tong: {len(feats)} chieu")
