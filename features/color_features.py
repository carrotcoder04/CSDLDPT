"""
color_features.py  [v2.1 – With Dominant Colors]
--------------------------------------------------
Trích rút đặc trưng màu sắc từ ảnh cây (18 chiều).

Vector màu (18 chiều):
    - hue_hist_0..5   : Histogram sắc độ (6 bins)
    - h_mean          : Circular mean Hue
    - s_mean, v_mean  : Mean S, V
    - s_std,  v_std   : Std S, V
    - dom_r1..dom_b2  : 2 màu chủ đạo (K=2, HSV, chuẩn hóa [0,1])
    - green_ratio     : Tỷ lệ pixel xanh lá

Tổng: 18 chiều.
"""

import cv2
import numpy as np
from typing import Optional

from features.mask_utils import create_tree_mask

# ─────────────────────────────────────────────
HUE_BINS = 6
N_DOMINANT = 2           # K=2 dominant colors


def extract_hue_histogram(image_bgr: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Hue Histogram 6 bins, L1-normalized (sum=1)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], mask, [HUE_BINS], [0, 180])
    total = hist.sum()
    if total > 0:
        hist = hist / total      # L1 normalize (sum=1) – better for Chi-square
    return hist.flatten().astype(np.float32)


def extract_hsv_statistics(image_bgr: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> dict:
    """Circular mean(H), mean(S,V), std(S,V)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == 255] if mask is not None else hsv.reshape(-1, 3)

    if len(pixels) == 0:
        return {"h_mean": 0.0, "s_mean": 0.0, "v_mean": 0.0,
                "s_std": 0.0, "v_std": 0.0}

    h = pixels[:, 0].astype(np.float64)
    s = pixels[:, 1].astype(np.float64)
    v = pixels[:, 2].astype(np.float64)

    angles = np.deg2rad(h * 2.0)
    mean_c = np.mean(np.exp(1j * angles))
    h_mean = float(np.degrees(np.angle(mean_c)) % 360.0) / 2.0

    return {
        "h_mean": h_mean,
        "s_mean": float(np.mean(s)),
        "v_mean": float(np.mean(v)),
        "s_std":  float(np.std(s)),
        "v_std":  float(np.std(v)),
    }


def extract_dominant_colors(image_bgr: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> dict:
    """
    K=2 dominant colors in HSV space, normalized to [0,1].
    Sử dụng phương pháp tìm đỉnh trên 3D Histogram (THUẦN TÚY KHÔNG DÙNG MACHINE LEARNING).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Lượng tử hóa không gian HSV thành các bins (VD: 16 H, 8 S, 8 V)
    h_bins, s_bins, v_bins = 16, 8, 8
    hist = cv2.calcHist([hsv], [0, 1, 2], mask,
                        [h_bins, s_bins, v_bins],
                        [0, 180, 0, 256, 0, 256])
    
    # Làm phẳng histogram và tìm index của các giá trị lớn nhất
    hist_flat = hist.flatten()
    
    # Lấy ra N_DOMINANT bins có số lượng pixel cao nhất
    top_indices = np.argsort(hist_flat)[-N_DOMINANT:][::-1]
    
    result = {}
    for i in range(N_DOMINANT):
        if i < len(top_indices) and hist_flat[top_indices[i]] > 0:
            idx = top_indices[i]
            # Giải mã index 1D trở lại 3D (h, s, v)
            h_idx = idx // (s_bins * v_bins)
            s_idx = (idx % (s_bins * v_bins)) // v_bins
            v_idx = idx % v_bins
            
            # Tính giá trị ở tâm của mỗi bin
            h_val = (h_idx + 0.5) * (180.0 / h_bins)
            s_val = (s_idx + 0.5) * (256.0 / s_bins)
            v_val = (v_idx + 0.5) * (256.0 / v_bins)
            
            result[f"dom_h{i+1}"] = float(h_val) / 180.0
            result[f"dom_s{i+1}"] = float(s_val) / 255.0
            result[f"dom_v{i+1}"] = float(v_val) / 255.0
        else:
            result[f"dom_h{i+1}"] = 0.0
            result[f"dom_s{i+1}"] = 0.0
            result[f"dom_v{i+1}"] = 0.0

    return result


def extract_green_ratio(image_bgr: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> float:
    """Green pixel ratio in tree region."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_lo = np.array([17, 40, 40], dtype=np.uint8)
    green_hi = np.array([42, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, green_lo, green_hi)

    if mask is not None:
        roi = float(np.sum(mask == 255))
        green_in = float(np.sum((green_mask == 255) & (mask == 255)))
    else:
        h, w = image_bgr.shape[:2]
        roi = float(h * w)
        green_in = float(np.sum(green_mask == 255))

    return green_in / roi if roi > 0 else 0.0


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def extract_color_features(image_bgr: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> dict:
    """
    Full color features (18 dims):
        hue_hist (6) + hsv_stats (5) + dominant_colors (6) + green_ratio (1)
    """
    if mask is None:
        mask = create_tree_mask(image_bgr)

    h_img, w_img = image_bgr.shape[:2]
    use_mask = mask if np.sum(mask == 255) >= h_img * w_img * 0.05 else None

    hue_hist = extract_hue_histogram(image_bgr, use_mask)
    hist_dict = {f"hue_hist_{i}": float(hue_hist[i]) for i in range(HUE_BINS)}

    hsv_stats = extract_hsv_statistics(image_bgr, use_mask)
    dom_colors = extract_dominant_colors(image_bgr, use_mask)
    gr = extract_green_ratio(image_bgr, use_mask)

    return {**hist_dict, **hsv_stats, **dom_colors, "green_ratio": gr}


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "tree.jpg")
    if img is None:
        print("[ERROR] Khong the doc anh.")
        sys.exit(1)
    feats = extract_color_features(img)
    print(f"=== DAC TRUNG MAU SAC ({len(feats)} chieu) ===")
    for k, v in feats.items():
        print(f"  {k:<20}: {v:.6f}")
