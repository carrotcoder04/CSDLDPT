"""
color_features.py
-----------------
Module trích rút đặc trưng màu sắc (Color Features) từ ảnh cây.

Các đặc trưng bao gồm:
    1. Hue Histogram (8 bins) – Phân bố sắc độ toàn ảnh (hoặc vùng cây nếu có mask)
    2. Thống kê kênh màu HSV (Mean & Std của H, S, V)
    3. Màu chủ đạo (3 Dominant Colors) bằng thuật toán KMeans
    4. Tỷ lệ màu xanh lá (Green Ratio) – chỉ số quan trọng cho ảnh cây

Lưu ý: Ảnh cây chụp ngoài trời có nền đa dạng (trời xanh, đất, tường).
        Nếu không thể tách nền, trích rút trên toàn ảnh vẫn cho kết quả hữu ích.

Tài liệu tham khảo: Nhóm 6 - Báo cáo ĐPT - Hệ CSDL Đa Phương Tiện
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional

from features.mask_utils import create_tree_mask


# ─────────────────────────────────────────────
#  Hằng số cấu hình
# ─────────────────────────────────────────────
HUE_BINS = 8              # Số khoảng chia histogram kênh Hue
N_DOMINANT_COLORS = 3     # Số màu chủ đạo cần trích xuất
KMEANS_MAX_PIXELS = 2000  # Số pixel tối đa đưa vào KMeans (subsample)


def extract_hue_histogram(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Tính Hue Histogram (8 bins) từ kênh H trong không gian HSV.

    Histogram được chuẩn hóa về khoảng [0, 1] bằng cv2.normalize().

    Args:
        image_bgr: Ảnh BGR đầu vào.
        mask:      Mặt nạ vùng cây (0/255). None = dùng toàn ảnh.

    Returns:
        hist_normalized: numpy array shape (8,) – tỷ lệ pixel mỗi khoảng sắc độ.
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image_hsv], [0], mask, [HUE_BINS], [0, 180])
    hist_normalized = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist_normalized.flatten()


def extract_hsv_statistics(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """
    Tính Mean và Standard Deviation cho từng kênh H, S, V.

    Sử dụng Circular Mean cho kênh H để tránh sai số wrap-around (màu đỏ).

    Args:
        image_bgr: Ảnh BGR đầu vào.
        mask:      Mặt nạ vùng cây (0/255). None = dùng toàn ảnh.

    Returns:
        dict chứa: h_mean, h_std, s_mean, s_std, v_mean, v_std
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if mask is not None:
        pixels = image_hsv[mask == 255]
    else:
        pixels = image_hsv.reshape(-1, 3)

    if len(pixels) == 0:
        return {
            "h_mean": 0.0, "h_std": 0.0,
            "s_mean": 0.0, "s_std": 0.0,
            "v_mean": 0.0, "v_std": 0.0,
        }

    h_channel = pixels[:, 0].astype(np.float32)
    s_channel = pixels[:, 1].astype(np.float32)
    v_channel = pixels[:, 2].astype(np.float32)

    # ── Circular mean cho Hue ─────────────────────────────
    angles_rad = np.deg2rad(h_channel.astype(np.float64) * 2.0)
    mean_complex = np.mean(np.exp(1j * angles_rad))
    h_mean_deg = np.degrees(np.angle(mean_complex)) % 360.0
    h_mean = float(h_mean_deg / 2.0)

    R = float(np.abs(mean_complex))
    h_std = float(np.sqrt(max(-2.0 * np.log(R + 1e-9), 0.0)) * 90.0 / np.pi)

    return {
        "h_mean": h_mean,
        "h_std":  h_std,
        "s_mean": float(np.mean(s_channel)),
        "s_std":  float(np.std(s_channel)),
        "v_mean": float(np.mean(v_channel)),
        "v_std":  float(np.std(v_channel)),
    }


def extract_dominant_colors(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Trích xuất 3 màu chủ đạo bằng KMeans.

    Args:
        image_bgr: Ảnh BGR đầu vào.
        mask:      Mặt nạ vùng cây (0/255). None = dùng toàn ảnh.

    Returns:
        dominant_colors: numpy array shape (9,) – 3 màu RGB [0, 1].
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if mask is not None:
        pixels = image_rgb[mask == 255].astype(np.float32)
    else:
        pixels = image_rgb.reshape(-1, 3).astype(np.float32)

    if len(pixels) < N_DOMINANT_COLORS:
        return np.zeros(N_DOMINANT_COLORS * 3, dtype=np.float32)

    # Subsample để KMeans chạy nhanh hơn
    if len(pixels) > KMEANS_MAX_PIXELS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pixels), KMEANS_MAX_PIXELS, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    kmeans = KMeans(n_clusters=N_DOMINANT_COLORS, n_init="auto", random_state=42)
    kmeans.fit(sample)

    # minlength đảm bảo counts luôn có đủ N_DOMINANT_COLORS phần tử,
    # ngay cả khi ảnh đồng nhất và KMeans không dùng hết tất cả cluster.
    counts = np.bincount(kmeans.labels_, minlength=N_DOMINANT_COLORS)
    sorted_indices = np.argsort(-counts)
    dominant_colors = kmeans.cluster_centers_[sorted_indices].flatten()
    dominant_colors = dominant_colors / 255.0

    return dominant_colors.astype(np.float32)


def extract_green_ratio(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Tính tỷ lệ pixel màu xanh lá trong vùng cây (Green Ratio).

    Định nghĩa "xanh lá": Hue ∈ [35°, 85°] (OpenCV: [17, 42]) trong HSV.
    Đây là chỉ số đặc trưng quan trọng phân biệt cây lá xanh vs cây khô/mùa đông.

    Args:
        image_bgr: Ảnh BGR đầu vào.
        mask:      Mặt nạ vùng cây (0/255). None = dùng toàn ảnh.

    Returns:
        float: Tỷ lệ pixel xanh lá [0, 1].
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Xanh lá: Hue [35°, 85°] → OpenCV [17, 42], S > 40, V > 40
    green_lower = np.array([17, 40, 40], dtype=np.uint8)
    green_upper = np.array([42, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(image_hsv, green_lower, green_upper)

    if mask is not None:
        roi_pixels = float(np.sum(mask == 255))
        green_in_roi = float(np.sum((green_mask == 255) & (mask == 255)))
    else:
        h, w = image_bgr.shape[:2]
        roi_pixels = float(h * w)
        green_in_roi = float(np.sum(green_mask == 255))

    if roi_pixels == 0:
        return 0.0
    return green_in_roi / roi_pixels


def extract_color_features(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Hàm chính: trích rút toàn bộ đặc trưng màu sắc từ ảnh cây.

    Args:
        image_bgr: Ảnh BGR đầu vào (numpy array H x W x 3).
        mask:      Mặt nạ vùng cây (0/255). Nếu None, sẽ tự tính.

    Returns:
        dict chứa tất cả đặc trưng màu:
            - hue_hist_0 .. hue_hist_7   (8 giá trị histogram sắc độ, [0, 1])
            - h_mean, h_std              (circular mean/std kênh H)
            - s_mean, s_std, v_mean, v_std
            - dom_r1, dom_g1, dom_b1 .. dom_r3, dom_g3, dom_b3
            - green_ratio               (tỷ lệ pixel xanh lá [0, 1])
    """
    if mask is None:
        mask = create_tree_mask(image_bgr)

    # Nếu mask quá nhỏ (< 5%), dùng toàn ảnh
    h, w = image_bgr.shape[:2]
    use_mask = mask if np.sum(mask == 255) >= h * w * 0.05 else None

    # ── 1. Hue Histogram ─────────────────────────────────
    hue_hist = extract_hue_histogram(image_bgr, use_mask)
    hist_dict = {f"hue_hist_{i}": float(hue_hist[i]) for i in range(HUE_BINS)}

    # ── 2. Thống kê kênh màu HSV ──────────────────────────
    hsv_stats = extract_hsv_statistics(image_bgr, use_mask)

    # ── 3. Màu chủ đạo (KMeans) ──────────────────────────
    dominant = extract_dominant_colors(image_bgr, use_mask)
    dom_dict = {}
    color_labels = ["r", "g", "b"]
    for color_idx in range(N_DOMINANT_COLORS):
        for ch_idx, ch_name in enumerate(color_labels):
            key = f"dom_{ch_name}{color_idx + 1}"
            dom_dict[key] = float(dominant[color_idx * 3 + ch_idx])

    # ── 4. Green Ratio ────────────────────────────────────
    green_ratio = extract_green_ratio(image_bgr, use_mask)

    return {**hist_dict, **hsv_stats, **dom_dict, "green_ratio": green_ratio}


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

    features = extract_color_features(img)
    print("\n=== DAC TRUNG MAU SAC (CAY) ===")
    for key, val in features.items():
        print(f"  {key:<20} : {val:.6f}")
    print(f"\nTong so dac trung mau: {len(features)}")
