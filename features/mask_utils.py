"""
mask_utils.py
-------------
Tiện ích tạo mặt nạ nhị phân (tree mask) dùng chung cho tất cả các module đặc trưng.

Đối tượng: Ảnh toàn cây (chụp cả thân + tán), nền đa dạng (trời, tường, đất).

Thuật toán:
    1. Otsu threshold trên grayscale → phân tách tiền cảnh / hậu cảnh sơ bộ.
    2. Flood-fill từ 4 góc ảnh → đánh dấu nền liên thông với biên.
    3. Fallback thông minh nếu mask rỗng (ảnh nền tối hoặc nền sáng).
    4. Morphology CLOSE + OPEN → lấp lỗ hổng, loại nhiễu nhỏ.

Tài liệu tham khảo: Nhóm 6 - Báo cáo ĐPT - Hệ CSDL Đa Phương Tiện
"""

import cv2
import numpy as np


def create_tree_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Tạo mặt nạ nhị phân xác định vùng cây (tiền cảnh).

    Thuật toán:
        1. GaussianBlur + Otsu threshold – tự động tìm ngưỡng phân tách.
        2. Flood-fill từ 4 góc ảnh – đánh dấu vùng nền kết nối với biên.
        3. Invert: vùng chưa bị fill = vùng cây → đặt thành 255.
        4. Fallback: nếu mask rỗng (< 1% ảnh), phán đoán dựa vào màu góc ảnh.
        5. Morphology CLOSE + OPEN – lấp lỗ hổng nhỏ và loại nhiễu.

    Args:
        image_bgr: Ảnh BGR đầu vào (numpy array H x W x 3).

    Returns:
        mask: Mảng nhị phân 0/255, pixel cây = 255, nền = 0.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ── Bước 1: Otsu threshold ────────────────────────────
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Bước 2: Flood-fill từ 4 góc ──────────────────────
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    canvas = otsu.copy()
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for (r, c) in corners:
        cv2.floodFill(
            canvas, flood_mask, (c, r), 128,
            loDiff=10, upDiff=10,
            flags=cv2.FLOODFILL_FIXED_RANGE,
        )

    # ── Bước 3: Pixel chưa fill (≠ 128) = vùng cây ───────
    tree_m = np.where(canvas == 128, 0, 255).astype(np.uint8)

    # ── Bước 4: Fallback nếu mask quá rỗng ───────────────
    if np.sum(tree_m == 255) < h * w * 0.01:
        corner_mean = float(np.mean([gray[r, c] for (r, c) in corners]))
        if corner_mean > 127:
            # Nền sáng (trời, tường trắng) → cây là vùng tối hơn
            tree_m = cv2.bitwise_not(otsu)
        else:
            # Nền tối → cây là vùng sáng hơn
            tree_m = otsu.copy()

    # ── Bước 5: Morphology cleanup ────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tree_m = cv2.morphologyEx(tree_m, cv2.MORPH_CLOSE, kernel)
    tree_m = cv2.morphologyEx(tree_m, cv2.MORPH_OPEN, kernel)

    return tree_m


# Alias giữ tương thích nội bộ nếu cần
create_leaf_mask = create_tree_mask
