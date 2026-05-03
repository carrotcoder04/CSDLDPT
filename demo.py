"""
demo.py
-------
Script demo trực quan hóa quá trình trích rút đặc trưng cây.

Hiển thị:
    - Ảnh gốc và mặt nạ cây (tiền cảnh/hậu cảnh)
    - Kết quả kết cấu (LBP map, Gradient magnitude)
    - Phân bố pixel theo chiều dọc (Vertical Profile)
    - Đặc trưng màu sắc (HUE histogram, Green Ratio)
    - Bảng tổng hợp toàn bộ đặc trưng

Sử dụng:
    python demo.py <đường_dẫn_ảnh_cây>

Tài liệu tham khảo: Nhóm 6 - Báo cáo ĐPT - Hệ CSDL Đa Phương Tiện
"""

import sys
import io
import cv2
import numpy as np

# Fix encoding cho terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from feature_extractor import TreeFeatureExtractor, FEATURE_GROUP_ORDER
from features.mask_utils import create_tree_mask
from features.texture_features import _compute_lbp


def visualize_tree_features(image_path: str):
    """
    Trực quan hóa toàn bộ pipeline trích rút đặc trưng cây.

    Bố cục giao diện (3 hàng × 4 cột):
        Hàng 1: Ảnh gốc | Mặt nạ cây | Contour tán | Gradient magnitude
        Hàng 2: LBP map | Vertical Profile | Hue Histogram | Green mask
        Hàng 3: Đặc trưng hình thái (bar) | Đặc trưng kết cấu (bar) | Bảng tóm tắt (span 2)

    Args:
        image_path: Đường dẫn tới file ảnh cây.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[LOI] Khong doc duoc anh: {image_path}")
        return

    img_resized = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # ── Trích rút đặc trưng ───────────────────────────────
    extractor = TreeFeatureExtractor(target_size=(256, 256))
    result = extractor.extract(image_path)
    features = result["features"]

    # ── Xử lý trung gian để visualize ────────────────────
    tree_mask = create_tree_mask(img_resized)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # LBP map
    lbp_map = _compute_lbp(gray)

    # Gradient magnitude (Sobel)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_max = grad_mag.max()
    grad_mag = (grad_mag / max(grad_max, 1e-6) * 255).astype(np.uint8)

    # Green mask
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(img_hsv, np.array([17, 40, 40]), np.array([42, 255, 255]))

    # Contour tán cây
    contours, _ = cv2.findContours(tree_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = img_rgb.copy()
    if contours:
        main_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(img_contour, [main_cnt], -1, (0, 255, 100), 2)

    # Vertical profile (pixel count per row)
    row_counts = np.sum(tree_mask == 255, axis=1)

    # ── Tạo figure ────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#1e1e2e")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    title_color = "#cdd6f4"
    label_color = "#a6adc8"

    def styled_ax(ax, title):
        ax.set_title(title, color=title_color, fontsize=9, pad=7)
        ax.set_facecolor("#181825")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

    # ── Hàng 1 ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    styled_ax(ax1, "Anh goc (256x256)")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(tree_mask, cmap="gray")
    styled_ax(ax2, "Mat na cay (Otsu + Flood-fill)")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_contour)
    styled_ax(ax3, "Contour tan cay")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(grad_mag, cmap="hot")
    styled_ax(ax4, "Gradient Magnitude (Sobel)")
    ax4.axis("off")

    # ── Hàng 2 ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(lbp_map, cmap="viridis")
    styled_ax(ax5, "LBP Map (ket cau cuc bo)")
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.barh(np.arange(len(row_counts)), row_counts, color="#89b4fa", edgecolor="none", height=1.0)
    ax6.invert_yaxis()
    styled_ax(ax6, "Vertical Profile (pixel/hang)")
    ax6.set_xlabel("So pixel", color=label_color, fontsize=8)
    ax6.set_ylabel("Hang (0=tren)", color=label_color, fontsize=8)
    ax6.tick_params(colors=label_color, labelsize=7)

    ax7 = fig.add_subplot(gs[1, 2])
    hue_vals = [features.get(f"color_hue_hist_{i}", 0) for i in range(8)]
    hue_labels = [f"H{i}" for i in range(8)]
    colors_bar = plt.cm.hsv(np.linspace(0, 1, 8))
    ax7.bar(hue_labels, hue_vals, color=colors_bar, edgecolor="#45475a", linewidth=0.5)
    styled_ax(ax7, "Hue Histogram (8 bins)")
    ax7.tick_params(colors=label_color, labelsize=8)
    ax7.set_ylabel("Ty le", color=label_color, fontsize=8)

    ax8 = fig.add_subplot(gs[1, 3])
    # Overlay green mask lên ảnh
    green_overlay = img_rgb.copy()
    green_overlay[green_mask == 255] = [100, 220, 100]
    ax8.imshow(green_overlay)
    gr = features.get("color_green_ratio", 0)
    styled_ax(ax8, f"Green Mask (green_ratio={gr:.3f})")
    ax8.axis("off")

    # ── Hàng 3 ───────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 0])
    shape_keys  = ["aspect_ratio", "solidity", "symmetry", "crown_ratio"]
    shape_vals  = [features.get(f"shape_{k}", 0) for k in shape_keys]
    shape_short = ["Aspect", "Solid.", "Sym.", "Crown"]
    ax9.bar(shape_short, shape_vals,
            color=["#89b4fa", "#a6e3a1", "#fab387", "#cba6f7"],
            edgecolor="#45475a")
    styled_ax(ax9, "Dac trung hinh thai")
    ax9.tick_params(colors=label_color, labelsize=8)
    ax9.set_ylim(0, max(max(shape_vals) * 1.3, 1.2))

    ax10 = fig.add_subplot(gs[2, 1])
    tex_keys  = ["contrast", "homogeneity", "energy", "roughness"]
    tex_vals  = [features.get(f"texture_{k}", 0) for k in tex_keys]
    tex_short = ["Contr.", "Homog.", "Energy", "Rough."]
    ax10.bar(tex_short, tex_vals,
             color=["#f38ba8", "#89dceb", "#a6e3a1", "#f9e2af"],
             edgecolor="#45475a")
    styled_ax(ax10, "Dac trung ket cau")
    ax10.tick_params(colors=label_color, labelsize=8)

    # Bảng tóm tắt (span 2 cột)
    ax11 = fig.add_subplot(gs[2, 2:])
    ax11.axis("off")
    styled_ax(ax11, "Bang tom tat dac trung")

    summary_data = [
        ["Nhom", "Dac trung", "Gia tri"],
        ["Mau sac",  "H_mean (°)",       f"{features.get('color_h_mean', 0):.2f}"],
        ["Mau sac",  "S_mean",            f"{features.get('color_s_mean', 0):.2f}"],
        ["Mau sac",  "Green Ratio",       f"{features.get('color_green_ratio', 0):.4f}"],
        ["Hinh thai","Aspect Ratio",      f"{features.get('shape_aspect_ratio', 0):.4f}"],
        ["Hinh thai","Solidity",          f"{features.get('shape_solidity', 0):.4f}"],
        ["Hinh thai","Symmetry",          f"{features.get('shape_symmetry', 0):.4f}"],
        ["Hinh thai","Crown Ratio",       f"{features.get('shape_crown_ratio', 0):.4f}"],
        ["Ket cau",  "Contrast",          f"{features.get('texture_contrast', 0):.4f}"],
        ["Ket cau",  "Roughness",         f"{features.get('texture_roughness', 0):.4f}"],
        ["Tan cay",  "Contour Complex.",  f"{features.get('canopy_contour_complexity', 0):.4f}"],
        ["Tan cay",  "N Components",      f"{features.get('canopy_n_components', 0):.0f}"],
        ["",         f"Tong: {result['n_features']} dac trung", f"{result['processing_time_ms']:.0f} ms"],
    ]

    table = ax11.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("#313244" if row % 2 == 0 else "#181825")
        cell.set_text_props(color=title_color)
        cell.set_edgecolor("#45475a")
        if row == 0:
            cell.set_facecolor("#45475a")
            cell.set_text_props(color="#cba6f7", fontweight="bold")

    # ── Tiêu đề tổng thể ──────────────────────────────────
    fig.suptitle(
        f"Trich rut Dac trung Cay – {Path(image_path).name}",
        color="#cba6f7",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig("demo_output.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Da luu ket qua: demo_output.png")
    plt.show()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Tự động tìm ảnh mẫu trong thư mục tree/
        sample = next(Path("tree").rglob("*.png"), None) if Path("tree").exists() else None
        if sample:
            print(f"Demo voi: {sample}")
            visualize_tree_features(str(sample))
        else:
            print("Su dung: python demo.py <duong_dan_anh_cay>")
            print("Vi du:   python demo.py tree/Ginkgo.../image.png")
            sys.exit(1)
    else:
        visualize_tree_features(sys.argv[1])
