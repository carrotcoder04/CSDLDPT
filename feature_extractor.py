"""
feature_extractor.py
--------------------
Module điều phối tổng hợp – tích hợp 4 nhóm đặc trưng thành một vector duy nhất.

Đối tượng: Ảnh toàn cây (chụp cả thân + tán, nền đa dạng).

Kết hợp:
    - color_features.py   → Đặc trưng màu sắc   (24 chiều)
    - shape_features.py   → Đặc trưng hình thái  (12 chiều)
    - texture_features.py → Đặc trưng kết cấu    (17 chiều)
    - canopy_features.py  → Đặc trưng tán cây    ( 9 chiều)

Tổng: 62 chiều đặc trưng.

Thiết kế:
    - Mask cây được tính MỘT LẦN DUY NHẤT rồi truyền vào cả 4 module.
    - Thứ tự vector ổn định theo nhóm ["color","shape","texture","canopy"],
      trong mỗi nhóm sắp xếp alphabetical → không thay đổi khi thêm/bỏ nhóm khác.
    - Tích hợp VectorNormalizer: chuẩn hóa vector ngay sau khi trích rút.

Sử dụng:
    from feature_extractor import TreeFeatureExtractor
    from vector_normalizer import VectorNormalizer

    norm = VectorNormalizer(method="zscore")
    extractor = TreeFeatureExtractor(normalizer=norm)
    result = extractor.extract("tree.jpg")
    print(result["vector"])            # vector thô (float32)
    print(result["vector_normalized"]) # vector đã chuẩn hóa (float32, None nếu chưa fit)

Tài liệu tham khảo: Nhóm 6 - Báo cáo ĐPT - Hệ CSDL Đa Phương Tiện
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

# Import 4 module đặc trưng
from features.color_features import extract_color_features
from features.shape_features import extract_shape_features
from features.texture_features import extract_texture_features
from features.canopy_features import extract_canopy_features
from features.mask_utils import create_tree_mask
from vector_normalizer import VectorNormalizer

# ─────────────────────────────────────────────
#  Cấu hình logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TreeFeatureExtractor")

# ─────────────────────────────────────────────
#  Thứ tự nhóm đặc trưng cố định
# ─────────────────────────────────────────────
# QUAN TRỌNG: không thay đổi thứ tự này sau khi đã xây dựng CSDL.
FEATURE_GROUP_ORDER = ["color", "shape", "texture", "canopy"]


class TreeFeatureExtractor:
    """
    Lớp điều phối trích rút đặc trưng ảnh cây.

    Tích hợp 4 nhóm đặc trưng: Màu sắc, Hình thái, Kết cấu, Tán cây.

    Attributes:
        target_size (tuple):  Kích thước ảnh chuẩn hóa (W, H). Mặc định (256, 256).
        enabled_groups (set): Nhóm đặc trưng được kích hoạt.
        normalizer:           VectorNormalizer tùy chọn.
    """

    DEFAULT_TARGET_SIZE = (256, 256)

    FEATURE_GROUPS = {
        "color":   extract_color_features,
        "shape":   extract_shape_features,
        "texture": extract_texture_features,
        "canopy":  extract_canopy_features,
    }

    def __init__(
        self,
        target_size: tuple = DEFAULT_TARGET_SIZE,
        enabled_groups: Optional[list] = None,
        normalizer: Optional[VectorNormalizer] = None,
    ):
        """
        Khởi tạo TreeFeatureExtractor.

        Args:
            target_size:     Kích thước resize ảnh (width, height). Mặc định (256, 256).
            enabled_groups:  Danh sách nhóm đặc trưng cần trích rút.
                             None = tất cả ['color', 'shape', 'texture', 'canopy'].
            normalizer:      VectorNormalizer đã fit (tùy chọn).
        """
        self.target_size = target_size
        self.enabled_groups = set(
            enabled_groups if enabled_groups else self.FEATURE_GROUPS.keys()
        )
        self.normalizer = normalizer

        invalid = self.enabled_groups - set(self.FEATURE_GROUPS.keys())
        if invalid:
            raise ValueError(
                f"Nhom dac trung khong hop le: {invalid}. "
                f"Chi ho tro: {set(self.FEATURE_GROUPS.keys())}"
            )

        logger.info(
            f"TreeFeatureExtractor khoi tao | "
            f"target_size={target_size} | "
            f"enabled_groups={self.enabled_groups} | "
            f"normalizer={'co' if normalizer else 'khong'}"
        )

    def load_image(self, image_path: Union[str, Path]):
        """
        Đọc và resize ảnh từ đường dẫn.
        Hỗ trợ ảnh RGBA (PNG đã qua SAM): dùng alpha channel làm tree mask.

        Args:
            image_path: Đường dẫn tới file ảnh.

        Returns:
            Tuple (img_bgr, alpha_mask):
                - img_bgr   : numpy array (H, W, 3) BGR, hoặc None nếu lỗi.
                - alpha_mask: numpy array (H, W) uint8 0/255 nếu có alpha,
                              None nếu ảnh không có kênh alpha.
        """
        image_path = str(image_path)
        # IMREAD_UNCHANGED để giữ alpha channel nếu có
        img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img_raw is None:
            logger.error(f"Khong the doc anh: {image_path}")
            return None, None

        alpha_mask = None

        if img_raw.ndim == 2:
            # Grayscale → chuyển sang BGR
            img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
        elif img_raw.ndim == 3 and img_raw.shape[2] == 4:
            # RGBA (PNG đã qua SAM) → tách alpha làm mask
            alpha = img_raw[:, :, 3]
            alpha_mask_full = (alpha > 127).astype(np.uint8) * 255
            img_bgr = img_raw[:, :, :3]  # chỉ giữ BGR
            # Resize alpha mask cùng kích thước
            alpha_mask = cv2.resize(
                alpha_mask_full, self.target_size, interpolation=cv2.INTER_NEAREST
            )
            alpha_mask = (alpha_mask > 127).astype(np.uint8) * 255
        elif img_raw.ndim == 3 and img_raw.shape[2] == 3:
            img_bgr = img_raw
        else:
            logger.error(f"Anh khong hop le (kenh khong xac dinh): {image_path}")
            return None, None

        img_resized = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        return img_resized, alpha_mask

    def _build_feature_names(self, all_features: dict) -> list:
        """
        Xây dựng danh sách tên đặc trưng theo thứ tự ổn định.

        Thứ tự: theo FEATURE_GROUP_ORDER, trong mỗi nhóm sắp alphabetical.

        Args:
            all_features: Dict toàn bộ đặc trưng (đã có prefix nhóm).

        Returns:
            list: Tên đặc trưng theo thứ tự ổn định.
        """
        feature_names = []
        for group in FEATURE_GROUP_ORDER:
            if group not in self.enabled_groups:
                continue
            group_keys = sorted(
                k for k in all_features if k.startswith(group + "_")
            )
            feature_names.extend(group_keys)
        return feature_names

    def extract(self, image_path: Union[str, Path]) -> dict:
        """
        Trích rút toàn bộ đặc trưng từ một ảnh cây.

        Cải tiến:
            - Mask cây tính MỘT LẦN DUY NHẤT, truyền vào cả 4 module.
            - Thứ tự vector ổn định theo nhóm.

        Args:
            image_path: Đường dẫn tới file ảnh cây.

        Returns:
            dict chứa:
                - image_path (str):     Đường dẫn ảnh
                - features (dict):      Toàn bộ đặc trưng theo từng nhóm
                - vector (np.ndarray):  Vector đặc trưng 1D (numpy float32)
                - feature_names (list): Danh sách tên đặc trưng theo thứ tự
                - n_features (int):     Tổng số đặc trưng
                - processing_time_ms (float): Thời gian xử lý (ms)
                - success (bool):       True nếu thành công
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise ValueError(f"File khong ton tai: {image_path}")

        t_start = time.perf_counter()

        # ── Bước 1: Đọc và resize ảnh ──────────────────────
        img, alpha_mask = self.load_image(image_path)
        if img is None:
            return self._error_result(str(image_path), "Khong the doc anh")

        # ── Bước 2: Tính mask cây 1 lần duy nhất ───────────
        # Ưu tiên alpha mask (từ SAM) nếu có và đủ lớn (> 1% ảnh)
        # → đảm bảo nhất quán giữa ảnh DB (đã qua SAM) và ảnh truy vấn
        h_img, w_img = img.shape[:2]
        if alpha_mask is not None and np.sum(alpha_mask == 255) >= h_img * w_img * 0.01:
            shared_mask = alpha_mask
            logger.debug("Dung alpha mask (SAM) lam tree mask")
        else:
            shared_mask = create_tree_mask(img)

        # ── Bước 3: Trích rút từng nhóm đặc trưng ──────────
        all_features = {}
        group_errors = {}

        for group_name in FEATURE_GROUP_ORDER:
            if group_name not in self.enabled_groups:
                continue
            try:
                func = self.FEATURE_GROUPS[group_name]
                group_features = func(img, mask=shared_mask)
                prefixed = {f"{group_name}_{k}": v for k, v in group_features.items()}
                all_features.update(prefixed)
                logger.debug(f"  [{group_name}] {len(prefixed)} dac trung")
            except Exception as e:
                logger.warning(f"Loi khi trich rut [{group_name}]: {e}")
                group_errors[group_name] = str(e)

        # ── Bước 4: Tạo vector đặc trưng 1D (thứ tự ổn định)
        feature_names = self._build_feature_names(all_features)
        vector = np.array(
            [all_features[k] for k in feature_names],
            dtype=np.float32,
        )

        # ── Bước 5: Chuẩn hóa vector (nếu normalizer đã fit)
        vector_normalized = None
        if self.normalizer is not None and self.normalizer.is_fitted:
            try:
                vector_normalized = self.normalizer.transform_one(vector)
                # BUG 8: Clip Z-score về [-3, 3] để giảm ảnh hưởng outlier
                # (Hu Moments, grad_mean có thể rất lớn → dominate Euclidean)
                if self.normalizer.method == "zscore":
                    vector_normalized = np.clip(vector_normalized, -3.0, 3.0)
            except Exception as e:
                logger.warning(f"Loi khi chuan hoa vector: {e}")

        t_end = time.perf_counter()
        processing_time_ms = (t_end - t_start) * 1000

        logger.info(
            f"Trich rut thanh cong: {image_path.name} | "
            f"{len(all_features)} dac trung | "
            f"{processing_time_ms:.1f} ms"
        )

        return {
            "image_path": str(image_path),
            "features": all_features,
            "vector": vector,
            "vector_normalized": vector_normalized,
            "feature_names": feature_names,
            "n_features": len(all_features),
            "processing_time_ms": processing_time_ms,
            "success": True,
            "errors": group_errors if group_errors else None,
        }

    def extract_batch(
        self,
        image_dir: Union[str, Path],
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        save_json: Optional[str] = None,
    ) -> list:
        """
        Trích rút đặc trưng cho toàn bộ ảnh trong một thư mục (bao gồm thư mục con).

        Args:
            image_dir:  Đường dẫn thư mục chứa ảnh cây.
            extensions: Các định dạng file ảnh hỗ trợ.
            save_json:  Nếu cung cấp, lưu kết quả ra file JSON này.

        Returns:
            list of dict: Kết quả trích rút cho từng ảnh.
        """
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise ValueError(f"Thu muc khong ton tai: {image_dir}")

        image_files = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        logger.info(f"Bat dau trich rut batch: {len(image_files)} anh tu {image_dir}")
        results = []

        for idx, img_path in enumerate(image_files, 1):
            try:
                result = self.extract(img_path)
                result_serializable = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in result.items()
                }
                results.append(result_serializable)
                logger.info(f"  [{idx}/{len(image_files)}] {img_path.name} OK")
            except Exception as e:
                logger.error(f"  [{idx}/{len(image_files)}] {img_path.name} FAILED - {e}")
                results.append(self._error_result(str(img_path), str(e)))

        if save_json:
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Da luu ket qua batch vao: {save_json}")

        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"Hoan thanh batch: {success_count}/{len(image_files)} thanh cong")
        return results

    @staticmethod
    def _error_result(image_path: str, error_msg: str) -> dict:
        """Tạo dict kết quả lỗi."""
        return {
            "image_path": image_path,
            "features": {},
            "vector": [],
            "vector_normalized": None,
            "feature_names": [],
            "n_features": 0,
            "processing_time_ms": 0.0,
            "success": False,
            "errors": {"general": error_msg},
        }


# Alias để tương thích với code cũ có thể dùng tên LeafFeatureExtractor
LeafFeatureExtractor = TreeFeatureExtractor


# ─────────────────────────────────────────────
#  Chạy thử nghiệm
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Su dung:")
        print("  python feature_extractor.py <duong_dan_anh>")
        print("  python feature_extractor.py <thu_muc_anh> --batch")
        sys.exit(0)

    extractor = TreeFeatureExtractor()

    if "--batch" in sys.argv:
        image_dir = sys.argv[1]
        output_json = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--output" else None
        results = extractor.extract_batch(image_dir, save_json=output_json)
        print(f"\nDa xu ly {len(results)} anh.")
        if output_json:
            print(f"Ket qua luu tai: {output_json}")
    else:
        image_path = sys.argv[1]
        result = extractor.extract(image_path)

        if result["success"]:
            print(f"\n{'='*60}")
            print(f"  Anh         : {Path(result['image_path']).name}")
            print(f"  Dac trung   : {result['n_features']} chieu")
            print(f"  Thoi gian   : {result['processing_time_ms']:.1f} ms")
            print(f"{'='*60}")
            for group in FEATURE_GROUP_ORDER:
                print(f"\n  [{group.upper()}]")
                for k, v in result["features"].items():
                    if k.startswith(group + "_"):
                        print(f"    {k:<45}: {v:.6f}")
        else:
            print(f"\n[LOI] Trich rut that bai: {result['errors']}")
