"""
feature_extractor.py  [v2 – Redesigned]
-----------------------------------------
Module điều phối tổng hợp – tích hợp 4 nhóm đặc trưng thành vector 31 chiều.

Đối tượng: Ảnh toàn cây (chụp cả thân + tán, nền đa dạng).

Kết hợp:
    - color_features.py   → Đặc trưng màu sắc   (12 chiều)
    - shape_features.py   → Đặc trưng hình thái  ( 7 chiều)
    - texture_features.py → Đặc trưng kết cấu    ( 7 chiều)
    - canopy_features.py  → Đặc trưng tán cây    ( 5 chiều)

Tổng: 31 chiều đặc trưng (giảm từ 62 chiều phiên bản cũ).

Thiết kế:
    - Mask cây được tính MỘT LẦN DUY NHẤT rồi truyền vào cả 4 module.
    - Thứ tự vector ổn định theo nhóm ["color","shape","texture","canopy"],
      trong mỗi nhóm sắp xếp alphabetical.
    - Tích hợp VectorNormalizer: chuẩn hóa vector ngay sau khi trích rút.

Sử dụng:
    from feature_extractor import TreeFeatureExtractor
    extractor = TreeFeatureExtractor()
    result = extractor.extract("tree.jpg")
    print(result["vector"])       # vector thô (float32, 31 chiều)
    print(result["n_features"])   # 31
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

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
FEATURE_GROUP_ORDER = ["color", "shape", "texture", "canopy"]


class TreeFeatureExtractor:
    """
    Lớp điều phối trích rút đặc trưng ảnh cây (31 chiều).

    Attributes:
        target_size (tuple):  Kích thước ảnh chuẩn hóa (W, H).
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
            f"TreeFeatureExtractor v2 | target_size={target_size} | "
            f"groups={sorted(self.enabled_groups)}"
        )

    def load_image(self, image_path: Union[str, Path]):
        """
        Đọc và resize ảnh. Hỗ trợ RGBA (PNG đã qua SAM).

        Returns:
            (img_bgr, alpha_mask): img_bgr (H,W,3) hoặc None; alpha_mask hoặc None.
        """
        image_path = str(image_path)
        img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img_raw is None:
            logger.error(f"Khong the doc anh: {image_path}")
            return None, None

        alpha_mask = None

        if img_raw.ndim == 2:
            img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
        elif img_raw.ndim == 3 and img_raw.shape[2] == 4:
            alpha = img_raw[:, :, 3]
            alpha_full = (alpha > 127).astype(np.uint8) * 255
            img_bgr = img_raw[:, :, :3]
            alpha_mask = cv2.resize(
                alpha_full, self.target_size, interpolation=cv2.INTER_NEAREST
            )
            alpha_mask = (alpha_mask > 127).astype(np.uint8) * 255
        elif img_raw.ndim == 3 and img_raw.shape[2] == 3:
            img_bgr = img_raw
        else:
            logger.error(f"Anh khong hop le: {image_path}")
            return None, None

        img_resized = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        return img_resized, alpha_mask

    def _build_feature_names(self, all_features: dict) -> list:
        """Xây dựng danh sách tên đặc trưng theo thứ tự ổn định."""
        names = []
        for group in FEATURE_GROUP_ORDER:
            if group not in self.enabled_groups:
                continue
            group_keys = sorted(k for k in all_features if k.startswith(group + "_"))
            names.extend(group_keys)
        return names

    def extract(self, image_path: Union[str, Path]) -> dict:
        """
        Trích rút toàn bộ đặc trưng từ một ảnh cây.

        Returns:
            dict: {image_path, features, vector, vector_normalized,
                   feature_names, n_features, processing_time_ms, success, errors}
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise ValueError(f"File khong ton tai: {image_path}")

        t_start = time.perf_counter()

        img, alpha_mask = self.load_image(image_path)
        if img is None:
            return self._error_result(str(image_path), "Khong the doc anh")

        # Mask cây: ưu tiên alpha (SAM) nếu có và đủ lớn
        h_img, w_img = img.shape[:2]
        if alpha_mask is not None and np.sum(alpha_mask == 255) >= h_img * w_img * 0.01:
            shared_mask = alpha_mask
        else:
            shared_mask = create_tree_mask(img)

        # Trích rút từng nhóm
        all_features = {}
        group_errors = {}

        for group_name in FEATURE_GROUP_ORDER:
            if group_name not in self.enabled_groups:
                continue
            try:
                func = self.FEATURE_GROUPS[group_name]
                feats = func(img, mask=shared_mask)
                prefixed = {f"{group_name}_{k}": v for k, v in feats.items()}
                all_features.update(prefixed)
            except Exception as e:
                logger.warning(f"Loi [{group_name}]: {e}")
                group_errors[group_name] = str(e)

        # Vector 1D ổn định
        feature_names = self._build_feature_names(all_features)
        vector = np.array([all_features[k] for k in feature_names], dtype=np.float32)

        # Chuẩn hóa (nếu có normalizer đã fit)
        vector_normalized = None
        if self.normalizer is not None and self.normalizer.is_fitted:
            try:
                vector_normalized = self.normalizer.transform_one(vector)
                if self.normalizer.method == "zscore":
                    vector_normalized = np.clip(vector_normalized, -3.0, 3.0)
            except Exception as e:
                logger.warning(f"Loi chuan hoa: {e}")

        t_end = time.perf_counter()
        ms = (t_end - t_start) * 1000

        logger.info(f"OK: {image_path.name} | {len(all_features)} dac trung | {ms:.1f} ms")

        return {
            "image_path": str(image_path),
            "features": all_features,
            "vector": vector,
            "vector_normalized": vector_normalized,
            "feature_names": feature_names,
            "n_features": len(all_features),
            "processing_time_ms": ms,
            "success": True,
            "errors": group_errors if group_errors else None,
        }

    def extract_batch(
        self,
        image_dir: Union[str, Path],
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        save_json: Optional[str] = None,
    ) -> list:
        """Trích rút đặc trưng cho toàn bộ ảnh trong thư mục (đệ quy)."""
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise ValueError(f"Thu muc khong ton tai: {image_dir}")

        image_files = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        logger.info(f"Batch: {len(image_files)} anh tu {image_dir}")
        results = []

        for idx, img_path in enumerate(image_files, 1):
            try:
                result = self.extract(img_path)
                result_ser = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in result.items()
                }
                results.append(result_ser)
                logger.info(f"  [{idx}/{len(image_files)}] {img_path.name} OK")
            except Exception as e:
                logger.error(f"  [{idx}/{len(image_files)}] {img_path.name} FAILED - {e}")
                results.append(self._error_result(str(img_path), str(e)))

        if save_json:
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        ok = sum(1 for r in results if r.get("success"))
        logger.info(f"Batch xong: {ok}/{len(image_files)} thanh cong")
        return results

    @staticmethod
    def _error_result(image_path: str, error_msg: str) -> dict:
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


# Alias tương thích
LeafFeatureExtractor = TreeFeatureExtractor


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Su dung: python feature_extractor.py <anh> [--batch]")
        sys.exit(0)

    extractor = TreeFeatureExtractor()

    if "--batch" in sys.argv:
        results = extractor.extract_batch(sys.argv[1])
        print(f"\nDa xu ly {len(results)} anh.")
    else:
        result = extractor.extract(sys.argv[1])
        if result["success"]:
            print(f"\n{'='*60}")
            print(f"  Anh       : {Path(result['image_path']).name}")
            print(f"  Dac trung : {result['n_features']} chieu")
            print(f"  Thoi gian : {result['processing_time_ms']:.1f} ms")
            print(f"{'='*60}")
            for group in FEATURE_GROUP_ORDER:
                print(f"\n  [{group.upper()}]")
                for k, v in result["features"].items():
                    if k.startswith(group + "_"):
                        print(f"    {k:<45}: {v:.6f}")
        else:
            print(f"\n[LOI] {result['errors']}")
