"""
main.py
-------
Pipeline hoàn chỉnh: Trích rút đặc trưng → Chuẩn hóa vector → Xây dựng Vector DB → Truy vấn ảnh.

Đối tượng: Ảnh toàn cây (thư mục tree/ với các loài cây khác nhau).

Các bước:
    1. Trích rút đặc trưng từ thư mục ảnh bằng TreeFeatureExtractor.
    2. Fit VectorNormalizer (zscore) trên toàn bộ vector để học mean/std.
    3. Chuẩn hóa và nạp vào VectorDatabase (KD-Tree).
    4. Truy vấn ảnh tương tự cho ảnh mới.
    5. Lưu DB và normalizer để dùng lại.

Sử dụng:
    # Xây dựng DB từ thư mục ảnh cây
    python main.py --build --image_dir tree/

    # Truy vấn ảnh cây mới
    python main.py --query <duong_dan_anh> --k 5
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from feature_extractor import TreeFeatureExtractor
from vector_normalizer import VectorNormalizer
from vector_db import VectorDatabase

# ─── Cấu hình logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ─── Đường dẫn mặc định ──────────────────────────────────
DB_PATH   = Path("vector_db.npz")
NORM_PATH = Path("normalizer.npz")


# ═════════════════════════════════════════════════════════
#  BƯỚC 1+2+3: Xây dựng DB từ thư mục ảnh
# ═════════════════════════════════════════════════════════

def build_database(image_dir: str, norm_method: str = "zscore") -> None:
    """
    Pipeline xây dựng Vector DB từ thư mục ảnh cây.

    Args:
        image_dir:   Thư mục chứa ảnh cây (có thể có thư mục con theo loài).
        norm_method: Phương pháp chuẩn hóa: 'l2' | 'zscore' | 'minmax'.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        logger.error(f"Thu muc khong ton tai: {image_dir}")
        sys.exit(1)

    # ── Bước 1: Trích rút đặc trưng ──────────────────────
    logger.info("=== BUOC 1: Trich rut dac trung cay ===")
    extractor = TreeFeatureExtractor()
    raw_results = extractor.extract_batch(str(image_dir))
    ok_results = [r for r in raw_results if r["success"]]

    if not ok_results:
        logger.error("Khong trich rut duoc anh nao thanh cong. Kiem tra thu muc.")
        sys.exit(1)

    logger.info(f"Trich rut xong: {len(ok_results)}/{len(raw_results)} anh thanh cong.")

    # ── Lọc vector có số chiều nhất quán ─────────────────
    dims = [len(r["vector"]) for r in ok_results]
    expected_dim = max(set(dims), key=dims.count)  # Chiều phổ biến nhất
    inconsistent = sum(1 for d in dims if d != expected_dim)
    if inconsistent:
        logger.warning(
            f"Bo qua {inconsistent} anh co so chieu khac ({expected_dim} chieu la chuan)."
        )
        ok_results = [r for r in ok_results if len(r["vector"]) == expected_dim]

    vectors_raw = np.array([r["vector"] for r in ok_results], dtype=np.float32)
    logger.info(f"Ma tran vector: {vectors_raw.shape}  (N x D)")

    # ── Bước 2: Fit VectorNormalizer ─────────────────────
    logger.info(f"=== BUOC 2: Fit VectorNormalizer (method='{norm_method}') ===")
    normalizer = VectorNormalizer(method=norm_method)
    normalizer.fit(vectors_raw)
    vectors_norm = normalizer.transform(vectors_raw)

    logger.info(
        f"Chuan hoa xong | "
        f"min={vectors_norm.min():.4f} max={vectors_norm.max():.4f} "
        f"mean={vectors_norm.mean():.4f}"
    )
    normalizer.save(NORM_PATH)
    logger.info(f"Da luu normalizer: {NORM_PATH}")

    # ── Bước 3: Xây dựng VectorDatabase (KD-Tree) ────────
    logger.info("=== BUOC 3: Xay dung VectorDatabase (KD-Tree) ===")
    db = VectorDatabase(distance="euclidean")

    for result, vec_norm in zip(ok_results, vectors_norm):
        fname = Path(result["image_path"]).stem
        # Tên loài từ tên file (định dạng "TenLoai_..._tree_1 (N)")
        label = fname.split("_")[0] if "_" in fname else None
        db.insert(
            image_path=result["image_path"],
            vector=vec_norm,
            label=label,
        )

    db.build_tree()
    db.save(DB_PATH)

    stats = db.stats()
    logger.info("=== KET QUA XAY DUNG DB ===")
    logger.info(f"  Records    : {stats['n_records']}")
    logger.info(f"  So chieu   : {stats['n_features']}")
    logger.info(f"  Nodes cay  : {stats['n_tree_nodes']}")
    logger.info(f"  Loai cay   : {stats['n_labels']} ({stats['labels']})")
    logger.info(f"  File DB    : {DB_PATH}")
    logger.info(f"  Normalizer : {NORM_PATH}")


# ═════════════════════════════════════════════════════════
#  BƯỚC 4: Truy vấn ảnh mới
# ═════════════════════════════════════════════════════════

def query_image(image_path: str, k: int = 5) -> None:
    """
    Truy vấn k ảnh cây tương tự với ảnh mới.

    Args:
        image_path: Đường dẫn ảnh cần truy vấn.
        k:          Số kết quả trả về.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"File anh khong ton tai: {image_path}")
        sys.exit(1)

    if not DB_PATH.exists() or not NORM_PATH.exists():
        logger.error(
            "Chua co DB hoac Normalizer. "
            "Chay: python main.py --build --image_dir tree/"
        )
        sys.exit(1)

    normalizer = VectorNormalizer.load(NORM_PATH)
    db = VectorDatabase.load(DB_PATH)

    extractor = TreeFeatureExtractor(normalizer=normalizer)
    result = extractor.extract(str(image_path))

    if not result["success"]:
        logger.error(f"Khong the trich rut dac trung: {result['errors']}")
        sys.exit(1)

    query_vec = result.get("vector_normalized")
    if query_vec is None:
        query_vec = result["vector"]

    logger.info(f"Truy van: {image_path.name} | {result['n_features']} dac trung")

    results = db.query(query_vec, k=k)

    print(f"\n{'='*65}")
    print(f"  ANH TRUY VAN : {image_path.name}")
    print(f"  SO CHIEU     : {result['n_features']}")
    print(f"  THOI GIAN    : {result['processing_time_ms']:.1f} ms")
    print(f"{'='*65}")
    print(f"  TOP-{k} ANH CAY TUONG TU:\n")
    for r in results:
        label_str = r["label"] or "?"
        print(
            f"  #{r['rank']:>2}  {Path(r['image_path']).name:<50}"
            f"  [{label_str:<20}]  dist={r['distance']:.4f}"
        )
    print(f"{'='*65}\n")


# ═════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tree Image Vector Database – KD-Tree"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Xay dung Vector DB tu thu muc anh cay"
    )
    parser.add_argument(
        "--image_dir", type=str, default="tree",
        help="Thu muc chua anh cay (dung voi --build, mac dinh: tree/)"
    )
    parser.add_argument(
        "--norm", type=str, default="zscore",
        choices=["l2", "zscore", "minmax"],
        help="Phuong phap chuan hoa vector (mac dinh: zscore)"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Duong dan anh cay can truy van"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="So ket qua tra ve (mac dinh: 5)"
    )

    args = parser.parse_args()

    if args.build:
        build_database(args.image_dir, norm_method=args.norm)
    elif args.query:
        query_image(args.query, k=args.k)
    else:
        # Demo nhanh
        sample_dirs = list(Path("tree").rglob("*.png"))[:1] if Path("tree").exists() else []
        if sample_dirs:
            demo_image = str(sample_dirs[0])
            logger.info("=== DEMO: Trich rut 1 anh va hien thi vector ===")
            extractor = TreeFeatureExtractor()
            result = extractor.extract(demo_image)
            if result["success"]:
                print(f"\n  Anh        : {Path(result['image_path']).name}")
                print(f"  So chieu   : {result['n_features']}")
                print(f"  Thoi gian  : {result['processing_time_ms']:.1f} ms")
                print(f"  Vector[0:5]: {result['vector'][:5]}")
                print(f"\n  De xay DB  : python main.py --build --image_dir tree/")
                print(f"  De truy van: python main.py --query <anh> --k 5")
        else:
            parser.print_help()


if __name__ == "__main__":
    main()