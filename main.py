"""
main.py  [v2.1 – Fisher-Weighted Features]
--------------------------------------------
Pipeline CBIR: trích rút → chuẩn hóa → Fisher weighting → BallTree → truy vấn.

Cải tiến:
    - Trích rút 37 chiều (bao gồm màu chủ đạo tính bằng 3D Histogram)
    - Tự động chuẩn hóa Z-score
    - Lập chỉ mục bằng cấu trúc BallTree (kNN search)

Sử dụng:
    python main.py --build --image_dir tree/
    python main.py --query <anh> --k 5
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from feature_extractor import TreeFeatureExtractor
from vector_normalizer import VectorNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

DB_PATH      = Path("vector_db.npz")
NORM_PATH    = Path("normalizer.npz")





# ════════════════════════════════════════════════════════
#  Build DB
# ════════════════════════════════════════════════════════

def build_database(image_dir: str) -> None:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        logger.error(f"Thu muc khong ton tai: {image_dir}")
        sys.exit(1)

    # ── Step 1: Extract features ─────────────────────────
    logger.info("=== BUOC 1: Trich rut dac trung ===")
    extractor = TreeFeatureExtractor()
    raw = extractor.extract_batch(str(image_dir))
    ok = [r for r in raw if r["success"]]

    if not ok:
        logger.error("Khong trich rut duoc anh nao.")
        sys.exit(1)

    # Filter consistent dimensions
    dims = [len(r["vector"]) for r in ok]
    expected = max(set(dims), key=dims.count)
    ok = [r for r in ok if len(r["vector"]) == expected]
    logger.info(f"OK: {len(ok)} anh, {expected} chieu")

    vectors_raw = np.array([r["vector"] for r in ok], dtype=np.float32)
    image_paths = [r["image_path"] for r in ok]
    labels = [_infer_label(r["image_path"]) for r in ok]

    # ── Step 2: Z-score normalize + clip ─────────────────
    logger.info("=== BUOC 2: Z-score normalize ===")
    normalizer = VectorNormalizer(method="zscore")
    normalizer.fit(vectors_raw)
    vectors_norm = normalizer.transform(vectors_raw)
    vectors_norm = np.clip(vectors_norm, -3.0, 3.0)
    normalizer.save(NORM_PATH)

    np.savez_compressed(
        DB_PATH,
        vectors=vectors_norm,
        image_paths=np.array(image_paths),
        labels=np.array(labels),
    )

    logger.info(f"=== KET QUA ===")
    logger.info(f"  Records    : {len(ok)}")
    logger.info(f"  So chieu   : {expected}")
    logger.info(f"  Loai cay   : {len(set(labels))}")


# ════════════════════════════════════════════════════════
#  Query
# ════════════════════════════════════════════════════════

def query_image(image_path: str, k: int = 5) -> list:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"File khong ton tai: {image_path}")

    for p in [DB_PATH, NORM_PATH]:
        if not p.exists():
            raise RuntimeError(f"Chua co {p}. Chay --build truoc.")

    normalizer = VectorNormalizer.load(NORM_PATH)
    data = np.load(DB_PATH, allow_pickle=True)
    paths_db = data["image_paths"].tolist()
    labels_db = data["labels"].tolist()
    vectors_db = data["vectors"]

    extractor = TreeFeatureExtractor()
    result = extractor.extract(str(image_path))
    if not result["success"]:
        raise RuntimeError(f"Trich rut that bai: {result['errors']}")

    q_vec = result["vector"]
    q_norm = normalizer.transform_one(q_vec)
    q_norm = np.clip(q_norm, -3.0, 3.0)
    q_norm = q_norm.reshape(1, -1)

    # Pure Numpy Vectorized Euclidean Distance
    distances = np.linalg.norm(vectors_db - q_norm, axis=1)
    
    # Lấy top-k
    indices = np.argsort(distances)[:k]

    results = []
    for rank, idx in enumerate(indices, 1):
        results.append({
            "rank": rank,
            "image_path": paths_db[idx],
            "label": labels_db[idx],
            "distance": float(distances[idx]),
        })
    return results


def _infer_label(image_path: str) -> str:
    p = Path(image_path)
    parent = p.parent.name
    if parent and parent not in ("res", "tree", "batch", "train", "test"):
        return parent
    stem = p.stem
    if "_tree_" in stem:
        return stem.split("_tree_")[0].split("_")[0]
    return parent or "unknown"


# ════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Tree CBIR v2.1")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--image_dir", type=str, default="tree")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.build:
        build_database(args.image_dir)
    elif args.query:
        try:
            results = query_image(args.query, k=args.k)
        except Exception as e:
            logger.error(str(e))
            sys.exit(1)

        print(f"\n{'='*65}")
        print(f"  TOP-{args.k} KET QUA:")
        print(f"{'='*65}")
        for r in results:
            print(f"  #{r['rank']:>2}  {Path(r['image_path']).name:<50}"
                  f"  [{r['label']:<20}]  dist={r['distance']:.4f}")
        print(f"{'='*65}\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()