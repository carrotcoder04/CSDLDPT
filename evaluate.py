"""
evaluate.py  [v2 – Redesigned]
--------------------------------
Đánh giá hiệu năng hệ thống CBIR (Content-Based Image Retrieval):
    - Precision@5: tỷ lệ kết quả đúng trong top-5
    - mAP@5: mean Average Precision at 5
    - Precision@1: tỷ lệ kết quả top-1 đúng

Sử dụng:
    python evaluate.py
    python evaluate.py --k 5 --sample 200
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("evaluate")

APP_DIR  = Path(__file__).parent.resolve()
DB_PATH  = APP_DIR / "vector_db.npz"


# ════════════════════════════════════════════════════════
#  Tải hệ thống
# ════════════════════════════════════════════════════════

def load_system():
    """
    Tải dữ liệu vector DB từ file.

    Returns:
        (vectors, image_paths, labels)
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay {DB_PATH}. "
            "Chay: python main.py --build --image_dir tree/"
        )

    print("Dang tai CSDL...", end=" ", flush=True)
    data = np.load(DB_PATH, allow_pickle=True)
    vectors = data["vectors"]
    image_paths = data["image_paths"].tolist()
    labels = data["labels"].tolist()

    print(f"OK ({len(image_paths)} records, {vectors.shape[1]} chieu)\n")
    return vectors, image_paths, labels


# ════════════════════════════════════════════════════════
#  Tính toán chỉ số đánh giá
# ════════════════════════════════════════════════════════

def precision_at_k(retrieved_labels: list, query_label: str, k: int) -> float:
    """
    Precision@k = (Số kết quả đúng trong top-k) / k.

    Args:
        retrieved_labels: Danh sách nhãn kết quả, theo thứ tự top-1 đến top-k.
        query_label:      Nhãn đúng của ảnh truy vấn.
        k:                Số kết quả xem xét.

    Returns:
        float: Precision@k trong khoảng [0, 1].
    """
    top_k = retrieved_labels[:k]
    correct = sum(1 for lbl in top_k if lbl == query_label)
    return correct / k


def average_precision_at_k(retrieved_labels: list, query_label: str, k: int) -> float:
    """
    Average Precision@k (AP@k) cho một query.

    Công thức:
        AP@k = (1 / min(R, k)) × Σ_{i=1}^{k} P(i) × rel(i)

    Trong đó:
        - R = tổng số ảnh cùng nhãn trong DB (trừ chính query).
        - P(i) = Precision tại vị trí i.
        - rel(i) = 1 nếu kết quả thứ i đúng nhãn, ngược lại = 0.

    Args:
        retrieved_labels: Danh sách nhãn kết quả (top-1 đến top-k).
        query_label:      Nhãn đúng của ảnh truy vấn.
        k:                Số kết quả xem xét.

    Returns:
        float: AP@k trong khoảng [0, 1].
    """
    hits = 0
    sum_precision = 0.0
    for i, lbl in enumerate(retrieved_labels[:k], start=1):
        if lbl == query_label:
            hits += 1
            sum_precision += hits / i
    return sum_precision / k if k > 0 else 0.0


def evaluate(vectors: np.ndarray, image_paths: list, labels: list,
             k: int = 5, sample_size: int = None, seed: int = 42) -> dict:
    """
    Vòng lặp đánh giá: tính Precision@k, Precision@1, mAP@k cho toàn bộ DB.

    Thuật toán:
        Với mỗi ảnh query i:
            1. Tính khoảng cách Euclidean thuần túy (Numpy) tới tất cả vector khác.
            2. Loại kết quả có khoảng cách ~ 0 (chính ảnh query).
            3. Tính Precision@k, AP@k.
        Trung bình tất cả query → P@k, mAP@k.

    Args:
        vectors:    numpy array (N, D) – vector đã chuẩn hóa.
        image_paths: Danh sách đường dẫn ảnh.
        labels:     Danh sách nhãn tương ứng.
        k:          Số kết quả top-k.
        sample_size: Số lượng query (None = tất cả).
        seed:       Random seed cho sampling.

    Returns:
        dict: {precision_at_1, precision_at_k, map_at_k, n_queries,
               per_label_precision, per_label_ap}
    """
    n = len(vectors)
    indices = list(range(n))
    if sample_size and sample_size < n:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=sample_size, replace=False).tolist()
        print(f"Lay mau {sample_size}/{n} anh de danh gia...\n")
    else:
        print(f"Danh gia toan bo {n} anh...\n")

    p1_list = []
    pk_list = []
    ap_list = []
    per_label_pk = defaultdict(list)
    per_label_ap = defaultdict(list)

    t0 = time.perf_counter()

    for qi, idx in enumerate(indices):
        q_vec = vectors[idx].reshape(1, -1)
        q_label = labels[idx]

        # Numpy Euclidean distance
        dists = np.linalg.norm(vectors - q_vec, axis=1)
        # Lấy top k+1 để phòng khi lấy chính nó
        top_k_plus_1 = np.argsort(dists)[:k + 1]

        # Lọc chính nó (distance ≈ 0)
        retrieved = []
        for j in top_k_plus_1:
            if dists[j] > 1e-9:
                retrieved.append(labels[j])
        retrieved = retrieved[:k]

        if not retrieved:
            continue

        p1 = 1.0 if retrieved[0] == q_label else 0.0
        pk = precision_at_k(retrieved, q_label, k)
        ap = average_precision_at_k(retrieved, q_label, k)

        p1_list.append(p1)
        pk_list.append(pk)
        ap_list.append(ap)
        per_label_pk[q_label].append(pk)
        per_label_ap[q_label].append(ap)

        if (qi + 1) % 100 == 0 or (qi + 1) == len(indices):
            elapsed = time.perf_counter() - t0
            print(
                f"  [{qi+1:>4}/{len(indices)}]  "
                f"mAP so bo: {np.mean(ap_list):.4f}  P@{k}: {np.mean(pk_list):.4f}  "
                f"({elapsed:.1f}s)",
                end="\r",
            )

    print()  # newline

    return {
        "precision_at_1": float(np.mean(p1_list)) if p1_list else 0.0,
        "precision_at_k": float(np.mean(pk_list)) if pk_list else 0.0,
        "map_at_k":       float(np.mean(ap_list)) if ap_list else 0.0,
        "k":              k,
        "n_queries":      len(p1_list),
        "per_label_precision": {lbl: float(np.mean(v)) for lbl, v in per_label_pk.items()},
        "per_label_ap":        {lbl: float(np.mean(v)) for lbl, v in per_label_ap.items()},
        "ap_scores":      ap_list,
    }


# ════════════════════════════════════════════════════════
#  Báo cáo
# ════════════════════════════════════════════════════════

def print_report(result: dict) -> None:
    """In báo cáo đánh giá chi tiết ra console."""
    k = result["k"]
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  DANH GIA HE THONG CBIR – Precision@{k} / mAP@{k}")
    print(sep)
    print(f"\n  Tong so query : {result['n_queries']}")
    print(f"\n[1] CHI SO TONG THE")
    print(f"  Precision@1    : {result['precision_at_1']:.4f}"
          f"   ({result['precision_at_1']*100:.1f}% top-1 dung)")
    print(f"  Precision@{k}    : {result['precision_at_k']:.4f}"
          f"   ({result['precision_at_k']*100:.1f}% trong top-{k})")
    print(f"  mAP@{k}         : {result['map_at_k']:.4f}",
          "  ★★★" if result["map_at_k"] > 0.7 else
          "  ★★" if result["map_at_k"] > 0.4 else "  ★")

    print(f"\n[2] PRECISION@{k} THEO TUNG LOAI (tang dan)")
    sorted_per_label = sorted(
        result["per_label_precision"].items(), key=lambda x: x[1]
    )
    for lbl, p in sorted_per_label:
        bar = "█" * int(p * 20)
        print(f"  {p:.4f} |{bar:<20}| {lbl[:45]}")

    print(f"\n[3] mAP@{k} THEO TUNG LOAI (tang dan)")
    sorted_ap = sorted(result["per_label_ap"].items(), key=lambda x: x[1])
    for lbl, ap in sorted_ap:
        bar = "█" * int(ap * 20)
        print(f"  {ap:.4f} |{bar:<20}| {lbl[:45]}")

    aps = result["ap_scores"]
    low = sum(1 for a in aps if a < 0.2)
    mid = sum(1 for a in aps if 0.2 <= a < 0.5)
    high = sum(1 for a in aps if a >= 0.5)
    print(f"\n[4] PHAN PHOI AP@{k}")
    print(f"  < 0.2  : {low:>4}  ({low/len(aps)*100:.0f}%)")
    print(f"  0.2-0.5: {mid:>4}  ({mid/len(aps)*100:.0f}%)")
    print(f"  >= 0.5 : {high:>4}  ({high/len(aps)*100:.0f}%)")

    print(f"\n[5] CHAN DOAN")
    m = result["map_at_k"]
    if m >= 0.7:
        print("  ✓ mAP cao – he thong hoat dong tot.")
    elif m >= 0.4:
        print("  ~ mAP trung binh – co the cai thien.")
        print("    → Goi y: kiem tra lai chat luong mask, thu tang k.")
    else:
        print("  ✗ mAP thap – vector dac trung chua du discriminative.")
        print("    → Goi y: (a) kiem tra anh dau vao, (b) xem lai cach tinh mask,")
        print("             (c) thu them dac trung hoac dung metric learning.")

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Danh gia he thong CBIR – Precision@k va mAP@k"
    )
    parser.add_argument("--k", type=int, default=5,
                        help="So ket qua truy van (default: 5)")
    parser.add_argument("--sample", type=int, default=None,
                        help="So luong query mau (default: toan bo DB)")
    args = parser.parse_args()

    try:
        vectors, image_paths, labels = load_system()
    except FileNotFoundError as e:
        print(f"[LOI] {e}")
        sys.exit(1)

    result = evaluate(vectors, image_paths, labels,
                      k=args.k, sample_size=args.sample)
    print_report(result)


if __name__ == "__main__":
    main()
