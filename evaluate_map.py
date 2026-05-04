"""
evaluate_map.py
---------------
Đánh giá hiệu năng hệ thống CBIR (Content-Based Image Retrieval):
  1. Phân tích phân phối khoảng cách (intra-class vs inter-class)
  2. Tính mAP@k (mean Average Precision at k)
  3. Tính Precision@1, Precision@5, Recall@5
  4. Gợi ý cải thiện nếu performance thấp

Sử dụng:
    python evaluate_map.py
    python evaluate_map.py --k 5 --sample 200
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Fix UnicodeEncodeError tren Windows (cp1252 -> utf-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Setup logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,   # tắt INFO spam từ các module khác
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_map")

APP_DIR = Path(__file__).parent.resolve()


# ── Load DB ──────────────────────────────────────────────────────
def load_system():
    from vector_db import VectorDatabase
    from vector_normalizer import VectorNormalizer

    print("Dang tai CSDL...", end=" ", flush=True)
    normalizer = VectorNormalizer.load(str(APP_DIR / "normalizer.npz"))
    db = VectorDatabase.load(str(APP_DIR / "vector_db.npz"))
    print(f"OK ({len(db)} records, {db._n_features} chieu)\n")
    return db, normalizer


# ── Tiện ích ─────────────────────────────────────────────────────
def get_label(record: dict) -> str:
    """
    Lay label chinh xac tu image_path.
    Dinh dang ten file: "TenLoai TenLoai_TenLoai TenLoai_tree_1 (N).png"
    → lay phan truoc "_tree_" → split "_" → lay phan dau = ten loai.
    
    Vi du: "Acer palmatum_Acer palmatum_tree_1 (8)" -> "Acer palmatum"
    """
    p = Path(record["image_path"])
    parent_name = p.parent.name
    
    # Uu tien lay ten loai tu thu muc cha
    if parent_name and parent_name not in ("res", "tree", "batch"):
        return parent_name
        
    stem = p.stem  # ten file bo duoi
    if "_tree_" in stem:
        species_part = stem.split("_tree_")[0]
        parts = species_part.split("_")
        return parts[0].strip()
        
    # Fallback: thu lay label da luu trong DB
    lbl = record.get("label")
    if lbl:
        return lbl
    return parent_name


def label_of_path(path_str: str, records: list) -> str:
    """Lấy label từ image_path string."""
    rec = next((r for r in records if r["image_path"] == path_str), None)
    if rec:
        return get_label(rec)
    return Path(path_str).parent.name


# ── mAP computation ──────────────────────────────────────────────
def average_precision_at_k(retrieved_labels: list, query_label: str, k: int) -> float:
    """
    Tính Average Precision@k cho một query.

    Args:
        retrieved_labels: List nhãn kết quả theo thứ tự (top-1 đến top-k).
        query_label:      Nhãn đúng của ảnh query.
        k:                Số kết quả xem xét.

    Returns:
        float: AP@k trong khoảng [0, 1].
    """
    if not retrieved_labels:
        return 0.0

    retrieved_labels = retrieved_labels[:k]
    hits = 0
    precision_sum = 0.0

    for i, lbl in enumerate(retrieved_labels, start=1):
        if lbl == query_label:
            hits += 1
            precision_sum += hits / i

    # Chuẩn hóa theo số lượng relevant thực sự (tối đa k)
    # Dùng min(n_relevant_in_db, k) nhưng ở đây ta chỉ dùng hits > 0 làm denominator = 1
    # (AP@k chuẩn: chia cho min(|relevant|, k))
    # Ta dùng công thức phổ biến: chia cho số relevant thực tế có trong retrieved
    return precision_sum / min(k, max(1, hits)) if hits > 0 else 0.0


def compute_map(db, k: int = 5, sample_size: int = None, seed: int = 42) -> dict:
    """
    Tính mAP@k trên toàn bộ (hoặc mẫu) DB.

    Với mỗi ảnh query:
      - Lấy k kết quả tương tự (loại trừ chính nó)
      - So sánh label kết quả với label query
      - Tính AP@k

    Returns:
        dict chứa: map_score, per_label_ap, distances_intra, distances_inter,
                   precision_at_1, precision_at_k, recall_at_k, n_queries
    """
    records = db._records
    n = len(records)

    # Chọn mẫu nếu cần
    indices = list(range(n))
    if sample_size and sample_size < n:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=sample_size, replace=False).tolist()
        print(f"Lay mau {sample_size}/{n} anh de danh gia...\n")
    else:
        print(f"Danh gia toan bo {n} anh...\n")

    ap_scores = []
    per_label_aps = defaultdict(list)
    distances_intra = []   # khoảng cách trong cùng class
    distances_inter = []   # khoảng cách khác class
    p1_hits = 0            # Precision@1
    pk_hits = []           # Precision@k (per query)
    recall_k_list = []     # Recall@k (per query)

    label_counts = defaultdict(int)
    for r in records:
        label_counts[get_label(r)] += 1

    t0 = time.perf_counter()

    for qi, idx in enumerate(indices):
        rec = records[idx]
        query_vec = rec["vector"]
        query_label = get_label(rec)

        # Query k+1 → loại chính nó
        try:
            raw_hits = db._kdtree.knn_search(query_vec, k + 1)
        except Exception as e:
            logger.warning(f"Loi query idx={idx}: {e}")
            continue

        # Lọc bỏ chính nó (distance ~ 0)
        results = [
            (dist, db._records[ridx])
            for dist, ridx in raw_hits
            if dist > 1e-9
        ][:k]

        if not results:
            continue

        retrieved_labels = [get_label(r) for _, r in results]
        retrieved_dists = [d for d, _ in results]

        # Phân loại intra/inter distance
        for dist, r in results:
            if get_label(r) == query_label:
                distances_intra.append(dist)
            else:
                distances_inter.append(dist)

        # AP@k
        ap = average_precision_at_k(retrieved_labels, query_label, k)
        ap_scores.append(ap)
        per_label_aps[query_label].append(ap)

        # Precision@1
        if retrieved_labels[0] == query_label:
            p1_hits += 1

        # Precision@k
        correct_in_k = sum(1 for l in retrieved_labels if l == query_label)
        pk_hits.append(correct_in_k / k)

        # Recall@k: số relevant lấy được / tổng relevant trong DB (trừ chính nó)
        n_relevant_total = label_counts[query_label] - 1
        recall_k = correct_in_k / max(1, min(n_relevant_total, k))
        recall_k_list.append(recall_k)

        # Progress
        if (qi + 1) % 100 == 0 or (qi + 1) == len(indices):
            elapsed = time.perf_counter() - t0
            print(f"  [{qi+1:>4}/{len(indices)}]  mAP so bo: {np.mean(ap_scores):.4f}  "
                  f"({elapsed:.1f}s)", end="\r")

    print()  # newline sau progress

    map_score = float(np.mean(ap_scores)) if ap_scores else 0.0
    p1 = p1_hits / len(indices)
    pk = float(np.mean(pk_hits)) if pk_hits else 0.0
    rk = float(np.mean(recall_k_list)) if recall_k_list else 0.0

    # Per-label mAP
    per_label_map = {
        lbl: float(np.mean(aps)) for lbl, aps in per_label_aps.items()
    }

    return {
        "map_score": map_score,
        "precision_at_1": p1,
        "precision_at_k": pk,
        "recall_at_k": rk,
        "k": k,
        "n_queries": len(indices),
        "per_label_map": per_label_map,
        "distances_intra": distances_intra,
        "distances_inter": distances_inter,
        "ap_scores": ap_scores,
    }


# ── Distance analysis ────────────────────────────────────────────
def analyze_distances(db) -> dict:
    """
    Phân tích phân phối khoảng cách:
      - Tính tất cả khoảng cách pairwise (lấy mẫu nếu DB lớn)
      - So sánh intra-class vs inter-class distance
    """
    records = db._records
    n = len(records)
    MAX_SAMPLE = 300  # giới hạn để không quá chậm

    sample_idx = list(range(min(n, MAX_SAMPLE)))
    if n > MAX_SAMPLE:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n, size=MAX_SAMPLE, replace=False).tolist()

    vectors = np.stack([records[i]["vector"] for i in sample_idx])
    labels = [get_label(records[i]) for i in sample_idx]

    # Tính ma trận khoảng cách Euclidean
    # ||a-b||² = ||a||² + ||b||² - 2*a·b
    sq = np.sum(vectors ** 2, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2 * vectors @ vectors.T
    dist_sq = np.clip(dist_sq, 0, None)
    dist_matrix = np.sqrt(dist_sq)

    intra = []
    inter = []
    m = len(sample_idx)
    for i in range(m):
        for j in range(i + 1, m):
            d = dist_matrix[i, j]
            if labels[i] == labels[j]:
                intra.append(d)
            else:
                inter.append(d)

    return {
        "intra": np.array(intra),
        "inter": np.array(inter),
        "n_sampled": m,
    }


# ── Report ───────────────────────────────────────────────────────
def print_report(map_result: dict, dist_analysis: dict):
    k = map_result["k"]
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  DANH GIA HE THONG CBIR – mAP@{k}")
    print(sep)

    # Distance analysis
    intra = dist_analysis["intra"]
    inter = dist_analysis["inter"]
    print(f"\n[1] PHAN TICH KHOANG CACH (mau {dist_analysis['n_sampled']} anh)")
    print(f"  Intra-class (cung loai) : mean={intra.mean():.4f}  std={intra.std():.4f}  "
          f"min={intra.min():.4f}  max={intra.max():.4f}")
    print(f"  Inter-class (khac loai) : mean={inter.mean():.4f}  std={inter.std():.4f}  "
          f"min={inter.min():.4f}  max={inter.max():.4f}")

    separation = (inter.mean() - intra.mean()) / (inter.std() + intra.std() + 1e-8)
    print(f"  Fisher separation ratio : {separation:.4f}  "
          f"(>1.0 tot, >2.0 rat tot)")

    overlap_pct = np.sum(intra > inter.mean()) / max(len(intra), 1) * 100
    print(f"  Intra > inter mean      : {overlap_pct:.1f}%  "
          f"(thấp = feature tốt)")

    # Metrics
    print(f"\n[2] CHI SO TRUY VAN (tren {map_result['n_queries']} queries)")
    print(f"  mAP@{k:<2}             : {map_result['map_score']:.4f}  "
          f"{'★★★' if map_result['map_score'] > 0.7 else '★★' if map_result['map_score'] > 0.4 else '★'}")
    print(f"  Precision@1        : {map_result['precision_at_1']:.4f}  "
          f"(top-1 dung: {map_result['precision_at_1']*100:.1f}%)")
    print(f"  Precision@{k}        : {map_result['precision_at_k']:.4f}")
    print(f"  Recall@{k}           : {map_result['recall_at_k']:.4f}")

    # Per-label (sắp xếp từ thấp đến cao)
    print(f"\n[3] mAP THEO TUNG NHAN (ap thap = kho phan biet)")
    per = sorted(map_result["per_label_map"].items(), key=lambda x: x[1])
    for lbl, ap in per:
        bar = "█" * int(ap * 20)
        short_lbl = lbl[:45] if len(lbl) > 45 else lbl
        print(f"  {ap:.4f} |{bar:<20}| {short_lbl}")

    # Chẩn đoán
    print(f"\n[4] CHAN DOAN VA GOI Y")
    map_s = map_result["map_score"]
    sep_r = separation

    if map_s >= 0.7:
        print("  ✓ mAP cao – hệ thống hoạt động tốt.")
    elif map_s >= 0.4:
        print("  ~ mAP trung bình – cần cải thiện.")
    else:
        print("  ✗ mAP thấp – vector đặc trưng chưa phân biệt tốt các lớp.")

    if sep_r < 1.0:
        print("  ✗ Fisher ratio thấp: intra-class và inter-class distance chồng lấp nhiều.")
        print("    → Gợi ý: thử cosine distance, thêm đặc trưng, hoặc dùng metric learning.")

    if intra.mean() > 1.5:
        print(f"  ✗ Intra-class distance lớn (mean={intra.mean():.3f}):")
        print("    → Các ảnh cùng loại quá khác biệt trong không gian đặc trưng.")
        print("    → Kiểm tra lại: (a) chất lượng ảnh, (b) mask cây, (c) normalize.")

    if overlap_pct > 30:
        print(f"  ✗ {overlap_pct:.0f}% intra-dist > inter-dist mean: đặc trưng chưa discriminative.")
        print("    → Gợi ý: tăng trọng số nhóm màu/shape, giảm texture, hoặc dùng PCA.")

    # AP distribution
    aps = map_result["ap_scores"]
    low_ap = sum(1 for a in aps if a < 0.2)
    print(f"\n  Phan phoi AP@{k}: "
          f"<0.2: {low_ap} ({low_ap/len(aps)*100:.0f}%)  "
          f"0.2-0.5: {sum(1 for a in aps if 0.2<=a<0.5)} "
          f"  >=0.5: {sum(1 for a in aps if a>=0.5)}")

    print(f"\n{sep}\n")


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Danh gia mAP he thong CBIR")
    parser.add_argument("--k", type=int, default=5, help="So ket qua truy van (default: 5)")
    parser.add_argument("--sample", type=int, default=None,
                        help="So luong query mau (default: toan bo DB)")
    parser.add_argument("--no-dist", action="store_true",
                        help="Bo qua phan tich distance matrix (nhanh hon)")
    args = parser.parse_args()

    db, normalizer = load_system()

    # Phân tích khoảng cách
    if not args.no_dist:
        print("[Buoc 1] Phan tich khoang cach pairwise...")
        dist_analysis = analyze_distances(db)
    else:
        dist_analysis = {"intra": np.array([0.0]), "inter": np.array([1.0]), "n_sampled": 0}

    # Tính mAP
    print(f"\n[Buoc 2] Tinh mAP@{args.k}...")
    map_result = compute_map(db, k=args.k, sample_size=args.sample)

    # In báo cáo
    print_report(map_result, dist_analysis)


if __name__ == "__main__":
    main()
