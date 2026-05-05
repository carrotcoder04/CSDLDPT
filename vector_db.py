"""
vector_db.py
------------
Vector Database sử dụng cấu trúc cây nhị phân (KD-Tree) để lưu trữ
và truy vấn ảnh tương tự theo vector đặc trưng.

Kiến trúc:
    ┌─────────────────────────────────────────────────────┐
    │                  VectorDatabase                      │
    │  ┌──────────────┐   ┌───────────────────────────┐   │
    │  │ _BinaryNode  │   │  KDTree (tự cài / scipy)  │   │
    │  │  .point      │◄──│  Phân chia không gian D   │   │
    │  │  .left/right │   │  theo median mỗi chiều    │   │
    │  └──────────────┘   └───────────────────────────┘   │
    │  ┌──────────────────────────────────────────────┐    │
    │  │ records: list[dict]  (image_path, vector, …) │    │
    │  └──────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────┘

Các tính năng:
    - insert()      : Thêm một ảnh (vector + metadata) vào DB
    - build_tree()  : Xây dựng KD-Tree từ toàn bộ records
    - query()       : Tìm k ảnh gần nhất (kNN) bằng duyệt cây
    - save() / load(): Lưu & tải DB ra file .npz
    - stats()       : Thống kê DB

Tài liệu tham khảo: Nhóm 6 – Báo cáo ĐPT – Hệ CSDL Đa Phương Tiện
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("VectorDB")


# ─────────────────────────────────────────────
#  Node của KD-Tree
# ─────────────────────────────────────────────

@dataclass
class _KDNode:
    """
    Node của cây KD-Tree.

    Mỗi node lưu:
        - point       : Vector đặc trưng tại node này (centroid của nhánh)
        - record_idx  : Index trong danh sách records[] của VectorDatabase
        - axis        : Chiều phân chia tại node này
        - left/right  : Nhánh trái (≤ median) và phải (> median)
    """
    point: np.ndarray           # vector (D,)
    record_idx: int             # vị trí trong self._records
    axis: int                   # chiều phân chia
    left: Optional["_KDNode"] = field(default=None, repr=False)
    right: Optional["_KDNode"] = field(default=None, repr=False)


# ─────────────────────────────────────────────
#  KD-Tree thuần Python
# ─────────────────────────────────────────────

class _KDTree:
    """
    KD-Tree (K-Dimensional Tree) – cây nhị phân phân vùng không gian D chiều.

    Thuật toán xây dựng (O(N log N)):
        1. Chọn chiều axis = depth % D.
        2. Tính median theo chiều axis → phân chia điểm thành 2 nhóm.
        3. Đệ quy xây dựng nhánh trái (≤ median) và phải (> median).

    Thuật toán kNN (O(log N) trung bình):
        - Duyệt từ gốc, đi vào nhánh gần query hơn.
        - Backtrack và kiểm tra nhánh kia nếu có thể có điểm gần hơn.
        - Dùng heap (best-first) để duy trì k ứng viên tốt nhất.
    """

    def __init__(self) -> None:
        self._root: Optional[_KDNode] = None
        self._n_nodes: int = 0

    def build(self, points: np.ndarray, indices: np.ndarray) -> None:
        """
        Xây dựng KD-Tree từ tập điểm.

        Args:
            points:  numpy array (N, D) – các vector đặc trưng.
            indices: numpy array (N,)   – index tương ứng trong records[].
        """
        self._n_nodes = 0
        self._root = self._build_recursive(points, indices, depth=0)
        logger.debug(f"KD-Tree xay dung xong: {self._n_nodes} nodes")

    def knn_search(
        self, query: np.ndarray, k: int
    ) -> List[Tuple[float, int]]:
        """
        Tìm k điểm gần nhất với query (Euclidean distance).

        Args:
            query: numpy array (D,) – vector truy vấn.
            k:     Số kết quả cần trả về.

        Returns:
            list of (distance, record_idx) đã sắp xếp tăng dần theo khoảng cách.
        """
        # Heap lưu (dist, idx) – dùng danh sách tối đa k phần tử
        best: List[Tuple[float, int]] = []
        self._search_recursive(self._root, query, k, best)
        best.sort(key=lambda x: x[0])
        return best

    # ── Xây dựng đệ quy ──────────────────────────────────

    def _build_recursive(
        self,
        points: np.ndarray,
        indices: np.ndarray,
        depth: int,
    ) -> Optional[_KDNode]:
        if len(points) == 0:
            return None

        D = points.shape[1]
        # Chọn chiều có phương sai lớn nhất (ổn định hơn depth % D)
        axis = int(np.argmax(np.var(points, axis=0))) if len(points) > 1 else depth % D

        # Sắp xếp theo chiều axis → lấy điểm median làm gốc nhánh
        sorted_order = np.argsort(points[:, axis])
        points = points[sorted_order]
        indices = indices[sorted_order]

        mid = len(points) // 2

        node = _KDNode(
            point=points[mid].copy(),
            record_idx=int(indices[mid]),
            axis=axis,
        )
        self._n_nodes += 1

        node.left = self._build_recursive(
            points[:mid], indices[:mid], depth + 1
        )
        node.right = self._build_recursive(
            points[mid + 1:], indices[mid + 1:], depth + 1
        )
        return node

    # ── Tìm kiếm đệ quy ──────────────────────────────────

    def _search_recursive(
        self,
        node: Optional[_KDNode],
        query: np.ndarray,
        k: int,
        best: List[Tuple[float, int]],
    ) -> None:
        if node is None:
            return

        # Khoảng cách từ query đến node hiện tại
        dist = float(np.linalg.norm(query - node.point))

        # Cập nhật danh sách k ứng viên tốt nhất
        if len(best) < k:
            best.append((dist, node.record_idx))
            best.sort(key=lambda x: x[0])
        elif dist < best[-1][0]:
            best[-1] = (dist, node.record_idx)
            best.sort(key=lambda x: x[0])

        # Quyết định đi vào nhánh nào trước
        axis = node.axis
        diff = query[axis] - node.point[axis]  # dương → phải, âm → trái

        near_branch = node.left if diff <= 0 else node.right
        far_branch = node.right if diff <= 0 else node.left

        # Duyệt nhánh gần trước
        self._search_recursive(near_branch, query, k, best)

        # Kiểm tra nhánh xa: nếu khoảng cách trục < khoảng cách xa nhất hiện tại
        worst_dist = best[-1][0] if len(best) == k else float("inf")
        if abs(diff) < worst_dist:
            self._search_recursive(far_branch, query, k, best)


# ─────────────────────────────────────────────
#  Vector Database
# ─────────────────────────────────────────────

class VectorDatabase:
    """
    Cơ sở dữ liệu vector ảnh sử dụng KD-Tree để truy vấn nhanh.

    Workflow:
        db = VectorDatabase()
        db.insert("leaf1.jpg", vector1, label="Quercus")
        db.insert("leaf2.jpg", vector2, label="Acer")
        db.build_tree()                          # Xây cây sau khi insert xong
        results = db.query(query_vector, k=5)    # Tìm 5 ảnh gần nhất
    """

    def __init__(self, distance: str = "euclidean") -> None:
        """
        Khởi tạo VectorDatabase.

        Args:
            distance: Độ đo khoảng cách: 'euclidean' | 'cosine'.
                      'cosine' → vector phải được L2-normalize trước.
        """
        if distance not in ("euclidean", "cosine"):
            raise ValueError(f"Distance khong hop le: '{distance}'")

        self.distance = distance
        self._records: List[Dict[str, Any]] = []   # [{image_path, vector, label, ...}]
        self._kdtree: _KDTree = _KDTree()
        self._tree_built: bool = False
        self._n_features: Optional[int] = None

    # ─────────────────────────────────────────────
    #  Thêm dữ liệu
    # ─────────────────────────────────────────────

    def insert(
        self,
        image_path: str,
        vector: np.ndarray,
        label: Optional[str] = None,
        **metadata,
    ) -> int:
        """
        Thêm một ảnh và vector đặc trưng vào DB.

        Args:
            image_path: Đường dẫn ảnh (dùng làm định danh).
            vector:     Vector đặc trưng shape (D,).
            label:      Nhãn loài (tuỳ chọn).
            **metadata: Bất kỳ metadata nào thêm (vd: processing_time_ms).

        Returns:
            int: Index của record vừa thêm.
        """
        vector = np.asarray(vector, dtype=np.float32).flatten()
        
        if self.distance == "cosine":
            norm = np.linalg.norm(vector)
            vector = (vector / (norm + 1e-8)).astype(np.float32)

        if self._n_features is None:
            self._n_features = len(vector)
        elif len(vector) != self._n_features:
            raise ValueError(
                f"Kich thuoc vector khong khop: du kien {self._n_features}, "
                f"nhan {len(vector)}."
            )

        record = {
            "image_path": str(image_path),
            "vector": vector,
            "label": label,
            **metadata,
        }
        idx = len(self._records)
        self._records.append(record)
        self._tree_built = False  # cây cần rebuild

        logger.debug(f"Insert [{idx}]: {Path(image_path).name} | label={label}")
        return idx

    def insert_batch(
        self,
        image_paths: List[str],
        vectors: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> None:
        """
        Thêm nhiều ảnh cùng lúc.

        Args:
            image_paths: Danh sách đường dẫn ảnh.
            vectors:     numpy array (N, D).
            labels:      Danh sách nhãn (tuỳ chọn).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if len(image_paths) != len(vectors):
            raise ValueError("image_paths va vectors phai co cung do dai.")

        if labels is None:
            labels = [None] * len(image_paths)

        for path, vec, lbl in zip(image_paths, vectors, labels):
            self.insert(path, vec, label=lbl)

        logger.info(f"Insert batch: {len(image_paths)} anh.")

    # ─────────────────────────────────────────────
    #  Xây cây KD-Tree
    # ─────────────────────────────────────────────

    def build_tree(self) -> None:
        """
        Xây dựng KD-Tree từ toàn bộ records hiện có.

        Phải gọi sau khi insert xong và trước khi query.
        Nếu insert thêm record mới → cần gọi lại build_tree().
        """
        if len(self._records) == 0:
            logger.warning("DB rong, khong co gi de build.")
            return

        t0 = time.perf_counter()

        points = np.stack([r["vector"] for r in self._records], axis=0)
        indices = np.arange(len(self._records))

        self._kdtree.build(points, indices)
        self._tree_built = True

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"KD-Tree build xong: {len(self._records)} records, "
            f"{self._n_features} chieu, {elapsed:.1f} ms"
        )

    # ─────────────────────────────────────────────
    #  Truy vấn
    # ─────────────────────────────────────────────

    def query(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        return_distances: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Tìm k ảnh gần nhất với query_vector trong DB.

        Args:
            query_vector:     Vector truy vấn shape (D,).
            k:                Số kết quả trả về.
            return_distances: Nếu True, thêm trường 'distance' vào kết quả.

        Returns:
            list of dict, mỗi dict gồm:
                - image_path : đường dẫn ảnh
                - label      : nhãn loài
                - distance   : khoảng cách đến query (nếu return_distances=True)
                - rank       : thứ hạng (1 = gần nhất)
                - + các metadata khác
        """
        if not self._tree_built:
            raise RuntimeError(
                "Chua build KD-Tree. Vui long goi build_tree() truoc khi query()."
            )

        query_vector = np.asarray(query_vector, dtype=np.float32).flatten()

        if self._n_features and len(query_vector) != self._n_features:
            raise ValueError(
                f"Kich thuoc query khong khop: du kien {self._n_features}, "
                f"nhan {len(query_vector)}."
            )

        k = min(k, len(self._records))

        if self.distance == "cosine":
            # Cosine distance = 1 - cosine_similarity
            # Khi vector đã L2-norm → cosine_sim = dot product
            # Đổi sang khoảng cách Euclidean: ||u-v||² = 2(1 - u·v) khi ||u||=||v||=1
            # Dùng Euclidean trên vector đã L2-norm → tương đương cosine
            q = query_vector.astype(np.float64)
            norm = np.linalg.norm(q)
            q = q / (norm + 1e-8)
            query_f32 = q.astype(np.float32)
        else:
            query_f32 = query_vector

        t0 = time.perf_counter()
        hits = self._kdtree.knn_search(query_f32, k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = []
        for rank, (dist, rec_idx) in enumerate(hits, start=1):
            rec = self._records[rec_idx]
            entry = {
                "rank": rank,
                "image_path": rec["image_path"],
                "label": rec.get("label"),
            }
            if return_distances:
                entry["distance"] = round(float(dist), 6)
            # Thêm metadata phụ (bỏ qua 'vector' để không trả về raw data lớn)
            for k_meta, v_meta in rec.items():
                if k_meta not in ("vector", "image_path", "label"):
                    entry[k_meta] = v_meta
            results.append(entry)

        logger.info(
            f"Query xong: k={k}, dist={self.distance}, "
            f"{elapsed_ms:.2f} ms | top1={results[0]['image_path'] if results else 'N/A'}"
        )
        return results

    def query_by_path(
        self,
        image_path: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Truy vấn ảnh tương tự bằng đường dẫn của một ảnh đã có trong DB.

        Args:
            image_path: Đường dẫn ảnh đã được insert.
            k:          Số kết quả (k+1 vì bản thân ảnh sẽ là kết quả đầu tiên).

        Returns:
            list kết quả (loại trừ chính ảnh query).
        """
        # Tìm record theo path
        rec = next(
            (r for r in self._records if r["image_path"] == str(image_path)),
            None,
        )
        if rec is None:
            raise ValueError(f"Anh '{image_path}' chua duoc insert vao DB.")

        results = self.query(rec["vector"], k=k + 1)
        # Loại bỏ chính ảnh query
        results = [r for r in results if r["image_path"] != str(image_path)]
        return results[:k]

    # ─────────────────────────────────────────────
    #  Lưu / Tải
    # ─────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """
        Lưu toàn bộ DB ra file .npz.

        Args:
            path: Đường dẫn file đầu ra.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if len(self._records) == 0:
            logger.warning("DB rong, khong co gi de luu.")
            return

        vectors = np.stack([r["vector"] for r in self._records])
        image_paths = np.array([r["image_path"] for r in self._records])
        labels = np.array(
            [r["label"] if r["label"] is not None else "" for r in self._records]
        )

        np.savez_compressed(
            path,
            vectors=vectors,
            image_paths=image_paths,
            labels=labels,
            distance=np.array([self.distance]),
            n_features=np.array([self._n_features or -1]),
        )
        logger.info(f"Da luu VectorDB: {len(self._records)} records → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VectorDatabase":
        """
        Tải VectorDB từ file .npz và tự động build KD-Tree.

        Args:
            path: Đường dẫn file .npz.

        Returns:
            VectorDatabase đã sẵn sàng query.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay file: {path}")

        data = np.load(path, allow_pickle=True)
        distance = str(data["distance"][0]) if "distance" in data else "euclidean"
        db = cls(distance=distance)

        if "n_features" in data:
            db._n_features = int(data["n_features"][0])

        vectors = data["vectors"]
        image_paths = data["image_paths"]
        labels = data["labels"]

        for vec, img_path, lbl in zip(vectors, image_paths, labels):
            db.insert(
                image_path=str(img_path),
                vector=vec,
                label=str(lbl) if lbl != "" else None,
            )

        db.build_tree()
        logger.info(f"Da tai VectorDB tu: {path} | {len(db._records)} records")
        return db

    # ─────────────────────────────────────────────
    #  Thông tin
    # ─────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Trả về dict thống kê trạng thái DB."""
        labels = [r["label"] for r in self._records if r.get("label")]
        unique_labels = list(set(labels))
        return {
            "n_records": len(self._records),
            "n_features": self._n_features,
            "tree_built": self._tree_built,
            "n_tree_nodes": self._kdtree._n_nodes,
            "distance_metric": self.distance,
            "n_labels": len(unique_labels),
            "labels": sorted(unique_labels),
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"VectorDatabase("
            f"n_records={len(self._records)}, "
            f"n_features={self._n_features}, "
            f"distance='{self.distance}', "
            f"tree_built={self._tree_built})"
        )


# ─────────────────────────────────────────────
#  Chạy thử nghiệm
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import random

    print("=== Thu nghiem VectorDatabase (KD-Tree) ===\n")

    rng = np.random.default_rng(0)
    N, D = 200, 62  # 62 chieu = thuc te cua TreeFeatureExtractor

    db = VectorDatabase(distance="euclidean")

    species = ["Quercus", "Acer", "Betula", "Fagus", "Pinus"]
    for i in range(N):
        vec = rng.random(D).astype(np.float32)
        sp = random.choice(species)
        db.insert(f"leaf_{i:04d}.jpg", vec, label=sp)

    print(f"Records da insert: {len(db)}")
    db.build_tree()
    print(f"Stats sau build: {db.stats()}\n")

    # Truy vấn
    query_vec = rng.random(D).astype(np.float32)
    results = db.query(query_vec, k=5)

    print("Top-5 ket qua truy van:")
    for r in results:
        print(f"  #{r['rank']} {r['image_path']:20s} | {r['label']:10s} | dist={r['distance']:.4f}")

    # Lưu và tải lại
    db.save("test_db.npz")
    db2 = VectorDatabase.load("test_db.npz")
    print(f"\nSau khi load lai: {db2}")
    r2 = db2.query(query_vec, k=3)
    print("Top-3 sau khi load:")
    for r in r2:
        print(f"  #{r['rank']} {r['image_path']:20s} | dist={r['distance']:.4f}")

    import os
    os.remove("test_db.npz")
    print("\nHoan thanh thu nghiem.")
