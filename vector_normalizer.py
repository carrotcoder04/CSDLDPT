"""
vector_normalizer.py
--------------------
Module chuẩn hóa vector đặc trưng sau khi trích rút (post-extraction normalization).

Hỗ trợ 3 phương pháp chuẩn hóa:
    1. L2 Normalization   : vector / ||vector||₂  → độ dài = 1 (dùng cho cosine similarity)
    2. Z-score (Standard) : (x − μ) / σ per-dim   → phân phối chuẩn N(0,1)
    3. Min-Max Scaling    : (x − min) / (max − min) → về khoảng [0, 1]

Thiết kế:
    - VectorNormalizer có thể ``fit()`` trên tập dữ liệu để học các tham số
      (mean, std, min, max per-dimension).
    - Sau khi fit, dùng ``transform()`` để chuẩn hóa vector mới.
    - ``fit_transform()`` = fit + transform trong một bước.
    - Hỗ trợ ``save()`` / ``load()`` tham số chuẩn hóa ra/vào file JSON/NPZ.

Sử dụng:
    norm = VectorNormalizer(method="zscore")
    norm.fit(vectors_train)          # học tham số từ tập train
    v_norm = norm.transform(vector)  # chuẩn hóa vector mới

Tài liệu tham khảo: Nhóm 6 – Báo cáo ĐPT – Hệ CSDL Đa Phương Tiện
"""

import json
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

logger = logging.getLogger("VectorNormalizer")

# Kiểu chuẩn hóa hợp lệ
NormMethod = Literal["l2", "zscore", "minmax"]


class VectorNormalizer:
    """
    Chuẩn hóa vector đặc trưng đa chiều sau bước feature extraction.

    Attributes:
        method (str):       Phương pháp chuẩn hóa: 'l2' | 'zscore' | 'minmax'.
        eps (float):        Hằng số nhỏ tránh chia cho 0.
        is_fitted (bool):   True sau khi đã gọi fit().
    """

    SUPPORTED_METHODS: tuple = ("l2", "zscore", "minmax")

    def __init__(
        self,
        method: NormMethod = "zscore",
        eps: float = 1e-8,
    ) -> None:
        """
        Khởi tạo VectorNormalizer.

        Args:
            method: Phương pháp chuẩn hóa.
                    - 'l2'     : Chia theo chuẩn L2 của từng vector (không cần fit).
                    - 'zscore' : Chuẩn hóa theo mean/std per-dimension (cần fit).
                    - 'minmax' : Scale về [0,1] theo min/max per-dimension (cần fit).
            eps:    Hằng số nhỏ để tránh chia cho 0 khi std hoặc range ≈ 0.
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Phuong phap khong hop le: '{method}'. "
                f"Ho tro: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.eps = eps
        self.is_fitted = False

        # Các tham số học được (chỉ dùng cho zscore / minmax)
        self._mean: Optional[np.ndarray] = None   # shape (D,)
        self._std: Optional[np.ndarray] = None    # shape (D,)
        self._min: Optional[np.ndarray] = None    # shape (D,)
        self._max: Optional[np.ndarray] = None    # shape (D,)
        self._n_features: Optional[int] = None    # số chiều D
        self._n_samples_fitted: int = 0           # số vector dùng để fit

    # ─────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────

    def fit(self, vectors: np.ndarray) -> "VectorNormalizer":
        """
        Học tham số chuẩn hóa từ tập vector huấn luyện.

        Args:
            vectors: numpy array shape (N, D) – N vector, mỗi vector D chiều.
                     Cũng chấp nhận shape (D,) cho 1 vector đơn.

        Returns:
            self (cho phép method chaining: norm.fit(X).transform(X))
        """
        vectors = self._validate_input(vectors, allow_1d=False)

        self._n_features = vectors.shape[1]
        self._n_samples_fitted = vectors.shape[0]

        if self.method == "l2":
            # L2 không cần fit (tự normalize từng vector độc lập)
            self.is_fitted = True
            logger.info("VectorNormalizer [l2] fit() – khong can tham so, san sang.")
            return self

        if self.method == "zscore":
            self._mean = np.mean(vectors, axis=0).astype(np.float64)
            self._std = np.std(vectors, axis=0).astype(np.float64)
            logger.info(
                f"VectorNormalizer [zscore] fit() – {self._n_samples_fitted} mau, "
                f"{self._n_features} chieu | "
                f"mean=[{self._mean.min():.4f},{self._mean.max():.4f}] "
                f"std=[{self._std.min():.4f},{self._std.max():.4f}]"
            )

        elif self.method == "minmax":
            self._min = np.min(vectors, axis=0).astype(np.float64)
            self._max = np.max(vectors, axis=0).astype(np.float64)
            logger.info(
                f"VectorNormalizer [minmax] fit() – {self._n_samples_fitted} mau, "
                f"{self._n_features} chieu | "
                f"min=[{self._min.min():.4f}] max=[{self._max.max():.4f}]"
            )

        self.is_fitted = True
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa vector (hoặc batch vector) bằng tham số đã fit.

        Args:
            vectors: numpy array shape (D,) hoặc (N, D).
        Returns:
            numpy array cùng shape, dtype float32.
        Raises:
            RuntimeError: Nếu chưa gọi fit() (với zscore/minmax).
        """
        is_1d = vectors.ndim == 1
        vectors = self._validate_input(vectors, allow_1d=True)

        if self.method in ("zscore", "minmax") and not self.is_fitted:
            raise RuntimeError(
                "Chua goi fit(). Vui long goi fit(train_vectors) truoc khi transform()."
            )

        if self._n_features is not None and vectors.shape[-1] != self._n_features:
            raise ValueError(
                f"So chieu khong khop: du kien {self._n_features}, "
                f"nhan {vectors.shape[-1]}."
            )

        v = vectors.astype(np.float64)

        if self.method == "l2":
            result = self._l2_normalize(v)
        elif self.method == "zscore":
            result = self._zscore_normalize(v)
        else:  # minmax
            result = self._minmax_normalize(v)

        result = result.astype(np.float32)
        return result[0] if is_1d else result

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Học tham số và chuẩn hóa ngay trên cùng tập dữ liệu.

        Args:
            vectors: numpy array shape (N, D).

        Returns:
            numpy array shape (N, D), dtype float32.
        """
        return self.fit(vectors).transform(vectors)

    def transform_one(self, vector: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa một vector đơn shape (D,).

        Shortcut cho transform() với input 1D.

        Args:
            vector: numpy array shape (D,).

        Returns:
            numpy array shape (D,), dtype float32.
        """
        if vector.ndim != 1:
            raise ValueError(f"transform_one() chi nhan vector 1D, nhan {vector.ndim}D.")
        return self.transform(vector)

    # ─────────────────────────────────────────────
    #  Lưu / Tải tham số
    # ─────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """
        Lưu tham số chuẩn hóa ra file .npz.

        Args:
            path: Đường dẫn file đầu ra (nên có đuôi .npz).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {
            "method": np.array([self.method]),
            "eps": np.array([self.eps]),
            "is_fitted": np.array([self.is_fitted]),
            "n_features": np.array([self._n_features if self._n_features else -1]),
            "n_samples_fitted": np.array([self._n_samples_fitted]),
        }

        if self._mean is not None:
            arrays["mean"] = self._mean
        if self._std is not None:
            arrays["std"] = self._std
        if self._min is not None:
            arrays["min"] = self._min
        if self._max is not None:
            arrays["max"] = self._max

        np.savez_compressed(path, **arrays)
        logger.info(f"Da luu tham so chuan hoa vao: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VectorNormalizer":
        """
        Tải tham số chuẩn hóa từ file .npz đã lưu.

        Args:
            path: Đường dẫn file .npz.

        Returns:
            VectorNormalizer đã được fit sẵn.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay file: {path}")

        data = np.load(path, allow_pickle=True)
        method = str(data["method"][0])
        eps = float(data["eps"][0])

        norm = cls(method=method, eps=eps)
        norm.is_fitted = bool(data["is_fitted"][0])
        norm._n_features = int(data["n_features"][0])
        if norm._n_features == -1:
            norm._n_features = None
        norm._n_samples_fitted = int(data["n_samples_fitted"][0])

        if "mean" in data:
            norm._mean = data["mean"]
        if "std" in data:
            norm._std = data["std"]
        if "min" in data:
            norm._min = data["min"]
        if "max" in data:
            norm._max = data["max"]

        logger.info(f"Da tai tham so chuan hoa tu: {path} | method={method}")
        return norm

    # ─────────────────────────────────────────────
    #  Thông tin debug
    # ─────────────────────────────────────────────

    def info(self) -> dict:
        """Trả về dict thông tin về trạng thái normalizer."""
        return {
            "method": self.method,
            "eps": self.eps,
            "is_fitted": self.is_fitted,
            "n_features": self._n_features,
            "n_samples_fitted": self._n_samples_fitted,
        }

    def __repr__(self) -> str:
        return (
            f"VectorNormalizer("
            f"method='{self.method}', "
            f"fitted={self.is_fitted}, "
            f"n_features={self._n_features})"
        )

    # ─────────────────────────────────────────────
    #  Các phương pháp chuẩn hóa nội bộ
    # ─────────────────────────────────────────────

    def _l2_normalize(self, v: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa L2: v / ||v||₂.

        Mỗi vector được chia cho chuẩn L2 của chính nó → độ dài = 1.
        Dùng cho cosine similarity (dot product = cosine sau khi L2-norm).

        v shape: (N, D) hoặc (1, D)
        """
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / (norms + self.eps)

    def _zscore_normalize(self, v: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa Z-score: (x − μ) / σ per-dimension.

        Chiều nào σ ≈ 0 (constant feature) → giữ nguyên (chia (σ + eps)).
        """
        return (v - self._mean) / (self._std + self.eps)

    def _minmax_normalize(self, v: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa Min-Max: (x − min) / (max − min) per-dimension.

        Kết quả nằm trong khoảng [0, 1].
        Chiều nào max ≈ min → giữ nguyên (chia (range + eps)).
        """
        range_ = self._max - self._min
        return (v - self._min) / (range_ + self.eps)

    # ─────────────────────────────────────────────
    #  Tiện ích
    # ─────────────────────────────────────────────

    @staticmethod
    def _validate_input(
        vectors: np.ndarray, allow_1d: bool = True
    ) -> np.ndarray:
        """
        Kiểm tra và chuẩn hóa đầu vào thành numpy array float64.

        Args:
            vectors:   Input (list, ndarray, v.v.)
            allow_1d:  Nếu True, chấp nhận shape (D,) và reshape thành (1, D).

        Returns:
            numpy array shape (N, D) hoặc (D,) tùy allow_1d.
        """
        vectors = np.asarray(vectors, dtype=np.float64)

        if vectors.ndim == 1:
            if allow_1d:
                return vectors.reshape(1, -1)
            raise ValueError(
                "Input phai la 2D array (N, D). "
                "Neu co 1 vector, truyen vao dang [[v1, v2, ...]]."
            )

        if vectors.ndim != 2:
            raise ValueError(
                f"Input phai la 2D array (N, D), nhan {vectors.ndim}D."
            )

        return vectors


# ─────────────────────────────────────────────
#  Chạy thử nghiệm
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Thu nghiem VectorNormalizer ===\n")

    # Tao du lieu mau
    rng = np.random.default_rng(42)
    X_train = rng.random((100, 52)).astype(np.float32)  # 100 mau, 52 chieu
    X_test = rng.random((5, 52)).astype(np.float32)

    for method in ["l2", "zscore", "minmax"]:
        print(f"--- Method: {method} ---")
        norm = VectorNormalizer(method=method)
        X_norm = norm.fit_transform(X_train)
        print(f"  Shape sau norm   : {X_norm.shape}")
        print(f"  Min/Max          : {X_norm.min():.4f} / {X_norm.max():.4f}")

        if method == "l2":
            norms = np.linalg.norm(X_norm, axis=1)
            print(f"  L2 norms (nen=1) : min={norms.min():.4f} max={norms.max():.4f}")
        elif method == "zscore":
            print(f"  Mean (nen=0)     : {X_norm.mean():.4f}")
            print(f"  Std  (nen=1)     : {X_norm.std():.4f}")
        elif method == "minmax":
            print(f"  Min  (nen=0)     : {X_norm.min():.4f}")
            print(f"  Max  (nen=1)     : {X_norm.max():.4f}")

        # Test transform mot vector moi
        v_new = norm.transform_one(X_test[0])
        print(f"  Transform 1 vec  : shape={v_new.shape}, dtype={v_new.dtype}")
        print()
