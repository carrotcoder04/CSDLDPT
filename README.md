# 🌳 Tree Feature Extractor

Hệ thống trích rút đặc trưng **ảnh toàn cây** tự động, phục vụ bài toán nhận dạng và tra cứu ảnh cây trong **Hệ Cơ Sở Dữ Liệu Đa Phương Tiện**.

> **Nhóm 6 – Báo cáo ĐPT – Hệ CSDL Đa Phương Tiện**

---

## 📦 Cài đặt thư viện

```bash
pip install -r requirements.txt
```

| Thư viện        | Phiên bản tối thiểu | Vai trò                                      |
| --------------- | ------------------- | -------------------------------------------- |
| `opencv-python` | ≥ 4.8.0             | Xử lý ảnh, tìm contour, morphology           |
| `numpy`         | ≥ 1.24.0            | Tính toán ma trận, thống kê                  |
| `scikit-learn`  | ≥ 1.3.0             | KMeans phân cụm màu chủ đạo                  |
| `matplotlib`    | ≥ 3.7.0             | Visualize kết quả (chỉ dùng trong `demo.py`) |

---

## 🗂️ Cấu trúc thư mục

```
CSDLDPT/
├── main.py                     # Script chạy pipeline hoàn chỉnh (Build DB & Query)
├── feature_extractor.py        # Lớp chính TreeFeatureExtractor
├── vector_normalizer.py        # Chuẩn hóa vector đặc trưng (Z-score, L2, MinMax)
├── vector_db.py                # CSDL Vector (KD-Tree) dùng cho tra cứu nhanh kNN
├── demo.py                     # Script visualize kết quả trực quan
├── requirements.txt
├── tree/                       # Thư mục ảnh cây (mỗi loài = 1 thư mục con)
│   ├── Ginkgo biloba_.../
│   ├── Acer palmatum_.../
│   └── ...
└── features/
    ├── mask_utils.py           # Tạo mặt nạ cây dùng chung (Otsu + flood-fill)
    ├── color_features.py       # Đặc trưng màu sắc   (24 chiều)
    ├── shape_features.py       # Đặc trưng hình thái  (12 chiều)
    ├── texture_features.py     # Đặc trưng kết cấu    (18 chiều)
    └── canopy_features.py      # Đặc trưng tán cây    ( 9 chiều)
```

---

## ⚡ Sử dụng nhanh

### Trích rút 1 ảnh cây

```python
from feature_extractor import TreeFeatureExtractor

extractor = TreeFeatureExtractor()
result = extractor.extract("tree/Ginkgo.../image.png")

print(result["n_features"])          # Số chiều đặc trưng (62)
print(result["vector"])              # numpy array float32, shape (62,)
print(result["features"])            # dict tên → giá trị
print(result["processing_time_ms"])  # Thời gian xử lý (ms)
```

### Trích rút cả thư mục (batch)

```python
extractor = TreeFeatureExtractor()
results = extractor.extract_batch(
    image_dir="tree/",
    save_json="features.json"   # Tùy chọn: lưu kết quả ra JSON
)
print(f"Đã xử lý {len(results)} ảnh")
```

### Chỉ trích rút một số nhóm đặc trưng

```python
# Chỉ màu sắc và hình thái
extractor = TreeFeatureExtractor(
    enabled_groups=["color", "shape"]
)
```

### Chạy từ command line (Pipeline Toàn Tập)

```bash
# Xây dựng Vector DB từ thư mục ảnh cây (zscore mặc định)
python main.py --build --image_dir tree/

# Xây dựng DB với chuẩn hóa L2
python main.py --build --image_dir tree/ --norm l2

# Truy vấn k=5 ảnh cây tương tự
python main.py --query path/to/tree.png --k 5
```

Các lệnh test lẻ:

```bash
# Trích rút 1 ảnh
python feature_extractor.py path/to/tree.png

# Visualize kết quả trích rút
python demo.py path/to/tree.png
```

---

## 🔬 Mô tả chi tiết các đặc trưng

Tổng cộng **~62 đặc trưng** chia thành 4 nhóm. Tất cả đều là số thực (`float32`).
Tên đặc trưng trong `result["features"]` có dạng `{nhóm}_{tên}`.

---

### 🎨 Nhóm 1: Màu sắc (`color_*`) — 24 chiều

Trích rút từ không gian màu **HSV**, tập trung vào vùng cây (bỏ nền khi có thể).

#### 1.1 Hue Histogram — 8 chiều

| Tên đặc trưng      | Giá trị | Ý nghĩa                         |
| ------------------ | ------- | ------------------------------- |
| `color_hue_hist_0` | [0, 1]  | Tỷ lệ pixel Hue [0°–22.5°) → đỏ |
| `color_hue_hist_1` | [0, 1]  | [22.5°–45°) → cam               |
| `color_hue_hist_2` | [0, 1]  | [45°–67.5°) → vàng              |
| `color_hue_hist_3` | [0, 1]  | [67.5°–90°) → vàng-xanh         |
| `color_hue_hist_4` | [0, 1]  | [90°–112.5°) → xanh lá          |
| `color_hue_hist_5` | [0, 1]  | [112.5°–135°) → xanh lá đậm     |
| `color_hue_hist_6` | [0, 1]  | [135°–157.5°) → lục lam         |
| `color_hue_hist_7` | [0, 1]  | [157.5°–180°) → lam             |

#### 1.2 Thống kê kênh HSV — 6 chiều

| Tên đặc trưng  | Đơn vị   | Ý nghĩa                                          |
| -------------- | -------- | ------------------------------------------------ |
| `color_h_mean` | [0, 180] | **Circular mean** Hue (tránh sai số wrap-around) |
| `color_h_std`  | ≥ 0      | Circular std Hue — cao → màu đa dạng             |
| `color_s_mean` | [0, 255] | Trung bình độ bão hòa                            |
| `color_s_std`  | ≥ 0      | Độ biến thiên độ bão hòa                         |
| `color_v_mean` | [0, 255] | Trung bình độ sáng                               |
| `color_v_std`  | ≥ 0      | Độ biến thiên độ sáng                            |

#### 1.3 Màu chủ đạo (KMeans) — 9 chiều

| Tên đặc trưng                                  | Giá trị | Ý nghĩa                        |
| ---------------------------------------------- | ------- | ------------------------------ |
| `color_dom_r1`, `color_dom_g1`, `color_dom_b1` | [0, 1]  | Màu chủ đạo #1 (phổ biến nhất) |
| `color_dom_r2`, `color_dom_g2`, `color_dom_b2` | [0, 1]  | Màu chủ đạo #2                 |
| `color_dom_r3`, `color_dom_g3`, `color_dom_b3` | [0, 1]  | Màu chủ đạo #3                 |

#### 1.4 Tỷ lệ màu xanh lá — 1 chiều

| Tên đặc trưng       | Giá trị | Ý nghĩa                                                                           |
| ------------------- | ------- | --------------------------------------------------------------------------------- |
| `color_green_ratio` | [0, 1]  | Tỷ lệ pixel xanh lá (Hue ∈ [35°, 85°]). Phân biệt cây lá xanh vs cây khô/mùa đông |

---

### 📐 Nhóm 2: Hình thái (`shape_*`) — 12 chiều

Phân tích hình dạng tổng thể tán cây từ mặt nạ và contour.

| Tên đặc trưng           | Giá trị    | Công thức / Ý nghĩa                                      |
| ----------------------- | ---------- | -------------------------------------------------------- |
| `shape_aspect_ratio`    | > 0        | `W / H` bounding box — phân biệt cây cao vs cây bụi      |
| `shape_extent_ratio`    | [0, 1]     | `Area / BoundingBoxArea` — mật độ tán trong bounding box |
| `shape_area_ratio`      | [0, 1]     | `Area / ImageArea` — cây chiếm bao nhiêu % ảnh           |
| `shape_solidity`        | [0, 1]     | `Area / ConvexHullArea` — tán dày đặc (1) vs thưa (thấp) |
| `shape_centroid_y_norm` | [0, 1]     | Vị trí trọng tâm theo chiều dọc (0 = trên, 1 = dưới)     |
| `shape_centroid_x_norm` | [0, 1]     | Vị trí trọng tâm theo chiều ngang (0.5 = giữa)           |
| `shape_symmetry`        | [0, 1]     | Độ đối xứng trái/phải tán (IoU hai nửa)                  |
| `shape_crown_ratio`     | [0, 1]     | Tỷ lệ pixel ở nửa trên — tán cao (> 0.5) vs tán thấp     |
| `shape_hu_0`            | log-scaled | Hu Moment #0: phân bố khối lượng hình học                |
| `shape_hu_1`            | log-scaled | Hu Moment #1: độ kéo dài                                 |
| `shape_hu_2`            | log-scaled | Hu Moment #2: mất cân đối                                |
| `shape_hu_3`            | log-scaled | Hu Moment #3: độ cong                                    |

> Hu Moments bất biến với tịnh tiến, tỷ lệ và xoay.

---

### 🔲 Nhóm 3: Kết cấu (`texture_*`) — 17 chiều

Phân tích bề mặt kết cấu tán cây (mịn/thô/dạng kim...).

#### 3.1 LBP Histogram — 10 chiều

| Tên đặc trưng                      | Giá trị | Ý nghĩa                                                |
| ---------------------------------- | ------- | ------------------------------------------------------ |
| `texture_lbp_0` .. `texture_lbp_9` | [0, 1]  | Phân bố kết cấu cục bộ (Local Binary Pattern, 10 bins) |

> LBP cây lá rộng: histogram tập trung ở uniform patterns.
> LBP cây lá kim: histogram phân tán, nhiều pattern cạnh.

#### 3.2 GLCM Statistics — 4 chiều

| Tên đặc trưng         | Giá trị | Ý nghĩa                                 |
| --------------------- | ------- | --------------------------------------- |
| `texture_contrast`    | ≥ 0     | Độ tương phản cục bộ (cây lá kim → cao) |
| `texture_homogeneity` | [0, 1]  | Độ đồng nhất kết cấu (tán mịn → cao)    |
| `texture_energy`      | [0, 1]  | Năng lượng kết cấu (đều → cao)          |
| `texture_correlation` | [-1, 1] | Tương quan pixel liền kề                |

#### 3.3 Gradient Statistics — 2 chiều

| Tên đặc trưng       | Ý nghĩa                                       |
| ------------------- | --------------------------------------------- |
| `texture_grad_mean` | Cường độ cạnh trung bình (tán phức tạp → cao) |
| `texture_grad_std`  | Độ lệch chuẩn cạnh                            |

#### 3.4 Roughness — 1 chiều

| Tên đặc trưng       | Ý nghĩa                                    |
| ------------------- | ------------------------------------------ |
| `texture_roughness` | Độ nhám bề mặt tán (Local Std, 5×5 window) |

> Cây dạng lá kim (Cedrus, Salix): roughness cao hơn cây lá rộng mịn.

---

### 🌲 Nhóm 4: Tán cây (`canopy_*`) — 9 chiều

Phân tích cấu trúc hình học tán cây từ phân bố pixel và contour.

#### 4.1 Vertical Profile — 3 chiều

| Tên đặc trưng           | Giá trị | Ý nghĩa                                             |
| ----------------------- | ------- | --------------------------------------------------- |
| `canopy_peak_row_norm`  | [0, 1]  | Vị trí dải ngang có nhiều pixel nhất (tán dày nhất) |
| `canopy_top25_ratio`    | [0, 1]  | Tỷ lệ pixel trong 25% trên ảnh                      |
| `canopy_bottom25_ratio` | [0, 1]  | Tỷ lệ pixel trong 25% dưới ảnh                      |

#### 4.2 Contour Complexity — 2 chiều

| Tên đặc trưng               | Giá trị | Ý nghĩa                                                |
| --------------------------- | ------- | ------------------------------------------------------ |
| `canopy_contour_complexity` | > 0     | `Perimeter / √Area` — viền phức tạp → cao (cây lá kim) |
| `canopy_convexity`          | (0, 1]  | `HullPerimeter / ContourPerimeter` — tán lồi → gần 1   |

#### 4.3 Horizontal Distribution — 3 chiều

| Tên đặc trưng           | Giá trị | Ý nghĩa                                                  |
| ----------------------- | ------- | -------------------------------------------------------- |
| `canopy_width_mean`     | [0, 1]  | Độ rộng tán trung bình theo từng hàng (chuẩn hóa)        |
| `canopy_width_std`      | [0, 1]  | Độ lệch chuẩn độ rộng (hình nón → cao; hình tròn → thấp) |
| `canopy_max_width_norm` | [0, 1]  | Độ rộng tán tối đa (chuẩn hóa)                           |

#### 4.4 Connected Components — 1 chiều

| Tên đặc trưng         | Giá trị | Ý nghĩa                                        |
| --------------------- | ------- | ---------------------------------------------- |
| `canopy_n_components` | ≥ 1     | Số vùng cây rời rạc trong ảnh (1 = cây đơn lẻ) |

---

## 🧰 Kỹ thuật xử lý ảnh nền

### Tạo mặt nạ cây (`mask_utils.py`)

Khi có thể, tất cả đặc trưng được tính trên **vùng cây** (loại bỏ nền trời, đất...).

```
1. GaussianBlur(5×5) → làm mịn nhiễu
2. Otsu Threshold    → tự động tìm ngưỡng phân tách (không hardcode)
3. Flood-fill từ 4 góc ảnh → đánh dấu vùng nền
4. Invert            → vùng còn lại = cây
5. MORPH_CLOSE + MORPH_OPEN → lấp lỗ, loại nhiễu
6. Fallback thông minh nếu mask rỗng
```

> **Lưu ý**: Nếu mask quá nhỏ (< 5% ảnh), các module tự động fallback dùng toàn ảnh.

### Thứ tự vector đặc trưng

Vector đặc trưng được sắp xếp theo thứ tự **cố định**:

```
[color_* × 24] → [shape_* × 12] → [texture_* × 18] → [canopy_* × 9]
```

Trong mỗi nhóm, đặc trưng sắp theo alphabet. Thứ tự **không thay đổi** khi thêm/bỏ nhóm khác.

---

## 🗄️ Chuẩn hóa Vector và Cơ sở dữ liệu KD-Tree

### 1. Chuẩn hóa Vector (`vector_normalizer.py`)

Do các đặc trưng có dải giá trị rất khác nhau, module `VectorNormalizer` giúp chuẩn hóa vector. Hỗ trợ 3 phương pháp:

- **`zscore`** (Mặc định): Đưa về mean = 0, std = 1. Tốt nhất cho KD-Tree Euclidean.
- **`l2`**: Chia mỗi vector cho chuẩn L2 (độ dài = 1). Dùng với độ đo Cosine.
- **`minmax`**: Scale về khoảng `[0, 1]`.

```python
from vector_normalizer import VectorNormalizer

norm = VectorNormalizer(method="zscore")
norm.fit(train_vectors)        # numpy array shape (N, D)
v_norm = norm.transform_one(raw_vector)

norm.save("normalizer.npz")
loaded_norm = VectorNormalizer.load("normalizer.npz")
```

### 2. KD-Tree Vector Database (`vector_db.py`)

Lưu trữ và truy vấn $k$-NN bằng cấu trúc **KD-Tree** nội bộ.

- Xây dựng cây: `O(N log N)`.
- Truy vấn: `O(log N)` nhờ Branch-and-Bound.
- Hỗ trợ: **Euclidean** (tương thích `zscore`) và **Cosine** (tương thích `l2`).

```python
from vector_db import VectorDatabase

db = VectorDatabase(distance="euclidean")
db.insert("tree1.png", vec_norm_1, label="Ginkgo biloba")
db.insert("tree2.png", vec_norm_2, label="Acer palmatum")

db.build_tree()
results = db.query(query_vector, k=5)
for r in results:
    print(f"Anh: {r['image_path']}, Khoang cach: {r['distance']}, Nhan: {r['label']}")

db.save("vector_db.npz")
loaded_db = VectorDatabase.load("vector_db.npz")
```

---

## 📊 Kết quả trả về của `extract()`

```python
result = extractor.extract("tree.png")

result["success"]            # bool   – True nếu thành công
result["image_path"]         # str    – Đường dẫn ảnh đầu vào
result["n_features"]         # int    – Tổng số chiều (~62)
result["feature_names"]      # list   – Tên các đặc trưng theo thứ tự vector
result["features"]           # dict   – {tên: giá trị float}
result["vector"]             # ndarray float32, shape (~62,) – vector GỐC
result["vector_normalized"]  # ndarray float32 hoặc None – vector ĐÃ CHUẨN HÓA
result["processing_time_ms"] # float  – Thời gian xử lý (ms)
result["errors"]             # dict hoặc None – Lỗi từng nhóm nếu có
```

---

## 🖼️ Demo trực quan

```bash
python demo.py path/to/tree.png
```

Tạo file `demo_output.png` gồm 12 panel (3 hàng × 4 cột):

| Hàng | Nội dung                                                         |
| ---- | ---------------------------------------------------------------- |
| 1    | Ảnh gốc · Mặt nạ cây · Contour tán · Gradient magnitude          |
| 2    | LBP Map · Vertical Profile · Hue Histogram · Green Mask overlay  |
| 3    | Bar chart hình thái · Bar chart kết cấu · Bảng tóm tắt đặc trưng |

---

## ⚠️ Lưu ý quan trọng

> [!IMPORTANT]
> **Không thay đổi `FEATURE_GROUP_ORDER`** trong `feature_extractor.py` sau khi đã xây dựng CSDL.
> Mọi thay đổi thứ tự sẽ làm vô hiệu toàn bộ vector đã lưu.

> [!NOTE]
> Ảnh đầu vào được **resize về 256×256** trước khi trích rút.
> Các đặc trưng hình thái đã chuẩn hóa theo kích thước ảnh → bất biến với kích thước gốc.

> [!TIP]
> Để tăng tốc khi xử lý batch lớn, có thể giảm `target_size` xuống `(128, 128)`:
>
> ```python
> extractor = TreeFeatureExtractor(target_size=(128, 128))
> ```

> [!NOTE]
> Tên class `LeafFeatureExtractor` vẫn được giữ lại như một alias để tương thích với code cũ:
>
> ```python
> from feature_extractor import LeafFeatureExtractor  # hoạt động bình thường
> ```
