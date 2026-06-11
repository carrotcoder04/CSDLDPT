# Chi Tiết 62 Đặc Trưng Của Hệ Thống Tra CứU Ảnh Cây (CBIR)

Tài liệu này trình bày chi tiết về 62 đặc trưng (features) được trích xuất từ mỗi bức ảnh cây, bao gồm ý nghĩa và công thức toán học/thuật toán được sử dụng. Hệ thống sử dụng 4 nhóm đặc trưng chính: Màu sắc (24), Hình thái (12), Kết cấu (17), và Tán cây (9).

---

## Tiền Xử Lý Ảnh (Preprocessing)

Trước khi trích xuất đặc trưng, mỗi ảnh đầu vào sẽ đi qua một quá trình tiền xử lý để chuẩn hóa kích thước và bóc tách vùng cây khỏi nền (background):

1. **Chuẩn hóa kích thước:** Mọi ảnh được chuyển về kích thước $256 \times 256$ pixel.
2. **Tách nền bằng AI (Rembg):** Hệ thống ưu tiên dùng `rembg` (nhân mạng U2Net) để loại bỏ nền, tạo ra một kênh Alpha mask chính xác. Nếu diện tích cây (mask > 127) lớn hơn 1% tổng diện tích ảnh thì sẽ dùng kênh Alpha này.
3. **Multi-method Fusion (Fallback):** Trong trường hợp ảnh không có kênh Alpha (chưa qua Rembg), hệ thống tự động bóc tách nền bằng thuật toán "bình chọn" kết hợp 3 phương pháp thủ công:
   - **GrabCut:** Tự động tách tiền cảnh/hậu cảnh với initial rect là $80\%$ vùng giữa ảnh.
   - **Otsu + Flood-fill:** Làm mờ Gaussian, áp dụng ngưỡng Otsu và tô ngập (flood-fill) từ 4 góc ảnh để nhận diện nền.
   - **HSV Color Mask:** Kết hợp vùng màu xanh lá ($H \in [10°, 90°]$) và màu nâu/tối của thân cây.
   - Các vùng đạt $\ge 2/3$ phương pháp đồng thuận sẽ được lấy làm mặt nạ (mask) cuối cùng.
4. **Morphology:** Áp dụng thuật toán hình thái học (MORPH_CLOSE rồi MORPH_OPEN) với kernel ellipse $9 \times 9$ để làm sạch nhiễu và lấp lỗ hổng trong mask.

Mặt nạ nhị phân (0 hoặc 255) này sẽ được truyền duy nhất 1 lần vào tất cả 4 nhóm phía dưới.

---

## 1. Nhóm Đặc Trưng Màu Sắc (Color Features) - 24 chiều

Nhóm này mô tả thông tin màu sắc của cây, được tính toán chủ yếu trên không gian màu HSV (Hue, Saturation, Value).

### 1.1. Histogram Sắc Độ (Hue Histogram) - 8 chiều
**Ý nghĩa:** Phân bố màu sắc thuần túy của tán cây (bỏ qua độ sáng và độ bão hòa).
**Đặc trưng:** `hue_hist_0` đến `hue_hist_7`
**Công thức:**
Chia không gian Hue (0-180 trong OpenCV) thành 8 khoảng (bins).
$$h_i = \frac{|\{p \in \text{ROI} : H(p) \in [22.5i, 22.5(i+1))\}|}{|\text{ROI}|}, \quad i \in \{0, \dots, 7\}$$
Trong đó $H(p)$ là giá trị Hue của pixel $p$ thuộc vùng cây (ROI).
Histogram này được chuẩn hóa L1 (tổng bằng 1).

### 1.2. Thống kê HSV (HSV Statistics) - 6 chiều
**Ý nghĩa:** Trị số trung bình và độ lệch chuẩn của các kênh màu, thể hiện tông màu tổng thể và mức độ biến thiên màu sắc.

1. **`h_mean` (Circular Mean của Hue):**
   Vì Hue là một vòng tròn (0° tương đương 360°), tính trung bình thông thường sẽ sai (ví dụ trung bình của 1° và 359° phải là 0° chứ không phải 180°). Ta dùng trung bình trên mặt phẳng phức:
   $$\bar{H} = \frac{1}{2} \cdot \arg\!\left(\frac{1}{N}\sum_{i=1}^{N} e^{j \cdot 2H_i \cdot \frac{\pi}{180}}\right) \cdot \frac{180}{\pi} \mod 360$$
2. **`h_std` (Circular Std của Hue):** Độ phân tán sắc độ trên vòng Hue.
3. **`s_mean` (Mean Saturation):** Trung bình độ bão hòa. $\frac{1}{N}\sum S(p)$
4. **`v_mean` (Mean Value):** Trung bình độ sáng. $\frac{1}{N}\sum V(p)$
5. **`s_std` (Std Saturation):** Độ lệch chuẩn của độ bão hòa. $\sqrt{\frac{1}{N}\sum (S(p) - \bar{S})^2}$
6. **`v_std` (Std Value):** Độ lệch chuẩn của độ sáng. $\sqrt{\frac{1}{N}\sum (V(p) - \bar{V})^2}$

### 1.3. Màu sắc chủ đạo (Dominant Colors) - 9 chiều
**Ý nghĩa:** Ba tông màu nổi bật nhất của cây (ví dụ: xanh lá, nâu thân, vùng sáng/tối).
**Đặc trưng:** `dom_b1..dom_b3`, `dom_g1..dom_g3`, `dom_r1..dom_r3`
**Cách tính:** 
Không sử dụng KMeans. Xây dựng Histogram 3D trên không gian BGR với $16 \times 16 \times 16$ bins. Tìm 3 bins có tần suất xuất hiện cao nhất. Giá trị B, G, R là tọa độ tâm của 3 bins đó, chuẩn hóa về khoảng $[0, 1]$.

### 1.4. Tỷ lệ xanh lá (Green Ratio) - 1 chiều
**Ý nghĩa:** Chỉ số quan trọng để phân biệt cây lá xanh với cây rụng lá/cây khô.
**Đặc trưng:** `green_ratio`
**Công thức:**
$$\text{green\_ratio} = \frac{|\{p \in \text{ROI} : 17 \le H(p) \le 42 \wedge S(p) > 40 \wedge V(p) > 40\}|}{|\text{ROI}|}$$
*(Lưu ý: H trong OpenCV từ 0-180, nên 17-42 tương đương 34°-84°).*

---

## 2. Nhóm Đặc Trưng Hình Thái (Shape Features) - 12 chiều

Nhóm này mô tả hình dạng tổng quát của cây dựa trên đường viền ngoài cùng (contour) lớn nhất.

### 2.1. Đặc trưng hình học cơ bản - 8 chiều

1. **`area_ratio` (Tỷ lệ diện tích):** Tỷ lệ diện tích cây so với diện tích toàn ảnh.

2. **`aspect_ratio` (Tỷ lệ khung hình):** Tỷ lệ giữa chiều rộng và chiều cao của bounding box bao quanh cây.
   $$\text{aspect\_ratio} = \frac{W}{H}$$
   *(Cây cao thẳng có aspect_ratio < 1, cây bụi có aspect_ratio > 1)*

3. **`centroid_x_norm`, `centroid_y_norm`:** Tọa độ tâm hình học của contour, chuẩn hóa theo chiều rộng/cao ảnh.
   
4. **`solidity` (Độ đặc):** Tỷ lệ diện tích của cây so với diện tích của đa giác lồi bao quanh (Convex Hull).
   $$\text{solidity} = \frac{\text{Area}}{\text{ConvexHullArea}}$$
   *(Tán cây rậm rạp ≈ 1, tán cây thưa thớt có nhiều lỗ hổng < 0.7)*

5. **`extent_ratio` (Mức lấp đầy):** Tỷ lệ diện tích cây so với diện tích bounding box.
   $$\text{extent\_ratio} = \frac{\text{Area}}{W \times H}$$

6. **`crown_ratio` (Tỷ lệ tán):** Tỷ lệ diện tích cây nằm ở nửa trên của bức ảnh so với toàn bộ cây.
   $$\text{crown\_ratio} = \frac{\text{Pixels in Top Half}}{\text{Total Pixels}}$$

7. **`symmetry` (Đối xứng):** Mức chồng khớp giữa mask cây và bản lật ngang của chính nó trong bounding box.

### 2.2. Hu Moments - 4 chiều
**Ý nghĩa:** Đặc trưng hình học bất biến với phép tịnh tiến, phép quay và thay đổi tỷ lệ.
**Đặc trưng:** `hu_0`, `hu_1`, `hu_2`, `hu_3`
**Công thức:**
Tính giá trị Hu moment $h_i$, sau đó áp dụng phép biến đổi logarit để thu hẹp dải giá trị:
$$\tilde{h}_i = -\text{sign}(h_i) \cdot \log_{10}(|h_i| + 10^{-12}), \quad i \in \{0, 1, 2, 3\}$$

---

## 3. Nhóm Đặc Trưng Kết Cấu (Texture Features) - 17 chiều

Mô tả bề mặt, độ nhám, và sự phân bố chi tiết của tán lá.

### 3.1. Local Binary Pattern (LBP) Histogram - 10 chiều
**Ý nghĩa:** Bắt các mẫu kết cấu vi mô (mịn, thô, cạnh, góc).
**Đặc trưng:** `lbp_0` đến `lbp_9`
**Công thức:**
Với mỗi pixel trung tâm $g_c$, so sánh với 8 pixel lân cận $g_n$ trên đường tròn bán kính 1:
$$\text{LBP}(x, y) = \sum_{n=0}^{7} s(g_n - g_c) \cdot 2^n, \quad \text{với } s(x) = \begin{cases} 1 & x \ge 0 \\ 0 & x < 0 \end{cases}$$
Giá trị LBP (0-255) được gom thành 10 bins và chuẩn hóa theo tổng số pixel.

### 3.2. Gray-Level Co-occurrence Matrix (GLCM) - 4 chiều
**Ý nghĩa:** Thống kê mức độ xuất hiện đồng thời của các cặp mức xám ở một khoảng cách $d=1$. Tính trung bình trên 4 hướng ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) để bất biến xoay.
1. **`contrast` (Độ tương phản):** Mức độ biến thiên cục bộ. Cây lá kim (thông) thường có độ tương phản cao.
   $$\text{Contrast} = \sum_{i,j} (i-j)^2 \cdot P(i,j)$$
2. **`homogeneity` (Độ đồng nhất):** Sự mượt mà của bề mặt. Tán lá to, mượt sẽ có giá trị cao.
   $$\text{Homogeneity} = \sum_{i,j} \frac{P(i,j)}{1 + |i-j| + 10^{-9}}$$
3. **`correlation` (Tương quan):** Mức tương quan thống kê giữa hai mức xám đồng hiện.
4. **`energy` (Năng lượng):** Mức tập trung của phân phối GLCM, tính bằng $\sum P(i,j)^2$.

### 3.3. Gradient và Roughness - 3 chiều
**Đặc trưng:** `grad_mean`, `grad_std`, `roughness`.
Gradient được tính bằng Sobel để đo mật độ biên/cạnh. Roughness là trung bình sai khác tuyệt đối giữa ảnh xám và ảnh đã làm mờ Gaussian.

---

## 4. Nhóm Đặc Trưng Tán Cây (Canopy Features) - 9 chiều

Mô tả cấu trúc và sự phân bố không gian của tán cây.

### 4.1. Phân bố dọc (Vertical Distribution) - 3 chiều
Đếm số pixel cây trên từng hàng ngang.
1. **`peak_row_norm`:** Vị trí hàng ngang có số lượng pixel cây nhiều nhất (tán dày nhất), chuẩn hóa theo chiều cao ảnh (về [0, 1]).
2. **`top25_ratio`:** Tỷ lệ số pixel cây nằm trong 25% hàng trên cùng so với tổng số pixel cây. Cây có tán tập trung ở ngọn sẽ có giá trị cao.
3. **`bottom25_ratio`:** Tỷ lệ số pixel cây nằm trong 25% hàng dưới cùng so với tổng số pixel cây.

### 4.2. Độ phức tạp đường viền (Contour Complexity) - 1 chiều
**Đặc trưng:** `contour_complexity`
**Ý nghĩa:** Dựa trên chỉ số Polsby-Popper nghịch đảo. Viền răng cưa (cây thông) có giá trị cao, viền tròn (cây xoài) có giá trị thấp.
**Công thức:**
$$\text{Complexity} = \frac{\text{Perimeter}}{\sqrt{\text{Area}}}$$

### 4.3. Cấu trúc contour - 2 chiều
1. **`convexity`:** Tỷ lệ chu vi convex hull so với chu vi contour, phản ánh độ lồi/lõm của viền.
2. **`n_components`:** Số thành phần liên thông có diện tích đủ lớn trong mask.

### 4.4. Phân bố ngang (Horizontal Width) - 3 chiều
Đếm số pixel cây trên từng hàng ngang (coi như độ rộng tán tại hàng đó), loại bỏ các hàng không có cây, sau đó chuẩn hóa theo chiều rộng bức ảnh.
1. **`max_width_norm`:** Độ rộng lớn nhất của tán cây, chuẩn hóa theo chiều rộng ảnh.
2. **`width_mean`:** Trung bình của các độ rộng này. Thể hiện cây mập hay ốm.
3. **`width_std`:** Độ lệch chuẩn của các độ rộng. Hình trụ/cầu sẽ có std thấp, hình nón (rộng ở dưới hẹp ở trên) sẽ có std cao.
