# BÁO CÁO HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN
**Nhóm lớp:** 01  
**Nhóm BTL:** 18  
**Đề tài:** Xây dựng hệ CSDL lưu trữ và tìm kiếm ảnh cây

---

## MỤC LỤC
**LỜI NÓI ĐẦU**
**YÊU CẦU BÀI TẬP LỚN**
**CHƯƠNG I: DỮ LIỆU VÀ THUỘC TÍNH**
1. Yêu cầu dữ liệu
2. Thuộc tính của ảnh cây
3. Tổng quan quy trình trích rút
**CHƯƠNG II: TIỀN XỬ LÝ ẢNH**
1. Quá trình xử lí ảnh
2. Các bước xử lí
**CHƯƠNG III: TRÍCH RÚT ĐẶC TRƯNG**
1. Nhóm Màu Sắc (18 chiều)
2. Nhóm Hình Thái (7 chiều)
3. Nhóm Kết Cấu (7 chiều)
4. Nhóm Tán Cây (5 chiều)
5. Vector đặc trưng tổng hợp
**CHƯƠNG IV: CHUẨN HÓA ĐẶC TRƯNG VÀ TÌM KIẾM**
1. Chuẩn hóa Vector đặc trưng (Feature Normalization)
2. Cấu trúc lưu trữ dữ liệu (Vector Database)
3. Thuật toán tìm kiếm KD-Tree
4. Đo lường khoảng cách (Distance Metric)
**CHƯƠNG V: DEMO HỆ THỐNG VÀ ĐÁNH GIÁ KẾT QUẢ**
1. Demo hệ thống
2. Đánh giá kết quả đạt được

---

## LỜI NÓI ĐẦU
Nhóm 18 chúng em xin chân thành gửi lời cảm ơn đến thầy Nguyễn Đình Hóa, giảng viên phụ trách môn Hệ CSDL đa phương tiện, đã chỉ dạy, hướng dẫn và đóng góp ý kiến để giúp chúng em hoàn thành tốt bài tập lớn với đề tài **Xây dựng hệ CSDL lưu trữ và tìm kiếm ảnh cây**.

Dù đã nỗ lực hết mình, nhưng do giới hạn về thời gian và kinh nghiệm, bài tập lớn của chúng em không thể tránh khỏi những thiếu sót. Nhóm 18 rất mong nhận được sự góp ý từ thầy để hoàn thiện hơn trong những dự án học tập và công việc sau này. Chúng em xin chân thành cảm ơn!

---

## CHƯƠNG I: DỮ LIỆU VÀ THUỘC TÍNH

### 1. Yêu cầu dữ liệu
Tập dữ liệu hệ thống sử dụng gồm **1769 ảnh cây**, thuộc 20 chủng loại khác nhau (như cây phong, cây dừa, cây thông, cây sồi...).
- **Cùng kích thước, độ phân giải:** Các ảnh đầu vào được module tiền xử lý tự động chuẩn hóa về độ phân giải cố định $256 \times 256$ pixel.
- **Đối tượng:** Mỗi ảnh chứa 1 cây duy nhất, chụp ngang toàn bộ cây. Các cây có độ tuổi đa dạng.
- **Nền ảnh:** Ảnh được loại bỏ nền tự động, giữ lại kênh Alpha, tạo sự nhất quán về vùng chọn đối tượng cây (ROI).

### 2. Thuộc tính của ảnh cây
Để nhận diện và so sánh các cây, nhóm thiết kế 4 nhóm thuộc tính (tổng 37 chiều). Lý do lựa chọn:
- **Màu sắc cây (Color):** Giúp phân biệt các cây có lá đổi màu (phong lá đỏ), cây thường xanh, hoặc tỷ lệ thân/lá.
- **Hình thái tán cây (Tree Shape):** Tính chất hình học giúp phân loại dáng cây (ví dụ cây thông dáng nón nhọn, cây dừa dáng vươn dài, cây đa dáng tròn).
- **Kết cấu (Texture):** Phân biệt bề mặt tán lá (lá kim rậm rạp thường có độ tương phản, góc cạnh cao hơn lá rộng mượt mà).
- **Cấu trúc tán cây (Canopy):** Đánh giá mức độ phức tạp của viền lá và sự phân bố dọc/ngang của tán.

### 3. Tổng quan quy trình trích rút
#### Sơ đồ khối
`Ảnh truy vấn -> Chuẩn hóa kích thước -> Tách nền tạo Mask -> Trích xuất 37 Đặc trưng -> Chuẩn hóa Z-Score -> Truy vấn KD-Tree -> Top 5 ảnh tương đồng.`

#### Quy trình
Mỗi ảnh đi qua tiền xử lý để tạo mask nhị phân cô lập vùng cây. Từ mask này, 4 module tính toán đặc trưng độc lập sẽ hoạt động để tạo ra 4 vector con. Các vector con được nối lại thành một vector 37 chiều duy nhất, sau đó đi qua module chuẩn hóa trước khi lưu vào/truy vấn trong CSDL.

---

## CHƯƠNG II: TIỀN XỬ LÝ ẢNH

### 1. Quá trình xử lí ảnh
**1.1. Mục tiêu**
Đưa tất cả ảnh về cùng một quy chuẩn không gian và loại bỏ các thành phần nhiễu (nền trời, mặt đất, công trình xung quanh) để thuật toán trích xuất đặc trưng chỉ tập trung vào đối tượng cây.

**1.2. Tổng quan hệ thống**
Module tiền xử lý kết hợp cả công nghệ tách nền AI (U2Net) và các thuật toán thị giác máy tính truyền thống (OpenCV).

### 2. Các bước xử lí
- **Bước 1: Kiểm tra ảnh hợp lệ.** Đọc ảnh đầu vào, nếu dung lượng quá nhỏ hoặc không chứa vùng cây đủ lớn sẽ bị loại bỏ.
- **Bước 2: Chuẩn hóa kích thước.** Dùng nội suy `INTER_AREA` để đưa ảnh về kích thước cố định $256 \times 256$ pixel.
- **Bước 3: Tách nền tự động.** 
  - *Chế độ 1:* Sử dụng mạng U2Net (`rembg`) để sinh kênh Alpha sắc nét, giữ lại pixel cây và biến nền thành trong suốt.
  - *Chế độ 2 (Dự phòng - Multi-method Fusion):* Nếu ảnh chưa qua Rembg, hệ thống biểu quyết kết quả của 3 phương pháp thủ công: GrabCut tự động, Otsu Thresholding, và HSV Color Mask. Vùng ảnh được $2/3$ phương pháp đồng ý sẽ được giữ làm vùng Foreground.
- **Bước 4: Khử nhiễu & Crop.** Dùng phép Toán hình thái học (Close + Open) với kernel $9 \times 9$ để xóa các điểm nhiễu li ti và lấp lỗ hổng trong tán cây. Mặt nạ (Mask) sinh ra ở bước này sẽ được dùng cho toàn bộ quá trình tính toán tính năng phía sau.

---

## CHƯƠNG III: TRÍCH RÚT ĐẶC TRƯNG

Hệ thống trích xuất một vector đặc trưng gồm tổng cộng **37 chiều** từ mỗi bức ảnh, được chia thành 4 nhóm nhằm nắm bắt trọn vẹn đặc tính của cây.

### 1. Nhóm Màu Sắc (18 chiều)
Sử dụng không gian màu HSV (Hue, Saturation, Value) vì nó mô tả màu sắc gần giống với cách cảm nhận của con người hơn không gian RGB.
- **Hue Histogram (6 chiều):** Chia dải màu Hue ($0 \to 180$ trong OpenCV) thành 6 khoảng (bins) bằng nhau, mỗi khoảng $30^\circ$. Đếm số lượng pixel rơi vào từng khoảng để tạo thành histogram, sau đó chuẩn hóa L1:
  $$h_i = \frac{|\{p \in \text{ROI} : H(p) \in [30i, 30(i+1))\}|}{|\text{ROI}|}, \quad i \in \{0, \dots, 5\}$$
- **Thống kê HSV (5 chiều):** Bao gồm $\bar{S}, \bar{V}$ (trung bình) và $\sigma_S, \sigma_V$ (độ lệch chuẩn). Riêng kênh Hue là một vòng tròn màu sắc ($0^\circ \equiv 360^\circ$) nên $\bar{H}$ phải được tính bằng trung bình góc trên mặt phẳng phức:
  $$\bar{H} = \frac{1}{2} \cdot \arg\!\left(\frac{1}{N}\sum e^{j \cdot 2H_i \cdot \frac{\pi}{180}}\right) \cdot \frac{180}{\pi} \mod 360$$
- **Màu chủ đạo (6 chiều):** Tọa độ $(H, S, V)$ của 2 màu chiếm diện tích lớn nhất. Tính bằng cách lập 3D Histogram kích thước $16 \times 8 \times 8$ và lấy tọa độ tâm của 2 bin cao nhất.
- **Tỷ lệ xanh lá - Green Ratio (1 chiều):** Tỷ lệ diện tích lá xanh so với tổng diện tích cây:
  $$\text{Green Ratio} = \frac{|\{p \in \text{ROI} : 17 \le H(p) \le 42 \wedge S(p) > 40 \wedge V(p) > 40\}|}{|\text{ROI}|}$$

### 2. Nhóm Hình Thái (7 chiều)
Mô tả dáng cây dựa trên đường viền bao quanh (contour).
- **Chỉ số hình học cơ bản (4 chiều):**
  - *Aspect Ratio (Tỷ lệ khung hình):* $W / H$. (Cây mọc vươn cao có giá trị nhỏ, cây bụi có giá trị lớn).
  - *Solidity (Độ đặc):* $\text{Area} / \text{ConvexHullArea}$. (Đánh giá mức độ khuyết/lõm của tán cây).
  - *Extent Ratio (Mức lấp đầy):* $\text{Area} / (W \times H)$.
  - *Crown Ratio:* Tỷ lệ pixel cây nằm ở nửa trên của ảnh so với toàn bộ cây.
- **Hu Moments (3 chiều):** 3 mô-men đầu tiên ($h_0, h_1, h_2$) mang đặc tính bất biến với phép xoay, tịnh tiến và thay đổi kích thước. Do giá trị rất nhỏ nên được biến đổi logarit:
  $$\tilde{h}_i = -\text{sign}(h_i) \cdot \log_{10}(|h_i| + 10^{-12})$$

### 3. Nhóm Kết Cấu (7 chiều)
Đánh giá độ nhám và hoa văn phân bố trên bề mặt tán lá, vỏ cây.
- **LBP Histogram (5 chiều):** Local Binary Pattern so sánh mức sáng của pixel trung tâm với 8 pixel xung quanh (bán kính 1). Nếu điểm xung quanh sáng hơn/bằng thì ghi bit 1, ngược lại bit 0. Các giá trị LBP được gom thành 5 mức độ từ mịn đến thô.
  $$\text{LBP} = \sum_{n=0}^{7} s(g_n - g_c) \cdot 2^n \quad (\text{với } s(x)=1 \text{ nếu } x \ge 0)$$
- **GLCM (2 chiều):** Ma trận đồng hiện mức xám (Gray-Level Co-occurrence Matrix), tính trung bình trên 4 hướng ($0^\circ, 45^\circ, 90^\circ, 135^\circ$):
  - *Contrast (Tương phản):* $\sum (i-j)^2 \cdot P(i,j)$. Tán lá kim nhiều chi tiết sắc nhọn sẽ có Contrast cao.
  - *Homogeneity (Đồng nhất):* $\sum \frac{P(i,j)}{1 + |i-j|}$.

### 4. Nhóm Tán Cây (5 chiều)
Mô tả sự phân bố cấu trúc của tán.
- **Phân bố dọc (2 chiều):** 
  - `peak_row_norm`: Xác định tọa độ hàng ngang có số lượng pixel cây lớn nhất (tán dày nhất), chuẩn hóa về khoảng $[0, 1]$.
  - `top25_ratio`: Số pixel nằm ở $25\%$ chiều cao trên cùng chia cho tổng pixel cây.
- **Độ phức tạp viền (1 chiều):** Đo lường độ gai góc của viền cây (lá kim sẽ cao hơn lá xoài/bàng).
  $$\text{Contour Complexity} = \frac{\text{Perimeter}}{\sqrt{\text{Area}}}$$
- **Phân bố ngang (2 chiều):** Chiều rộng trung bình của tán dọc theo thân cây (`width_mean`) và độ lệch chuẩn của chiều rộng đó (`width_std` - phân biệt dáng nón và dáng cầu tròn).

### 5. Vector đặc trưng tổng hợp
Đầu ra của quy trình này là một vector số thực `1D NumPy Array` kích thước $1 \times 37$.

---

## CHƯƠNG IV: CHUẨN HÓA ĐẶC TRƯNG VÀ TÌM KIẾM

### 1. Chuẩn hóa Vector đặc trưng (Feature Normalization)
- **1.1. Sự cần thiết của việc chuẩn hóa:** Vector 37 chiều chứa các giá trị với thang đo khác nhau (LBP ở mức [0,1], trong khi GLCM Contrast có thể lên tới >1000). Nếu không chuẩn hóa, đặc trưng có dải giá trị lớn sẽ làm sai lệch bộ tính khoảng cách.
- **1.2. Phương pháp Z-score Standardization:** Đưa mọi chiều đặc trưng về phân bố có trung bình $\mu = 0$ và độ lệch chuẩn $\sigma = 1$: $z_i = (x_i - \mu_i) / \sigma_i$.
- **1.3. Outlier Clipping:** Giới hạn (clip) các giá trị trong khoảng $[-3.0, 3.0]$ để giảm tác động của các ảnh có giá trị cực đoan.

### 2. Cấu trúc lưu trữ dữ liệu (Vector Database)
Hệ thống lưu toàn bộ dữ liệu vector của 1769 ảnh vào tệp cấu trúc nén `vector_db.npz`. Mỗi bản ghi (record) bao gồm: đường dẫn file ảnh gốc, nhãn lớp (ví dụ: Ginkgo_Tree), và mảng vector 37 chiều đã được chuẩn hóa.

### 3. Thuật toán tìm kiếm KD-Tree
Hệ thống sử dụng cấu trúc cây **K-Dimensional Tree** thuần túy để tự động hóa tìm kiếm không dùng thư viện Machine Learning cao cấp.
- **3.1. Quy trình xây dựng cây KD-Tree:** Ở mỗi Node, thuật toán chọn chiều có **phương sai lớn nhất** để chia đôi không gian điểm dữ liệu. Điểm trung vị (median) được chọn làm Node chia nhánh.
- **3.2. Quy trình truy vấn và cơ chế Backtracking:** Khi nhận vector tìm kiếm, KD-Tree sẽ đi sâu xuống Node lá chứa query, tính khoảng cách, sau đó *backtrack* ngược lên cây để tìm xem có nhánh nào khác nằm gần query hơn không.

### 4. Đo lường khoảng cách (Distance Metric)
Hệ thống tính **Khoảng cách Euclidean (L2 Distance)** giữa vector query và vector trong KD-Tree. Top 5 ảnh có khoảng cách nhỏ nhất sẽ được xếp hạng theo thứ tự giảm dần sự tương đồng.

---

## CHƯƠNG V: DEMO HỆ THỐNG VÀ ĐÁNH GIÁ KẾT QUẢ

### 1. Demo hệ thống
- **Công cụ:** Giao diện trực quan được viết bằng framework Gradio, dễ dàng khởi chạy qua file `app.py`.
- **Quy trình tương tác:** 
  1. Người dùng kéo thả 1 bức ảnh cây (có hoặc không có trong tập dữ liệu) vào trình duyệt.
  2. Kết quả trung gian (Intermediate results) bao gồm ảnh sau tách nền (Mask) và Logs trích xuất được in trực tiếp lên màn hình Console và UI, giúp người dùng theo dõi tiến trình 37 đặc trưng.
  3. Giao diện xuất ra **Top 5 ảnh tương đồng nhất**, bao gồm Tên ảnh, Xếp hạng (Rank), và Độ đo Euclidean (Distance).

### 2. Đánh giá kết quả đạt được
Nhóm xây dựng script `evaluate.py` dùng phương pháp *Leave-one-out* (chạy từng ảnh truy vấn toàn DB). Hệ thống hiện tại với 1769 ảnh đã đạt được:
- Độ ổn định cao, nhận diện các cây khác biệt rõ rệt (như Dừa, Thông, Phong đỏ) rất tốt.
- mAP@5 (Mean Average Precision) và Precision@5 cho thấy khả năng xếp hạng top 5 chính xác nhờ mô hình 37 chiều tối ưu thủ công không phụ thuộc Deep Learning.
- Thời gian trích xuất và query K-NN cực nhanh (~vài mili-giây/ảnh) nhờ KD-Tree. Mọi yêu cầu của môn học đã được hoàn thiện.
