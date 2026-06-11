# Câu Hỏi Hay Gặp Khi Bảo Vệ Project

Project hiện tại: hệ truy vấn ảnh cây theo nội dung ảnh (CBIR), dùng **1769 ảnh hợp lệ**, **20 loại cây**, vector đặc trưng **62 chiều**.

- Màu sắc: 24 chiều
- Hình thái: 12 chiều
- Kết cấu: 17 chiều
- Tán cây: 9 chiều

---

## 1. Câu Hỏi Chưa Đụng Đến Code

### Tổng Quan

1. Project của nhóm giải quyết bài toán gì?
   - Trả lời ngắn: Hệ thống truy vấn ảnh cây theo nội dung ảnh. Người dùng đưa vào một ảnh cây, hệ thống trả về Top-K ảnh cây tương tự trong database.

2. CBIR là gì?
   - CBIR là Content-Based Image Retrieval, tức tìm kiếm ảnh dựa trên nội dung như màu sắc, hình dạng, kết cấu, thay vì tìm bằng tên file hay nhãn.

3. Đầu vào và đầu ra của hệ thống là gì?
   - Đầu vào: một ảnh cây truy vấn.
   - Đầu ra: Top-5 ảnh cây tương tự nhất kèm nhãn và khoảng cách Euclidean.

4. Dataset có bao nhiêu ảnh và bao nhiêu lớp?
   - Database hiện tại có 1769 ảnh hợp lệ, thuộc 20 loại cây.
   - Thư mục có 1770 file, nhưng 1 ảnh lỗi đọc nên bị loại khi build.

5. Pipeline tổng quát của hệ thống là gì?
   - Ảnh truy vấn -> resize -> tạo mask cây -> trích 62 đặc trưng -> chuẩn hóa Z-score -> tìm kiếm vector gần nhất -> trả Top-5 ảnh tương tự.

6. Vì sao phải tách nền trước khi trích đặc trưng?
   - Vì nền trời, đất, nhà cửa có thể làm sai màu sắc, texture và hình dạng. Tách nền giúp feature tập trung vào vùng cây.

7. Mask cây được tạo như thế nào?
   - Nếu ảnh có alpha mask hợp lệ thì dùng alpha.
   - Nếu không, hệ thống dùng fusion từ GrabCut, Otsu/flood-fill và HSV color mask, sau đó làm sạch bằng morphology.

8. Nếu mask sai thì ảnh hưởng thế nào?
   - Sai mask sẽ làm sai gần như toàn bộ feature: màu, hình dạng, texture và tán cây đều phụ thuộc vùng cây.

### Đặc Trưng 62 Chiều

9. Vector 62 chiều gồm những nhóm nào?
   - Color 24, Shape 12, Texture 17, Canopy 9.

10. Vì sao chọn 4 nhóm đặc trưng này?
    - Vì cây khác nhau ở màu lá/thân, dáng cây, độ nhám texture và cấu trúc tán.

11. Nhóm màu sắc gồm những gì?
    - Hue histogram 8 chiều, thống kê HSV 6 chiều, 3 màu chủ đạo BGR 9 chiều, green ratio 1 chiều.

12. Vì sao dùng HSV?
    - HSV tách sắc độ, độ bão hòa và độ sáng, gần với cách con người cảm nhận màu hơn RGB.

13. Vì sao Hue mean phải tính theo circular mean?
    - Vì Hue là vòng tròn màu. Ví dụ màu gần 0 độ và gần 180/360 độ thực chất gần nhau, lấy trung bình thường có thể sai.

14. Green ratio dùng để làm gì?
    - Đo tỷ lệ pixel xanh lá trong vùng cây, giúp phân biệt cây xanh với cây khô, cây lá đỏ/vàng.

15. Dominant colors được tính như thế nào?
    - Lập histogram 3D trên không gian BGR, lấy 3 bin có số pixel nhiều nhất làm 3 màu chủ đạo.

16. Nhóm hình thái gồm những gì?
    - Area ratio, aspect ratio, centroid x/y, crown ratio, extent ratio, 4 Hu moments, solidity, symmetry.

17. Aspect ratio, solidity, extent ratio khác nhau thế nào?
    - Aspect ratio: tỷ lệ rộng/cao của bounding box.
    - Solidity: diện tích cây chia diện tích convex hull.
    - Extent ratio: diện tích cây chia diện tích bounding box.

18. Hu Moments dùng để làm gì?
    - Mô tả hình dạng và tương đối bất biến với tịnh tiến, xoay, thay đổi tỷ lệ.

19. Nhóm texture gồm những gì?
    - LBP histogram 10 chiều, GLCM 4 chiều, gradient mean/std 2 chiều, roughness 1 chiều.

20. LBP là gì?
    - Local Binary Pattern, so sánh pixel trung tâm với 8 pixel xung quanh để mô tả kết cấu cục bộ.

21. GLCM là gì?
    - Gray-Level Co-occurrence Matrix, thống kê mức độ xuất hiện đồng thời của các cặp mức xám theo các hướng khác nhau.

22. Contrast và homogeneity trong GLCM nói lên điều gì?
    - Contrast cao nghĩa là nhiều biến thiên/cạnh.
    - Homogeneity cao nghĩa là texture mượt và đồng nhất hơn.

23. Nhóm tán cây gồm những gì?
    - Bottom25 ratio, top25 ratio, peak row, contour complexity, convexity, max width, width mean/std, số component.

24. Contour complexity cao nghĩa là gì?
    - Viền cây phức tạp, răng cưa hoặc nhiều chi tiết, ví dụ tán lá kim hoặc tán phân mảnh.

### Chuẩn Hóa Và Tìm Kiếm

25. Vì sao phải chuẩn hóa vector?
    - Vì các feature có thang đo khác nhau. Nếu không chuẩn hóa, feature có giá trị lớn như GLCM contrast sẽ lấn át các feature nhỏ.

26. Hệ thống chuẩn hóa bằng gì?
    - Z-score theo từng chiều: `z = (x - mean) / std`, sau đó clip về `[-3, 3]`.

27. Mean/std lấy từ đâu?
    - Được fit trên toàn bộ vector trong database khi build, sau đó lưu vào `normalizer.npz`.

28. Vì sao query phải dùng cùng normalizer với database?
    - Để vector query nằm cùng không gian chuẩn hóa với vector trong database.

29. Database vector lưu gì?
    - `vector_db.npz` lưu `vectors`, `image_paths`, `labels`.
    - Shape hiện tại là `(1769, 62)`.

30. Hệ thống dùng khoảng cách gì?
    - Euclidean distance trên vector đã chuẩn hóa.

31. KD-Tree dùng để làm gì?
    - Dùng để tăng tốc tìm kiếm các vector gần nhất trong không gian nhiều chiều.

32. KD-Tree xây dựng như thế nào?
    - Chọn chiều có phương sai lớn nhất, lấy median làm node, chia dữ liệu thành nhánh trái/phải và đệ quy.

33. KD-Tree có luôn tốt với 62 chiều không?
    - Không chắc. KD-Tree giảm hiệu quả khi số chiều cao, nhưng với 1769 ảnh thì vẫn chạy được nhanh.

34. App có dùng KD-Tree thật không?
    - Có. `app.py` load `VectorDatabase`, và `VectorDatabase.load()` tự build KD-Tree để query.

35. Vì sao `main.py --query` lại dùng NumPy brute-force?
    - CLI dùng brute-force vectorized để đơn giản và dễ kiểm chứng. App chính dùng `VectorDatabase` KD-Tree.

### Đánh Giá

36. Hệ thống đánh giá bằng chỉ số nào?
    - Precision@1, Precision@5, mAP@5.

37. Precision@5 là gì?
    - Tỷ lệ ảnh đúng nhãn trong 5 kết quả đầu.

38. mAP@5 là gì?
    - Trung bình Average Precision@5 trên nhiều query, có xét thứ tự xuất hiện của kết quả đúng.

39. Leave-one-out là gì?
    - Mỗi ảnh trong database lần lượt được dùng làm query, sau đó loại chính nó khỏi kết quả để đánh giá.

40. Nếu kết quả chưa cao thì nguyên nhân có thể là gì?
    - Mask chưa tốt, các loài cây giống nhau, ảnh dataset nhiễu, hoặc feature thủ công chưa đủ phân biệt.

41. Muốn cải thiện độ chính xác thì làm gì?
    - Cải thiện mask, thêm/trọng số feature, thử metric khác, hoặc dùng deep learning embedding.

---

## 2. Câu Hỏi Hay Hỏi Về Code

42. File nào là entry point để build database?
    - `main.py`
    - Lệnh: `python3 main.py --build --image_dir Raw_Tree_Dataset_Test`

43. File nào chạy giao diện?
    - `app.py`
    - Lệnh: `python3 app.py`

44. File nào điều phối trích xuất 62 đặc trưng?
    - `feature_extractor.py`, class `TreeFeatureExtractor`.

45. Các nhóm feature nằm ở đâu?
    - `features/color_features.py`
    - `features/shape_features.py`
    - `features/texture_features.py`
    - `features/canopy_features.py`

46. File nào tạo mask cây?
    - `features/mask_utils.py`, hàm `create_tree_mask()`.

47. File nào quản lý vector database và KD-Tree?
    - `vector_db.py`.

48. File nào chuẩn hóa vector?
    - `vector_normalizer.py`.

49. `TreeFeatureExtractor.extract()` làm gì?
    - Load ảnh, resize, tạo mask, gọi 4 module feature, ghép feature thành vector 62 chiều, chuẩn hóa nếu có normalizer.

50. Vì sao cần `FEATURE_GROUP_ORDER`?
    - Để thứ tự ghép vector luôn ổn định giữa lúc build database và lúc query.

51. Nếu thêm feature mới thì cần chú ý gì?
    - Phải đảm bảo thứ tự vector ổn định, cập nhật tài liệu, rebuild `vector_db.npz` và `normalizer.npz`.

52. `VectorNormalizer` lưu những gì?
    - Method, eps, số chiều, số mẫu fit, mean và std từng chiều.

53. `VectorDatabase.load()` làm gì?
    - Load vector/path/label từ `vector_db.npz`, insert records và build KD-Tree.

54. `VectorDatabase.query()` kiểm tra gì trước khi query?
    - Kiểm tra KD-Tree đã build chưa và số chiều query có khớp DB không.

55. App query bằng vector thô hay vector chuẩn hóa?
    - App dùng `vector_normalized` nếu extractor có normalizer đã fit.

56. App lưu ảnh sau tiền xử lý ở đâu?
    - App tạo file `.png` tạm trong thư mục project, sau đó xóa file tạm sau khi load lên UI.

57. Vì sao app query `k=6` rồi lấy 5 kết quả?
    - Để loại chính ảnh query nếu ảnh đó đã nằm trong database.

58. Nếu app báo lỗi số chiều không khớp thì nguyên nhân là gì?
    - Code feature và `vector_db.npz`/`normalizer.npz` không đồng bộ. Cần build lại database.

59. Nếu app không load được DB thì kiểm tra gì?
    - Kiểm tra `vector_db.npz`, `normalizer.npz`, dependency, và đường dẫn chạy app.

60. Ảnh hỏng trong dataset được xử lý như thế nào khi build?
    - Extract thất bại, kết quả `success=False`, sau đó `main.py` lọc ra khỏi database.

---

## 3. Câu Hỏi Hay Hỏi Về Sửa Code

61. Muốn lưu ảnh sau xử lý thay vì xóa thì sửa ở đâu?
    - Sửa `app.py`, hàm `preprocess_query_image()` và bỏ đoạn xóa `Path(processed_path).unlink(...)`.
    - Có thể lưu vào thư mục cố định như `processed_outputs/`.

62. Muốn đổi Top-5 thành Top-10 thì sửa ở đâu?
    - Trong `app.py`, đổi query `k=6` thành `k=11` và đổi `[:5]` thành `[:10]`.
    - Với CLI thì dùng `--k 10`.

63. Muốn thêm một đặc trưng mới thì làm gì?
    - Thêm feature vào module tương ứng, kiểm tra số chiều mới, cập nhật báo cáo, rebuild DB và normalizer.

64. Muốn bỏ một đặc trưng thì làm gì?
    - Xóa feature khỏi module, cập nhật số chiều, rebuild DB và normalizer.

65. Muốn đổi metric từ Euclidean sang cosine thì sửa ở đâu?
    - Sửa `VectorDatabase(distance="cosine")` hoặc phần tính distance trong `main.py`.
    - Cần L2-normalize vector.

66. Muốn tăng trọng số nhóm màu sắc thì sửa thế nào?
    - Thêm weight vector sau chuẩn hóa, nhân các chiều thuộc nhóm color trước khi tính distance.

67. Muốn lưu `feature_names` vào database thì sửa ở đâu?
    - Sửa `main.py` khi `np.savez_compressed()` để lưu thêm `feature_names`, và sửa load nếu cần kiểm tra.

68. Muốn đảm bảo query và DB luôn cùng thứ tự feature thì nên làm gì?
    - Lưu `feature_names` khi build DB, khi query thì assert danh sách feature names khớp.

69. Muốn xử lý dataset lớn hơn, ví dụ 100k ảnh, thì nên đổi gì?
    - Dùng FAISS/HNSW/Annoy thay KD-Tree thuần Python, tối ưu batch extract và lưu metadata riêng.

70. Muốn tăng tốc build database thì làm gì?
    - Dùng multiprocessing, cache mask, giảm log, tối ưu LBP/GLCM, hoặc xử lý batch.

71. Muốn đánh giá công bằng hơn thì sửa gì?
    - Tách train/test rõ ràng, fit normalizer trên train, query/evaluate trên test hoặc protocol riêng.

72. Muốn app hiển thị đủ 62 feature thì sửa ở đâu?
    - Sửa phần tạo `lines` trong `search_similar_trees()` của `app.py`.

73. Muốn đổi kích thước xử lý từ 256 lên 512 thì sửa ở đâu?
    - Sửa `TreeFeatureExtractor.DEFAULT_TARGET_SIZE` hoặc truyền `target_size=(512,512)`, sau đó rebuild DB.

74. Muốn dùng deep learning thay feature thủ công thì thay phần nào?
    - Thay `TreeFeatureExtractor` bằng extractor CNN/CLIP/ResNet embedding, còn database/query có thể giữ logic tương tự.

75. Muốn sửa lỗi app không tìm thấy ảnh kết quả thì xem hàm nào?
    - `resolve_abs_path()` trong `app.py`.

---

## 4. Câu Nên Chuẩn Bị Trả Lời Chắc

1. Project dùng bao nhiêu chiều?
   - 62 chiều.

2. DB hiện tại shape bao nhiêu?
   - `(1769, 62)`.

3. Vì sao có 1769 ảnh?
   - Dataset có 1770 file nhưng 1 ảnh lỗi đọc, nên build thành công 1769 ảnh.

4. App dùng gì để query?
   - `VectorDatabase` và KD-Tree.

5. Chuẩn hóa bằng gì?
   - Z-score, clip `[-3, 3]`.

6. Distance là gì?
   - Euclidean distance.

7. Lệnh build là gì?
   - `python3 main.py --build --image_dir Raw_Tree_Dataset_Test`

8. Lệnh chạy app là gì?
   - `python3 app.py`

9. Nếu đổi feature thì cần làm gì?
   - Rebuild `vector_db.npz` và `normalizer.npz`.

10. Ảnh sau xử lý lưu ở đâu?
    - Hiện chỉ lưu tạm rồi xóa, chưa lưu cố định.
