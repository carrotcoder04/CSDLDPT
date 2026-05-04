import os
import time
from pathlib import Path
from preprocessing import TreePreprocessingPipeline

INPUT_DIR  = r"Raw_Tree_Dataset_Test"
OUTPUT_DIR = r"res"

def get_all_images(folder: str) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    paths = []
    for p in Path(folder).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
    return sorted(paths)

def run_preprocessing():
    all_image_paths = get_all_images(INPUT_DIR)
    
    if not all_image_paths:
        print(f"Không tìm thấy ảnh nào trong thư mục {INPUT_DIR}.")
        return

    print(f"Tìm thấy {len(all_image_paths)} ảnh. Bắt đầu tiền xử lý...")

    start_time = time.time()
    
    # Khởi tạo pipeline
    pipeline = TreePreprocessingPipeline(
        output_size=(512, 512), 
        segment_method="rembg",
        verbose=True
    )
    
    # Chạy batch processing với base_dir để giữ nguyên cấu trúc thư mục
    results = pipeline.run_batch(
        image_paths=all_image_paths,
        output_dir=OUTPUT_DIR,
        base_dir=INPUT_DIR,
        save_masks=True, # Lưu thêm ảnh mask nếu cần
        save_viz=False   # Bật True nếu muốn lưu ảnh các bước (visualization)
    )

    end_time = time.time()
    valid_count = sum(1 for r in results if r.is_valid)
    
    print("\n--- HOÀN THÀNH ---")
    print(f"Xử lý thành công: {valid_count}/{len(all_image_paths)} ảnh.")
    print(f"Tổng thời gian chạy: {(end_time - start_time)/60:.2f} phút.")
    print(f"Kết quả được lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_preprocessing()