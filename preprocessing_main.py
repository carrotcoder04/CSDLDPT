import os
import time
from pathlib import Path
from PIL import Image
from modules.file_handler import get_all_images, save_image
from modules.transform import resize_only
from modules.segmentation import remove_tree_background

INPUT_DIR  = r"res/"
OUTPUT_DIR = r"res/"   # Lưu đè vào cùng thư mục (jpg → png)

def run_preprocessing():
    all_image_paths = get_all_images(INPUT_DIR)
    total_in_folder = len(all_image_paths)

    START_INDEX = 1
    image_paths = all_image_paths[START_INDEX - 1:]

    if not image_paths:
        print(f"Không còn ảnh nào để xử lý từ vị trí {START_INDEX}.")
        return

    print(f"start from {START_INDEX} (remain {len(image_paths)} ảnh).")

    start_time = time.time()

    for i, path in enumerate(image_paths):
        current_number = START_INDEX + i
        path_obj  = Path(path)
        filename  = path_obj.name
        # Giữ nguyên cấu trúc subdirectory tương đối so với INPUT_DIR
        rel_path  = path_obj.relative_to(INPUT_DIR)
        new_rel   = rel_path.with_suffix(".png")
        out_path  = Path(OUTPUT_DIR) / new_rel
        out_dir   = out_path.parent

        # Bỏ qua nếu file PNG đầu ra đã tồn tại (tránh double-process)
        if out_path.exists() and path_obj.suffix.lower() != ".png":
            print(f"[{current_number}/{total_in_folder}] skip (da xu ly): {filename}")
            continue

        try:
            with Image.open(path).convert("RGBA") as img:
                img_no_bg = remove_tree_background(img)
                final_img = resize_only(img_no_bg)
                save_image(final_img, str(out_dir), out_path.name)
                print(f"[{current_number}/{total_in_folder}] done: {new_rel}")

        except Exception as e:
            print(f"Lỗi tại file {filename}: {e}")

    end_time = time.time()
    print("DONE")
    print(f"Tổng thời gian chạy: {(end_time - start_time)/60:.2f} phút.")

if __name__ == "__main__":
    run_preprocessing()