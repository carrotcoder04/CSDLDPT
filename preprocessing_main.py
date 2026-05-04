import os
import time
from pathlib import Path
from preprocessing import TreePreprocessingPipeline

INPUT_DIR  = r"res/"
OUTPUT_DIR = r"res/"

def get_all_images(folder: str) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    paths = []
    for p in Path(folder).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
    return sorted(paths)

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
    
    pipeline = TreePreprocessingPipeline(output_size=(512, 512), segment_method="rembg")

    for i, path in enumerate(image_paths):
        current_number = START_INDEX + i
        path_obj  = Path(path)
        filename  = path_obj.name
        
        rel_path  = path_obj.relative_to(INPUT_DIR)
        out_path  = Path(OUTPUT_DIR) / rel_path
        
        try:
            result = pipeline.run(path)
            if result.is_valid:
                pipeline.save_result(result, str(out_path))
                print(f"[{current_number}/{total_in_folder}] done: {rel_path}")
            else:
                print(f"[{current_number}/{total_in_folder}] invalid: {rel_path} - {result.validation.reason if result.validation else 'Error'}")

        except Exception as e:
            print(f"Lỗi tại file {filename}: {e}")

    end_time = time.time()
    print("DONE")
    print(f"Tổng thời gian chạy: {(end_time - start_time)/60:.2f} phút.")

if __name__ == "__main__":
    run_preprocessing()