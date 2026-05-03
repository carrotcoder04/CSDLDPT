import os
import time
from PIL import Image
from modules.file_handler import get_all_images, save_image
from modules.transform import resize_only
from modules.segmentation import remove_tree_background

INPUT_DIR = r"res/"
OUTPUT_DIR = r"res"

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
        filename = os.path.basename(path)
        new_filename = os.path.splitext(filename)[0] + ".png"
        
        try:
            with Image.open(path).convert("RGBA") as img:
        
                img_no_bg = remove_tree_background(img)
                final_img = resize_only(img_no_bg)
                save_image(final_img, OUTPUT_DIR, new_filename)
                
                print(f"[{current_number}/{total_in_folder}] done: {new_filename}")
                    
        except Exception as e:
            print(f"Lỗi tại file {filename}: {e}")

    end_time = time.time()
    print(f"DONE")
    print(f"Tổng thời gian chạy: {(end_time - start_time)/60:.2f} phút.")

if __name__ == "__main__":
    run_preprocessing()