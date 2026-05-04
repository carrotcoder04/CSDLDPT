import os
import glob

def get_all_images(input_path):
    """Tìm tất cả ảnh trong thư mục, kể cả subdirectory (recursive)."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_path, '**', ext), recursive=True))
    return sorted(files)

def save_image(image, output_path, filename):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #luu anh thanh png
    image.save(os.path.join(output_path, filename))