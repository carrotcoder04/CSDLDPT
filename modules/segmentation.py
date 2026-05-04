from ultralytics import SAM
import numpy as np
import cv2
from PIL import Image, ImageOps

model = SAM('sam_b.pt')

def remove_tree_background(pil_image):
    pil_image = ImageOps.exif_transpose(pil_image)

    # Chỉ xoay sang ngang nếu ảnh là portrait (h > w)
    # SAM hoạt động tốt hơn với ảnh ngang, nhưng không xoay ảnh đã ngang sẵn
    orig_w, orig_h = pil_image.size
    need_rotate = orig_h > orig_w
    if need_rotate:
        process_img = pil_image.rotate(-90, expand=True)
    else:
        process_img = pil_image

    w, h = process_img.size
    img_rgb = process_img.convert("RGB")

    box = [int(w * 0.1), int(h * 0.05), int(w * 0.9), int(h * 0.92)]

    input_points = [
        # pos point: tán, thân, gốc
        [w // 2, int(h * 0.25)],
        [w // 2, int(h * 0.55)],
        [w // 2, int(h * 0.85)],

        # neg point: Chặn 2 bên
        [int(w * 0.35), int(h * 0.95)],
        [int(w * 0.65), int(h * 0.95)],
        [int(w * 0.20), int(h * 0.90)],
        [int(w * 0.80), int(h * 0.90)],
    ]
    input_labels = [1, 1, 1, 0, 0, 0, 0]

    results = model.predict(
        img_rgb,
        bboxes=[box],
        points=[input_points],
        labels=[input_labels],
        conf=0.35,
        device='mps',
        imgsz=1024,
        verbose=False,
    )

    if results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h))

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        img_rgba = np.array(process_img.convert("RGBA"))
        img_rgba[:, :, 3] = mask
        result_img = Image.fromarray(img_rgba)

        return result_img.rotate(90, expand=True) if need_rotate else result_img

    return process_img.rotate(90, expand=True) if need_rotate else process_img