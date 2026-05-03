from PIL import Image

def resize_only(pil_image, size=(512, 512)):
    # resize 512x512
    return pil_image.resize(size, Image.Resampling.LANCZOS)