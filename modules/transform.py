from PIL import Image

def resize_only(pil_image, size=(256, 256)):
    # resize 256x256 – khớp với DEFAULT_TARGET_SIZE của TreeFeatureExtractor
    # tránh double-resize làm mất chất lượng (preprocessing → extractor)
    return pil_image.resize(size, Image.Resampling.LANCZOS)