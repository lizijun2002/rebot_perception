import numpy as np
from PIL import Image


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def resize_to_shorter_side(rgb_uint8: np.ndarray, shorter: int) -> np.ndarray:
    h, w = rgb_uint8.shape[:2]
    if min(h, w) == shorter:
        return rgb_uint8
    if w < h:
        new_w = shorter
        new_h = int(h * (shorter / w))
    else:
        new_h = shorter
        new_w = int(w * (shorter / h))
    img = Image.fromarray(rgb_uint8).resize((new_w, new_h), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)
