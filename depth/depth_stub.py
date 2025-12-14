import numpy as np
import cv2


def estimate_depth_stub(rgb_uint8: np.ndarray) -> np.ndarray:
    """Deterministic depth-like proxy from luminance smoothing. Output normalized to 0..1."""
    rgb = rgb_uint8.astype(np.float32) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    d = cv2.GaussianBlur(lum, (0, 0), sigmaX=5, sigmaY=5)
    d = d - d.min()
    d = d / (d.max() + 1e-6)
    return d.astype(np.float32)


def depth_to_vis(depth_float01: np.ndarray) -> np.ndarray:
    """Convert depth [H,W] in 0..1 to a 3-channel uint8 heatmap."""
    d = np.clip(depth_float01, 0.0, 1.0)
    d8 = (d * 255).astype(np.uint8)
    vis = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis
