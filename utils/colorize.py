import numpy as np


def colorize_labels(label_hw: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-colormap for label visualization."""
    lab = label_hw.astype(np.uint8)
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(256, 3), dtype=np.uint8)
    return colors[lab]
