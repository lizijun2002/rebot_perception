from abc import ABC, abstractmethod
import numpy as np


class SegAdapter(ABC):
    @abstractmethod
    def predict(self, rgb_uint8: np.ndarray, depth_float01: np.ndarray | None = None) -> np.ndarray:
        """Return label map uint8 [H,W]."""
        raise NotImplementedError

    @abstractmethod
    def visualize(self, label_hw: np.ndarray) -> np.ndarray:
        """Return colored visualization uint8 [H,W,3]."""
        raise NotImplementedError
