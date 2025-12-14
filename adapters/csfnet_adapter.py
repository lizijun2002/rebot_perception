import numpy as np
import torch
import torch.nn.functional as F

from adapters.base import SegAdapter
from utils.colorize import colorize_labels


class CSFNetAdapter(SegAdapter):
    """Baseline adapter for CSFNet.

    Supported:
      - mode: "jit" -> torch.jit.load(weight_path)
      - mode: "pth" -> you must implement build_model() and load_state_dict()

    Output:
      - label map [H,W] uint8
    """

    def __init__(self, cfg: dict, device: torch.device, num_classes: int = 19):
        self.cfg = cfg
        self.device = device
        self.num_classes = int(num_classes)

        mode = cfg.get("mode", "jit")
        wpath = cfg.get("weight_path", "")
        if mode == "jit":
            self.model = torch.jit.load(wpath, map_location=device).eval()
        elif mode == "pth":
            # ====== YOU MUST EDIT THIS PART if you use .pth ======
            # Example:
            # from csfnet_model_def import CSFNet
            # self.model = CSFNet(num_classes=self.num_classes).to(device).eval()
            # ckpt = torch.load(wpath, map_location=device)
            # self.model.load_state_dict(ckpt.get("state_dict", ckpt))
            raise RuntimeError("CSFNet mode=pht requires you to implement model definition loading in csfnet_adapter.py")
        else:
            raise ValueError(f"Unknown csfnet mode: {mode}")

        self.thresholded_argmax = bool(cfg.get("thresholded_argmax", True))

    @torch.no_grad()
    def predict(self, rgb_uint8: np.ndarray, depth_float01=None) -> np.ndarray:
        x = self._preprocess_rgb(rgb_uint8).to(self.device)  # [1,3,H,W]
        logits = self.model(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        pred = logits.argmax(dim=1)[0].byte().cpu().numpy()
        return pred

    def visualize(self, label_hw: np.ndarray) -> np.ndarray:
        return colorize_labels(label_hw)

    def _preprocess_rgb(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(rgb_uint8).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        return x
