import numpy as np
import torch
import torch.nn.functional as F

from adapters.base import SegAdapter
from utils.colorize import colorize_labels


class OursAdapter(SegAdapter):
    """Adapter for Ours (RGB + Depth).

    Supported:
      - mode: "jit" -> torch.jit.load(weight_path)
        We assume either:
          (a) model(rgb, depth) -> logits, OR
          (b) model(cat(rgb, depth)) -> logits
        This adapter tries (a) first, then falls back to (b).

      - mode: "pth" -> you must implement build_model() and load_state_dict()

    depth_float01: numpy float32 [H,W] in ~0..1
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
            # from ours_model_def import OursNet
            # self.model = OursNet(num_classes=self.num_classes).to(device).eval()
            # ckpt = torch.load(wpath, map_location=device)
            # self.model.load_state_dict(ckpt.get("state_dict", ckpt))
            raise RuntimeError("Ours mode=pht requires you to implement model definition loading in ours_adapter.py")
        else:
            raise ValueError(f"Unknown ours mode: {mode}")

        self.thresholded_argmax = bool(cfg.get("thresholded_argmax", True))

    @torch.no_grad()
    def predict(self, rgb_uint8: np.ndarray, depth_float01: np.ndarray) -> np.ndarray:
        rgb = self._preprocess_rgb(rgb_uint8).to(self.device)  # [1,3,H,W]
        depth = self._preprocess_depth(depth_float01, rgb.shape[-2:]).to(self.device)  # [1,1,H,W]

        logits = None
        try:
            logits = self.model(rgb, depth)
        except Exception:
            x = torch.cat([rgb, depth], dim=1)  # [1,4,H,W]
            logits = self.model(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if logits.shape[-2:] != rgb.shape[-2:]:
            logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        pred = logits.argmax(dim=1)[0].byte().cpu().numpy()
        return pred

    def visualize(self, label_hw: np.ndarray) -> np.ndarray:
        return colorize_labels(label_hw)

    def _preprocess_rgb(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(rgb_uint8).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        return x

    def _preprocess_depth(self, depth_float01: np.ndarray, hw: tuple[int, int]) -> torch.Tensor:
        d = torch.from_numpy(depth_float01).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        if d.shape[-2:] != hw:
            d = F.interpolate(d, size=hw, mode="bilinear", align_corners=False)
        d = (d - d.mean(dim=[2, 3], keepdim=True)) / (d.std(dim=[2, 3], keepdim=True) + 1e-6)
        return d
