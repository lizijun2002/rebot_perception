import numpy as np
import torch

from depth.depth_stub import estimate_depth_stub, depth_to_vis


class MarigoldDepthEstimator:
    """Marigold depth estimator with automatic fallback to a stub.

    If diffusers pipeline is available and model_id can be loaded, use it.
    Otherwise use a lightweight deterministic depth proxy (still produces a depth-like map).
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg or {}
        self.device = device
        self.enabled = bool(self.cfg.get("enabled", True))
        self.use_fp16 = bool(self.cfg.get("use_fp16", True))
        self.model_id = self.cfg.get("model_id", "")

        self._pipe = None
        self._mode = "stub"

        if not self.enabled:
            self._mode = "stub"
            return

        try:
            from diffusers import DiffusionPipeline
            torch_dtype = torch.float16 if (self.use_fp16 and device.type == "cuda") else torch.float32
            self._pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch_dtype)
            self._pipe = self._pipe.to(device)
            self._pipe.set_progress_bar_config(disable=True)
            self._mode = "marigold"
        except Exception as e:
            self._pipe = None
            self._mode = "stub"
            print(f"[WARN] Marigold not available; using stub depth. Reason: {repr(e)}")

    @torch.no_grad()
    def __call__(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """rgb_uint8: [H,W,3] uint8 -> depth_float01: [H,W] float32 ~0..1"""
        if self._mode == "marigold":
            from PIL import Image
            img = Image.fromarray(rgb_uint8)

            out = self._pipe(img)
            depth = None
            if hasattr(out, "prediction"):
                depth = out.prediction
            elif isinstance(out, dict):
                depth = out.get("prediction", None) or out.get("depth", None)
            else:
                depth = getattr(out, "depth", None)

            if depth is None:
                return estimate_depth_stub(rgb_uint8)

            if isinstance(depth, torch.Tensor):
                d = depth.squeeze().detach().float().cpu().numpy()
            else:
                d = np.array(depth).astype(np.float32)

            d = d - d.min()
            d = d / (d.max() + 1e-6)
            return d.astype(np.float32)

        return estimate_depth_stub(rgb_uint8)

    def visualize(self, depth_float01: np.ndarray) -> np.ndarray:
        return depth_to_vis(depth_float01)
