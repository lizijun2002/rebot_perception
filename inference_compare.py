import argparse
import os
from pathlib import Path

import yaml
import torch

from utils.io import load_rgb, resize_to_shorter_side
from utils.mosaic import make_2x3_mosaic
from depth.marigold import MarigoldDepthEstimator
from adapters.csfnet_adapter import CSFNetAdapter
from adapters.ours_adapter import OursAdapter


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run(cfg: dict):
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load models
    csfnet = CSFNetAdapter(cfg["csfnet"], device=device, num_classes=cfg.get("num_classes", 19))
    ours = OursAdapter(cfg["ours"], device=device, num_classes=cfg.get("num_classes", 19))

    # 2) Marigold (or fallback)
    depth_est = MarigoldDepthEstimator(cfg.get("marigold", {}), device=device)

    # 3) For each condition: RGB -> Depth -> CSFNet pred -> Ours pred
    items = cfg["inputs"]  # dict: night/rain/snow -> path

    rgb_imgs = []
    depth_imgs = []
    csf_preds = []
    our_preds = []

    # preserve order: night, rain, snow
    for name in ["night", "rain", "snow"]:
        path = items[name]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input image for {name}: {path}")

        rgb = load_rgb(path)  # numpy uint8 (H,W,3)
        rgb = resize_to_shorter_side(rgb, int(cfg.get("img_size", 768)))
        rgb_imgs.append(rgb)

        depth = depth_est(rgb)  # numpy float32 (H,W) in ~[0,1]
        depth_imgs.append(depth_est.visualize(depth))  # numpy uint8 (H,W,3)

        csf = csfnet.predict(rgb)  # numpy uint8 (H,W)
        csf_preds.append(csfnet.visualize(csf))

        ours_pred = ours.predict(rgb, depth)  # numpy uint8 (H,W)
        our_preds.append(ours.visualize(ours_pred))

    # 4) Build mosaic (2x3 layout described in your paper)
    mosaic = make_2x3_mosaic(
        rgb_imgs=rgb_imgs,
        depth_rgb_imgs=depth_imgs,
        csfnet_rgb_imgs=csf_preds,
        ours_rgb_imgs=our_preds,
        titles_top=[
            "RGB-Night", "RGB-Rain", "RGB-Snow",
            "Depth-Night", "Depth-Rain", "Depth-Snow"
        ],
        titles_bottom=[
            "CSFNet-Night", "CSFNet-Rain", "CSFNet-Snow",
            "Ours-Night", "Ours-Rain", "Ours-Snow"
        ],
    )

    out_path = out_dir / "qualitative_2x3.png"
    mosaic.save(out_path)
    print(f"[OK] Saved mosaic to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
