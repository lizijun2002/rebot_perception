# Environment-Aware RGB+Depth Segmentation (Course Final Project)

This repository provides an inference pipeline to compare:
- **CSFNet (baseline)** segmentation results
- **Ours** environment-aware RGB+Depth fusion segmentation

It additionally integrates **Marigold** for pseudo-depth estimation, with automatic fallback to a lightweight stub if Marigold is not available.

## Quick Start
1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Edit config paths in `configs/default.yaml`:
- input images (night/rain/snow)
- CSFNet weights
- Ours weights
- Marigold model id (optional)

3) Run:
```bash
python inference_compare.py --config configs/default.yaml
```

Outputs:
- `outputs/qualitative_2x3.png`

## Notes
- Recommended: export CSFNet and Ours to **TorchScript** (`.pt`) and set `mode: "jit"` in the config.
- If you only have `.pth` checkpoints, implement model definitions inside the adapters where indicated.
