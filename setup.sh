#!/usr/bin/env bash
# Downloads the model weights the notebooks expect, into notebooks/ (paths match the notebook cells).
# Run from the repo root after `pip install -r requirements.txt`.
set -euo pipefail
cd "$(dirname "$0")/notebooks"

mkdir -p checkpoints
if [ ! -f checkpoints/sam2.1_hiera_large.pt ]; then
  echo "Downloading SAM 2.1 (hiera large, ~900 MB)..."
  curl -L -o checkpoints/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

if [ ! -d Depth-Anything-V2 ]; then
  echo "Cloning Depth Anything V2..."
  git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git
fi
if [ ! -f depth_anything_v2_vitb.pth ]; then
  echo "Downloading Depth Anything V2 (ViT-B, ~390 MB)..."
  curl -L -o depth_anything_v2_vitb.pth \
    https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
fi

echo "Done. PiDiNet and CLIP weights download automatically on first use."
