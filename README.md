<h1 align="center">Ovarian Tumor Segmentation with SEA-RAFT Post-processing</h1>

## Overview
This repository demonstrates how to refine U‑Net segmentation masks of ovarian tumors by incorporating temporal information through SEA-RAFT optical flow as a post‑processing step. By warping previous-frame masks into the current frame and fusing them with U‑Net outputs, we achieve smoother, more accurate segmentations over time.

## Pipeline Structure
- **Frame-by-Frame Segmentation**: Get the initial segmentation masks from the segmentation model at choice (we used a U-Net)
- **Dense Optical Flow Estimation**: Compute an estimation of the optical flow between frame $t-1$ and frame $t$ using SEA-RAFT
- **Mask Warping**: Warp the previous mask $M_{t-1}$ into the current frame using the optical flow field computed in the previous step (done via bilinear interpolation)
- **Final Mask**: Combine the warped mask with the U‑Net prediction via a weighted average to get the final segmentation mask.


 ## Original SEA-RAFT:
 
repo: https://github.com/princeton-vl/SEA-RAFT 
paper: https://arxiv.org/abs/2405.14793

 ~~~
