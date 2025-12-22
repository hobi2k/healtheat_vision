# Data Directory

This directory contains the dataset for the HealthEat Pill Detection Model.

## Directory Structure
- `train_images/`: Contains the training images (`*.png`).
- `train_annotations/`: Contains COCO-like format JSON annotations.
- `test_images/`: Contains header-only test images for inference.
- `yolo/`: (Generated) Converted YOLO format dataset.
- `splits/`: (Generated) Train/Val split indices.

## Instructions
1. Place extracted training images in `train_images/`.
2. Place extracted JSON annotations in `train_annotations/`.
3. Place test images in `test_images/`.

**Note**: Most files in this directory are git-ignored to prevent large file uploads.
