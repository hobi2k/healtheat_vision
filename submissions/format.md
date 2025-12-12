# Submission Format

This document defines the strict format for submission files in the HealthEat Pill Detection task.

## File Naming
`submission_{YYYYMMDD_HHMMSS}.csv` (e.g., `submission_20251212_183000.csv`)

## File Format
- **Format**: CSV (Comma Separated Values)
- **Header**: `Image_ID,Prediction_String` (Case sensitive)
- **Rows**: One row per Test Image.

## Columns

| Column Name | Description | Example |
| :--- | :--- | :--- |
| `Image_ID` | The base filename (stem) of the test image. | `train_10000` |
| `Prediction_String` | Space-separated list of detections. format: `class_id score xcenter ycenter w h` | `10 0.95 0.5 0.5 0.1 0.2` |

## Detailed Rules

1. **Image_ID**:
   - Must match the filename without extension of files in `data/test_images`.
   - Example: if file is `test_001.png`, Image_ID is `test_001`.

2. **Prediction_String**:
   - Format: `class_id confidence x y w h` repeated for each box.
   - `class_id`: **Original Class ID** (NOT the YOLO 0-indexed ID).
   - `confidence`: Float between 0 and 1.
   - `x, y, w, h`: **Relative coordinates** (0.0 to 1.0) normalized by image width/height.
     - `x`: Center X
     - `y`: Center Y
     - `w`: Width
     - `h`: Height
   - If no objects are detected, leave field empty (or check specific competition rules if '14 0 0 0 0 0' required - assumed empty is valid for now).

3. **Coordinate Space**:
   - Predictions must be normalized (0-1).

## Example content
```csv
Image_ID,Prediction_String
test_0001,15 0.98 0.45 0.30 0.10 0.10 45 0.88 0.70 0.60 0.05 0.10
test_0002,
test_0003,12 0.55 0.1 0.1 0.2 0.2
```
