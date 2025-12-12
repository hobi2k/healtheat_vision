from ultralytics import YOLO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "configs" / "yolo_data.yaml"

import matplotlib.pyplot as plt
from src.utils import get_device

plt.rcParams["font.family"] = "AppleGothic"  # macOS 기본 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

def main():
    device = get_device()
    model = YOLO("yolo11n.pt")  # nano로 시작
    model.train(
        data=str(DATA_YAML),
        imgsz=640,
        epochs=1,
        batch=8,
        workers=2,
        seed=42,
        cache=False,
        project=str(REPO_ROOT / "artifacts" / "runs"),
        name="yolo11n_smoke",
        device=device,
    )

if __name__ == "__main__":
    main()