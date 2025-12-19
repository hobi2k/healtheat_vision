from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils import get_device

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "configs" / "yolo_data.yaml"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

def main():
    device = get_device()

    run_dir = REPO_ROOT / "artifacts" / "runs" / "yolo11n_main_v1"
    last_pt = run_dir / "weights" / "last.pt"

    model = YOLO(str(last_pt))
    model.train(
        data=str(DATA_YAML),
        resume=str(last_pt),
        imgsz=640,
        epochs=100,
        batch=4,
        workers=0,
        seed=42,
        cache=False,
        project=str(REPO_ROOT / "artifacts" / "runs"),
        name="yolo11n_main_v1",
        exist_ok=True,
        device=device,
    )

if __name__ == "__main__":
    main()