from ultralytics import YOLO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "configs" / "yolo_data.yaml"

import matplotlib.pyplot as plt
# from src.utils import get_device # 윈도우 옵션
import platform

# healtheat_vision(=REPO_ROOT) 폴더에서 실행 코드 "python -m src.train.train_yolo"
def set_korean_font():
    os_name = platform.system()

    # Windows: 맑은 고딕
    if os_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"

    # macOS: 애플고딕(혹은 Apple SD Gothic Neo)
    elif os_name == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"

    # Linux: 나눔고딕(설치돼 있어야 함)
    else:
        plt.rcParams["font.family"] = "NanumGothic"

    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

def main():
    # device = get_device() # 윈도우 옵션으로 ULTRALYRICS_DEVICE 환경 변수에 따라 결정
    model = YOLO("yolo11n.pt")  # nano로 시작
    model.train(
        data=str(DATA_YAML),
        imgsz=640,
        epochs=100,
        batch=16, #GPU 메모리에 따라 조정
        workers=4, #CPU 코어 수에 따라 조정
        seed=42,
        exist_ok=False, #이어학습 시 True로
        cache="disk", #TRUE 옵션은 Caching images (3.1GB RAM), shared file mapping 1455 + Unable to allocate 2.86 MiB 로그 확인 후 RAM압박으로 인한 에러 발생
        project=str(REPO_ROOT / "artifacts" / "runs"),
        name="yolo11n_main_v2",
        device="cuda:0",
    )

if __name__ == "__main__":
    main()