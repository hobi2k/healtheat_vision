from pathlib import Path
'''
    프로젝트 전체에서 공통으로 사용하는 경로 정의 및 관리 모듈입니다.
    pathlib 기반 Path 객체를 사용하여 파일 시스템 경로를 다룹니다.
    def ensure_dirs() : 프로젝트 실행에 필요한 필수 폴더 생성
'''
# Project Root (healtheat_vision)
PROJECT_ROOT = Path(__file__).resolve().parents[2] # src/xxx/xxxx.py 기준 -> src 폴더에 내부 분류 안으로 .py를 생성하여야 함

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"

# AIHUB 데이터셋
AIHUB_DIR = DATA_DIR / "aihub_downloads"
# 1. 원본 이미지들이 들어있는 곳
RAW_IMAGES_DIR = AIHUB_DIR / "raw_images"
# 2. 전처리 후 이미지들을 모을 곳
COLLECTED_IMAGES_DIR = AIHUB_DIR / "collected_images"

# 모델 적용 전처리 이전 train/test 데이터 셋
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TRAIN_ANNOTATIONS_DIR = DATA_DIR / "train_annotations"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

# YOLO 데이터셋
YOLO_DIR = DATA_DIR / "yolo"
YOLO_IMAGES_DIR = YOLO_DIR / "images"
YOLO_LABELS_DIR = YOLO_DIR / "labels"

# 모델 구동 (각 모델별 결과는 /run, 결과물 코드 맵핑 class_map.csv)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# yaml 설정 파일
CONFIGS_DIR = PROJECT_ROOT / "configs"

# 제출 파일 저장
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

def ensure_dirs() -> None:
    """프로젝트 실행에 필요한 필수 폴더 생성"""
    for d in [DATA_DIR, COLLECTED_IMAGES_DIR, ARTIFACTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Images: {RAW_IMAGES_DIR}")
    print(f"Collected Images: {COLLECTED_IMAGES_DIR}")
    print(f"Train Images: {TRAIN_IMAGES_DIR}")
    print(f"Train Annotations: {TRAIN_ANNOTATIONS_DIR}")
    print(f"Test Images: {TEST_IMAGES_DIR}")
    print(f"YOLO Images: {YOLO_IMAGES_DIR}")
    print(f"YOLO Labels: {YOLO_LABELS_DIR}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"Configs: {CONFIGS_DIR}")
    print(f"Submissions: {SUBMISSIONS_DIR}")