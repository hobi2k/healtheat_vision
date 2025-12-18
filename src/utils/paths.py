from pathlib import Path
'''
    프로젝트 전체에서 공통으로 사용하는 경로 정의 및 관리 모듈입니다.
    pathlib 기반 Path 객체를 사용하여 파일 시스템 경로를 다룹니다.
    def ensure_dirs() : 프로젝트 실행에 필요한 필수 폴더 생성
'''
# Project Root (healtheat_vision)
# src/utils/paths.py 기준 -> 상위로 두 번 이동 (utils -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_ANNOTATIONS_DIR = DATA_DIR / "train_annotations"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"

# YOLO 데이터셋 (yolo_data.yaml에서 참조할 경로)
YOLO_DIR = DATA_DIR / "yolo"
YOLO_DATA_YAML = PROJECT_ROOT / "configs" / "yolo_data.yaml"
# "yolo_data.yaml" 내에 path: data/yolo, train: images/train, val: images/val 로 설정되어 있어야 합니다.
    # 학습/검증 데이터셋 Yolo 분할 (참고):  DATA_DIR / "splits" 폴더에 train.txt, val.txt 파일이 있습니다.

# AIHub 데이터셋
AIHUB_DIR = DATA_DIR / "aihub_downloads"
RAW_IMAGES_DIR = AIHUB_DIR / "raw_images"
COLLECTED_IMAGES_DIR = AIHUB_DIR / "collected_images"

# 모델 설정 관련 (추가)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"  # 원본 모델 저장 폴더
CONFIGS_DIR = PROJECT_ROOT / "configs"
HPARAMS_DIR = CONFIGS_DIR / "hparams" # 하이퍼파라미터 YAML들만 따로 모을 경우

# 실험 결과 관련
RUNS_DIR = ARTIFACTS_DIR / "runs" # 학습 결과(weights, charts)가 저장될 곳

# 제출 파일 저장
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

def ensure_dirs() -> None:
    """프로젝트 실행에 필요한 필수 폴더 생성"""
    # 실험 기록을 위한 runs 폴더와 설정 폴더 등을 목록에 추가
    dirs = [
        DATA_DIR, 
        COLLECTED_IMAGES_DIR, 
        MODELS_DIR,
        ARTIFACTS_DIR, 
        RUNS_DIR, 
        CONFIGS_DIR,
        SUBMISSIONS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # 경로 확인용 로그
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"YOLO Data YAML: {YOLO_DATA_YAML}")
    print(f"Runs Directory: {RUNS_DIR}")