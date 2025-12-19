from pathlib import Path
'''
    프로젝트 전체에서 공통으로 사용하는 경로 정의 및 관리 모듈입니다.
    pathlib 기반 Path 객체를 사용하여 파일 시스템 경로를 다룹니다.
    def ensure_dirs() : 프로젝트 실행에 필요한 필수 폴더 생성
'''
# Project Root (healtheat_vision)
# src/utils/paths.py 기준 -> 상위로 두 번 이동 (utils -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- 기본 데이터 디렉토리 ---
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_ANNOTATIONS_DIR = DATA_DIR / "train_annotations" # 초기 학습용 JSON
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"           # 초기 학습용 이미지
SPLITS_DIR = DATA_DIR / "splits"                       # train.txt, val.txt (초기 데이터용)

# --- YOLO 표준 데이터셋 (초기 학습용) ---
YOLO_DIR = DATA_DIR / "yolo" # YOLO 표준 데이터셋 기준 하위 디렉토리(images, labels는 해당 변환코드에서 생성)
YOLO_DATA_YAML = PROJECT_ROOT / "configs" / "yolo_data.yaml"

# AIHub 데이터셋
AIHUB_DIR = DATA_DIR / "aihub_downloads" # AIHub 데이터셋이 다운로드된 폴더
RAW_IMAGES_DIR = AIHUB_DIR / "raw_images" # AIHub 원본 이미지 폴더
COLLECTED_IMAGES_DIR = AIHUB_DIR / "collected_images" # AIHub 수집 이미지 폴더
RAW_ANNOTATIONS_DIR = AIHUB_DIR / "raw_annotations" # AIHub 원본 라벨 폴더
EDITED_ANNOTATIONS_DIR = AIHUB_DIR / "annotations_json_edited" # AIHub 수정된 라벨 폴더

# 모델 설정 관련 (추가)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"  # 원본 모델 저장 폴더
CONFIGS_DIR = PROJECT_ROOT / "configs"

# 클래스 맵 경로
CLASS_MAP_PATH = ARTIFACTS_DIR / "class_map.csv"

# 실험 결과 관련
RUNS_DIR = ARTIFACTS_DIR / "runs" # 학습 결과(weights, charts)가 저장될 곳

# 제출 파일 저장
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# 추가 학습 관련 경로 정의
ERROR_CLASS_LIST_PATH = ARTIFACTS_DIR / "error_class_list.csv" # 에러 클래스 위치
ADDITIONAL_DATA_DIR = DATA_DIR / "additional_yolo"  # 전체 추가 데이터 루트



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
        SUBMISSIONS_DIR,
        ADDITIONAL_DATA_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # 경로 확인용 로그
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"YOLO Data YAML: {YOLO_DATA_YAML}")
    print(f"Runs Directory: {RUNS_DIR}")