"""
YOLO 모델 학습 메인 스크립트

실행 방법:
    터미널에서 프로젝트 루트(healtheat_vision) 폴더로 이동 후 아래 명령어 실행
    python -m src.train.train_yolo --cfg [설정파일명.yaml]

예시:
    python -m src.train.train_yolo --cfg hparams_v11.yaml
    python -m src.train.train_yolo --cfg hparams_v8.yaml
"""

import yaml
import argparse
from ultralytics import YOLO

# utils에서 logger와 나머지 도구들을 가져옵니다.
from src.utils import (
    YOLO_DATA_YAML, CONFIGS_DIR, RUNS_DIR, MODELS_DIR,
    get_device, set_korean_font, ensure_dirs,
    logger  # 로거 추가
)

def main():
    # 0. 환경 체크
    ensure_dirs()
    set_korean_font()
    
    # 1. 인자 설정
    parser = argparse.ArgumentParser(description="HealthEat Vision YOLO Training")
    parser.add_argument("--cfg", type=str, required=True, help="configs/ 폴더 내 YAML 파일명")
    args = parser.parse_args()

    config_path = CONFIGS_DIR / args.cfg
    if not config_path.exists():
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}") # print 대신 error 로그
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        hp = yaml.safe_load(f)

    # 2. 장치 탐지 및 로그 기록
    device = get_device()
    logger.info(f"사용 가능 장치 탐지 완료: {device}")

    # 3. 필수 인자 검증
    try:
        model_path = hp['model']
        epochs = hp['epochs']
        name = hp['name']
    except KeyError as e:
        logger.error(f"필수 항목 누락됨 ({args.cfg}): {e}")
        return

    # 4. 학습 시작 정보 로그 기록
    logger.info(f"--- 학습 시작: {name} ---")
    logger.info(f"Model: {model_path} | Epochs: {epochs} | Device: {device}")

    # 5. 모델 로드 및 학습
    model = YOLO(model_path)
    
    # results는 학습 후 결과 객체입니다.
    results = model.train(
        data=str(YOLO_DATA_YAML),
        imgsz=hp.get('imgsz', 640),
        epochs=epochs,
        batch=hp.get('batch', 16),
        optimizer=hp.get('optimizer', 'auto'),
        project=str(RUNS_DIR),
        name=name,
        device=device,
        cache="disk",
        seed=42,
        workers=hp.get('workers', 4),
        patience=hp.get('patience', 50),
        val=True,
        cos_lr=hp.get('cos_lr', True),
        exist_ok=False,
    )
    
    logger.info(f"--- 학습 완료: {name} ---")
    logger.info(f"결과 저장 경로: {RUNS_DIR / name}")

if __name__ == "__main__":
    main()