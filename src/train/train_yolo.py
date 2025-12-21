"""
YOLO 모델 학습 메인 스크립트

주요 기능:
지정된 YAML 설정 파일을 로드하여 YOLO 모델 학습 수행

실행 방법:
    터미널에서 프로젝트 루트(healtheat_vision) 폴더로 이동 후 아래 명령어 실행
    python -m src.train.train_yolo --cfg [설정파일명.yaml]

예시:
    python -m src.train.train_yolo --cfg hparams_v11.yaml
    python -m src.train.train_yolo --cfg hparams_v8.yaml
    python -m src.train.train_yolo --cfg hparams_additional.yaml
    python -m src.train.train_yolo --cfg hparams_v11_finetune.yaml
    python -m src.train.train_yolo --cfg hparams_v11s.yaml
    python -m src.train.train_yolo --cfg hparams_v11s_filtered_ft.yaml
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

    # 5. 모델 로드
    model = YOLO(model_path)
    
    # 하이퍼파라미터 YAML에 정의된 데이터 설정 파일명을 가져옵니다.
    # 예: hparams_additional.yaml에 'data_yaml: additional_data.yaml' 이라고 써있으면 그걸 사용
    data_cfg_name = hp.get('data_yaml', 'yolo_data.yaml') 
    
    # 실제 경로: CONFIGS_DIR (configs/) + 파일명
    actual_data_path = CONFIGS_DIR / data_cfg_name
    
    if not actual_data_path.exists():
        logger.error(f"데이터 설정 파일을 찾을 수 없습니다: {actual_data_path}")
        return

    # 6. 학습 실행
    results = model.train(
        data=str(actual_data_path),
        imgsz=hp.get('imgsz', 640),
        epochs=epochs,
        batch=hp.get('batch', 16),
        optimizer=hp.get('optimizer', 'auto'),
        project=str(RUNS_DIR),
        lr0=hp.get('lr0', 0.01),
        lrf=hp.get('lrf', 0.01),          # ✅ 추가 권장: 최종 학습률 감쇄 비율
        fliplr=hp.get('fliplr', 0.5),
        flipud=hp.get('flipud', 0.0),
        degrees=hp.get('degrees', 0.0),
        mosaic=hp.get('mosaic', 1.0),
        mixup=hp.get('mixup', 0.0),        # ✅ 추가 권장: YAML의 mixup: 0.1을 반영하기 위함
        scale=hp.get('scale', 0.5),        # ✅ 추가 권장: YAML의 scale 반영
        name=name,
        device=device,
        cache="disk",
        seed=42,
        workers=hp.get('workers', 4),
        patience=hp.get('patience', 50),
        warmup_epochs=hp.get('warmup_epochs', 3.0), # ✅ 추가 권장
        val=True,
        cos_lr=hp.get('cos_lr', True),
        exist_ok=False,
    )
    
    logger.info(f"--- 학습 완료: {name} ---")
    logger.info(f"결과 저장 경로: {RUNS_DIR / name}")

if __name__ == "__main__":
    main()