'''
[FILE] update_yolo_data_yaml.py
[DESCRIPTION]
    - artifacts/class_map.csv 정보를 읽어 YOLO 학습용 설정 파일(yolo_data.yaml)을 생성 또는 업데이트합니다.
    - 새로운 클래스가 추가되거나 순서가 변경되었을 때 nc(클래스 수)와 names 리스트를 동기화합니다.
[STATUS]
    - 2025-12-19: paths.py 통합 및 리팩토링 완료
    - 특이사항: 'path' 설정은 YOLO 학습 실행 위치에 맞춰 상대 경로로 고정함
'''

from pathlib import Path
import pandas as pd
import yaml
import sys
import os

# 프로젝트 루트 경로 추가 및 paths 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths, logger

def main():
    # 1. 경로 설정 (paths.py 활용)
    # yolo_data.yaml은 초기 학습용 데이터셋(data/yolo)을 가리킵니다.
    yolo_yaml_path = paths.YOLO_DATA_YAML
    yolo_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. 클래스 맵 로드 및 정렬
    if not paths.CLASS_MAP_PATH.exists():
        logger.error(f"❌ 클래스 맵 파일이 없습니다: {paths.CLASS_MAP_PATH}")
        return

    df = pd.read_csv(paths.CLASS_MAP_PATH)
    # yolo_id를 기준으로 정렬해야 YAML 파일 내 인덱스가 일치함
    df = df.sort_values("yolo_id")

    # 3. YAML 내용 구성
    # names: {0: 'class_a', 1: 'class_b', ...} 형식
    names = {int(r.yolo_id): str(r.class_name) for r in df.itertuples()}

    yolo_cfg = {
        "path": "data/yolo",           # 프로젝트 루트 기준 상대 경로
        "train": "images/train",       # path/train
        "val": "images/val",           # path/val
        "nc": int(df["yolo_id"].max() + 1), # 최대 ID + 1 (클래스 총 개수)
        "names": names,
    }

    # 4. YAML 파일 저장 (safe_dump를 사용하여 데이터 안정성 확보)
    with open(yolo_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yolo_cfg, f, allow_unicode=True, sort_keys=False)

    logger.info("-" * 30)
    logger.info(f"✅ YAML 업데이트 완료: {yolo_yaml_path}")
    logger.info(f"🚀 총 클래스 수(nc): {yolo_cfg['nc']}")
    logger.info("-" * 30)

if __name__ == "__main__":
    main()