'''
[FILE] split_check_class.py
[DESCRIPTION]
    - 분할된 YOLO 데이터셋(train/val)의 라벨 파일을 전수 조사하여 클래스별 분포를 확인합니다.
    - 각 클래스당 박스(BBox) 개수와 해당 클래스가 포함된 이미지 파일 개수를 집계합니다.
    - 특정 클래스가 train이나 val 세트에서 누락되었는지 체크하여 리포트를 생성합니다.
[STATUS]
    - 2025-12-19: paths.py 통합 및 로직 최적화 완료
    - 결과물: artifacts/split_class_counts.csv (또는 지정된 csv 폴더)
'''

import sys
import logging
from pathlib import Path
from collections import Counter
import pandas as pd
import os

# 프로젝트 루트 경로 추가 및 paths 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths, logger


def scan_labels(label_dir: Path) -> tuple[Counter, Counter, int]:
    """
    라벨 디렉토리를 스캔하여 클래스별 박스 수와 파일 등장 횟수를 카운트합니다.
    """
    box_count = Counter()
    file_count = Counter()
    bad_lines = 0

    if not label_dir.exists():
        logger.warning(f"라벨 폴더가 존재하지 않습니다: {label_dir}")
        return box_count, file_count, bad_lines

    txt_paths = list(label_dir.rglob("*.txt"))
    for p in txt_paths:
        seen_in_file = set()
        try:
            # errors="replace"를 통해 인코딩 에러 방지
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            logger.error(f"파일 읽기 실패 ({p.name}): {e}")
            continue

        for ln in lines:
            parts = ln.strip().split()
            if len(parts) < 5: # YOLO 형식 (cls, cx, cy, w, h) 미달 시
                if parts: bad_lines += 1
                continue

            try:
                yid = int(parts[0])
                box_count[yid] += 1
                seen_in_file.add(yid)
            except ValueError:
                bad_lines += 1

        for yid in seen_in_file:
            file_count[yid] += 1

    return box_count, file_count, bad_lines

def main():
    logger.info("YOLO 데이터셋 클래스 분포 검증 시작...")
    
    # 1. 클래스 맵 로드 (paths.py 활용)
    if not paths.CLASS_MAP_PATH.exists():
        logger.error(f"클래스 맵이 없습니다: {paths.CLASS_MAP_PATH}")
        return
    
    class_map = pd.read_csv(paths.CLASS_MAP_PATH)
    
    # 2. 경로 설정 (paths.py의 YOLO_DIR 하위)
    label_train_dir = paths.YOLO_DIR / "labels" / "train"
    label_val_dir   = paths.YOLO_DIR / "labels" / "val"

    # 3. 라벨 스캔 실행
    train_box, train_file, train_bad = scan_labels(label_train_dir)
    val_box,   val_file,   val_bad   = scan_labels(label_val_dir)

    # 4. 데이터프레임 구축
    df = class_map.copy()
    df["train_box_count"]  = df["yolo_id"].map(train_box).fillna(0).astype(int)
    df["train_file_count"] = df["yolo_id"].map(train_file).fillna(0).astype(int)
    df["val_box_count"]    = df["yolo_id"].map(val_box).fillna(0).astype(int)
    df["val_file_count"]   = df["yolo_id"].map(val_file).fillna(0).astype(int)

    # 누락 여부 체크
    df["missing_in_train"] = df["train_box_count"] == 0
    df["missing_in_val"]   = df["val_box_count"] == 0

    # 5. 결과 저장 (artifacts 폴더 내에 저장하는 것을 추천)
    out_dir = paths.ARTIFACTS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "split_class_counts.csv"
    
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 6. 결과 요약 출력
    missing_train = df[df["missing_in_train"]]
    missing_val   = df[df["missing_in_val"]]

    logger.info("=" * 50)
    logger.info(f"검증 완료! 리포트 저장 위치: {out_path}")
    logger.info(f"파싱 실패 라인 수: Train={train_bad}, Val={val_bad}")
    logger.info(f"Train 세트 누락 클래스: {len(missing_train)}개")
    logger.info(f"Val 세트 누락 클래스: {len(missing_val)}개")
    
    if len(missing_val) > 0:
        logger.warning("검증(Val) 세트에 없는 클래스가 발견되었습니다. 학습 결과에 영향을 줄 수 있습니다.")
        logger.info(missing_val[["yolo_id", "class_name"]].to_string(index=False))
    logger.info("=" * 50)

if __name__ == "__main__":
    main()