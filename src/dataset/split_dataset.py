'''
[FILE] split_dataset.py
[DESCRIPTION]
    - 원본 이미지 데이터셋을 지정된 비율(val_ratio)에 따라 Train/Val 세트로 분할합니다.
    - 단순히 랜덤하게 나누는 것을 넘어, 분할 후 검증 세트에 누락된 클래스가 없는지 체크합니다.
    - 결과물로 train.txt와 val.txt에 이미지 파일명(stem) 목록을 저장합니다.
[STATUS]
    - 2025-12-19: paths.py 통합 및 pathlib 기반 리팩토링 완료
    - 특이사항: stratification(층화 추출) 기능을 로그로 확인하며, 불균형이 심할 경우 경고를 띄움
'''

import random
import sys
import json
import logging
from collections import Counter
import os

# 프로젝트 루트 경로 추가 및 paths 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths, logger


def split_dataset(val_ratio=0.1, seed=42):
    logger.info(f"데이터셋 분할 시작 (val_ratio={val_ratio}, seed={seed})")
    
    random.seed(seed)
    
    # 1. 학습 이미지 리스트업
    image_files = list(paths.TRAIN_IMAGES_DIR.glob("*.png"))
    
    if not image_files:
        logger.error(f"이미지를 찾을 수 없습니다: {paths.TRAIN_IMAGES_DIR}")
        return

    stems = [f.stem for f in image_files]
    logger.info(f"총 이미지 개수: {len(stems)}")
    
    # 2. 클래스 분포 분석 (검증 셋 품질 체크용)
    logger.info("클래스 분포 분석 중 (층화 분석)...")
    image_classes = {}
    all_classes_counter = Counter()
    missing_annotations = 0
    
    for stem in stems:
        json_path = paths.TRAIN_ANNOTATIONS_DIR / f"{stem}.json"
        if not json_path.exists():
            missing_annotations += 1
            image_classes[stem] = []
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 카테고리 ID 수집
            cats = [c['id'] for c in data.get('categories', [])]
            image_classes[stem] = cats
            all_classes_counter.update(cats)
            
        except Exception as e:
            logger.error(f"파일 읽기 오류 ({json_path.name}): {e}")
            image_classes[stem] = []

    if missing_annotations > 0:
        logger.warning(f"어노테이션이 없는 이미지가 {missing_annotations}개 발견되었습니다.")

    # 3. 랜덤 분할
    shuffled_stems = stems.copy()
    random.shuffle(shuffled_stems)
    
    num_val = int(len(stems) * val_ratio)
    val_stems = shuffled_stems[:num_val]
    train_stems = shuffled_stems[num_val:]
    
    logger.info(f"분할 완료 - Train: {len(train_stems)}, Val: {len(val_stems)}")
    
    # 4. 분할 품질 검증 (Val 세트에 클래스 누락 여부 확인)
    train_classes = Counter()
    val_classes = Counter()
    
    for s in train_stems:
        train_classes.update(image_classes.get(s, []))
    for s in val_stems:
        val_classes.update(image_classes.get(s, []))
        
    all_class_ids = set(all_classes_counter.keys())
    val_class_ids = set(val_classes.keys())
    missing_in_val = all_class_ids - val_class_ids
    
    logger.info(f"학습 세트 클래스 수: {len(train_classes)}")
    logger.info(f"검증 세트 클래스 수: {len(val_classes)}")
    
    if missing_in_val:
        logger.warning(f"⚠️ 경고: 검증 세트에 {len(missing_in_val)}개의 클래스가 누락되었습니다!")
        logger.debug(f"누락된 클래스: {sorted(list(missing_in_val))}")
        
    # 5. 결과 저장 (paths.py의 SPLITS_DIR 활용)
    paths.ensure_dirs()
    
    train_txt = paths.SPLITS_DIR / "train.txt"
    val_txt = paths.SPLITS_DIR / "val.txt"
    
    train_txt.write_text('\n'.join(train_stems), encoding='utf-8')
    val_txt.write_text('\n'.join(val_stems), encoding='utf-8')
    
    logger.info(f"분할 목록 저장 완료: {train_txt.name}, {val_txt.name}")

if __name__ == "__main__":
    split_dataset()
