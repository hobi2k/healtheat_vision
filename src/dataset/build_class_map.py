'''
[FILE] build_class_map.py
[DESCRIPTION]
    - 모든 JSON 어노테이션 파일을 전수 조사하여 고유한 클래스(카테고리) 목록을 추출합니다.
    - AIHub 원본 ID를 YOLO 학습에 적합한 0부터 시작하는 연속된 인덱스(yolo_id)로 매핑합니다.
    - 결과물은 artifacts/class_map.csv에 저장되어 이후 모든 변환 및 학습 과정의 기준점이 됩니다.
[STATUS]
    - 2025-12-19: paths.py 통합 및 pathlib 기반 리팩토링 완료
    - 특이사항: ID와 클래스명이 충돌할 경우 경고 로그를 발생시킴
'''

import json
import pandas as pd
import sys
import os

# 프로젝트 루트 경로 추가 및 paths 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths, logger


def build_class_map():
    logger.info("어노테이션 파일로부터 클래스 맵 구축 시작...")
    
    # 1. 모든 JSON 파일 탐색 (Path.rglob 사용)
    annotation_files = list(paths.TRAIN_ANNOTATIONS_DIR.rglob("*.json"))
    
    if not annotation_files:
        logger.error(f"JSON 파일을 찾을 수 없습니다: {paths.TRAIN_ANNOTATIONS_DIR}")
        return

    unique_classes = {}
    
    # 2. 모든 JSON을 순회하며 유니크 클래스 추출
    for json_file in annotation_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            categories = data.get('categories', [])
            for cat in categories:
                cat_id = cat['id']
                cat_name = cat['name']
                
                if cat_id not in unique_classes:
                    unique_classes[cat_id] = cat_name
                elif unique_classes[cat_id] != cat_name:
                    # 동일 ID에 다른 이름이 매핑되어 있는 데이터 오류 체크
                    logger.warning(f"ID 충돌 발견: ID {cat_id}가 '{unique_classes[cat_id]}'와 '{cat_name}'에 중복 사용됨")
                    
        except Exception as e:
            logger.error(f"파일 읽기 오류 ({json_file.name}): {e}")

    # 3. 원본 ID 기준으로 정렬 (숫자형 정렬 보장)
    sorted_classes = sorted(unique_classes.items(), key=lambda x: int(x[0]))
    
    # 4. YOLO ID 부여 (0번부터 순차적 할당)
    class_map_data = []
    for idx, (orig_id, name) in enumerate(sorted_classes):
        class_map_data.append({
            'orig_id': int(orig_id),
            'yolo_id': idx,
            'class_name': name
        })
        
    df = pd.DataFrame(class_map_data)
    
    # 5. 결과 저장 (artifacts/class_map.csv)
    paths.ensure_dirs()
    df.to_csv(paths.CLASS_MAP_PATH, index=False, encoding='utf-8-sig') # 한글 깨짐 방지
    
    logger.info(f"클래스 맵 저장 완료: {paths.CLASS_MAP_PATH}")
    logger.info(f"총 고유 클래스 수: {len(df)}")
    
    return df

if __name__ == "__main__":
    build_class_map()
