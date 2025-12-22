'''
[FILE] temp_yolo_val.py
[목적]
    - 모델 간 평가를 위한 별도 데이터셋 구축을 위해 활용된 코드입니다.
    - AIHub 데이터셋의 TS_6번 데이터셋을 활용했습니다.
[DESCRIPTION]
    - AIHub 데이터셋의 COCO JSON 어노테이션을 YOLO txt 형식으로 변환합니다.
    - 데이터셋을 train/val split 리스트(txt)에 따라 물리적으로 분류합니다.
[STATUS]
    - 2025-12-22: 초기 구현
    - 절대경로 리펙토링 이전
'''

import json
import shutil
import os
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# 1. 절대 경로 설정
IMG_SRC_DIR = Path('/Users/youuchul/Documents/github/03_projects/01_HealthEat Pill Detection Model/healtheat_vision/data/aihub_downloads/collected_images_val')
JSON_SRC_DIR = Path('/Users/youuchul/Documents/github/03_projects/01_HealthEat Pill Detection Model/healtheat_vision/data/aihub_downloads/annotations_json_edited')
CLASS_MAP_PATH = Path('/Users/youuchul/Documents/github/03_projects/01_HealthEat Pill Detection Model/healtheat_vision/artifacts/class_map.csv')

# 최종 결과물이 모일 곳
OUT_ROOT = Path('/Users/youuchul/Documents/github/03_projects/01_HealthEat Pill Detection Model/healtheat_vision/data/aihub_downloads/yolo_val_ready')
OUT_IMG_DIR = OUT_ROOT / "images"
OUT_LBL_DIR = OUT_ROOT / "labels"

def load_class_map(path: Path):
    df = pd.read_csv(path)
    return {int(r.orig_id): int(r.yolo_id) for r in df.itertuples()}

def xywh_to_yolo(x, y, w, h, W, H):
    return (x + w / 2.0) / W, (y + h / 2.0) / H, w / W, h / H

def process_yolo_conversion():
    # 폴더 생성
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 클래스 맵 로드
    orig_to_yolo = load_class_map(CLASS_MAP_PATH)

    # 2. 대상 이미지 목록 확보
    image_paths = {f.stem: f for f in IMG_SRC_DIR.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
    print(f"[INFO] 대상 이미지 개수: {len(image_paths)}개")

    # 3. JSON 소스 전체 인덱싱 (하나의 이미지에 여러 JSON이 있으므로 리스트로 수집)
    print("[INFO] JSON 어노테이션 인덱싱 중...")
    ann_index = defaultdict(list)
    for jp in JSON_SRC_DIR.rglob("*.json"):
        if jp.stem in image_paths:
            ann_index[jp.stem].append(jp)

    # 4. 변환 및 복사 루프
    print("[INFO] YOLO 변환 및 파일 수집 시작...")
    success_count = 0

    for stem, img_path in tqdm(image_paths.items()):
        json_list = ann_index.get(stem)
        if not json_list:
            continue

        yolo_lines = []
        
        # 이미지의 모든 JSON 파일을 열어 어노테이션 통합
        for json_path in json_list:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not data.get("images"): continue
                img_info = data["images"][0]
                W, H = float(img_info.get("width")), float(img_info.get("height"))

                for ann in data.get("annotations", []):
                    orig_id = int(ann.get("category_id"))
                    if orig_id not in orig_to_yolo: continue

                    bbox = ann.get("bbox") # [x, y, w, h]
                    if not bbox or len(bbox) != 4: continue

                    x, y, w, h = map(float, bbox)
                    
                    # 정규화 및 변환
                    cx, cy, nw, nh = xywh_to_yolo(x, y, w, h, W, H)
                    yolo_lines.append(f"{orig_to_yolo[orig_id]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            except Exception:
                continue

        if yolo_lines:
            # 1. 이미지 복사
            shutil.copy2(img_path, OUT_IMG_DIR / img_path.name)
            # 2. 통합된 YOLO 라벨 저장
            label_path = OUT_LBL_DIR / f"{stem}.txt"
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines) + "\n")
            
            success_count += 1

    print("-" * 50)
    print(f"[FINISH] 작업 완료")
    print(f"- 최종 변환 성공 이미지: {success_count}개")
    print(f"- 결과 저장 경로: {OUT_ROOT}")
    print("-" * 50)

if __name__ == "__main__":
    process_yolo_conversion()