import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths

def crop_and_save(img, bbox, padding=0.2):
    """
    COCO bbox [x, y, w, h]를 받아 패딩을 포함해 크롭하고 YOLO 포맷 좌표를 반환합니다.
    """
    H, W = img.shape[:2]
    x, y, w, h = bbox
    
    # 패딩 계산 (bbox 크기 대비 비율)
    pad_w = w * padding
    pad_h = h * padding
    
    # 정수 좌표 변환 및 이미지 경계 방어
    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(W, int(x + w + pad_w))
    y2 = min(H, int(y + h + pad_h))
    
    cropped_img = img[y1:y2, x1:x2]
    
    # 크롭된 이미지의 새로운 크기
    new_w = x2 - x1
    new_h = y2 - y1
    
    if new_w <= 0 or new_h <= 0:
        return None, None
    
    # 크롭 이미지 내에서의 YOLO 상대 좌표 (cx, cy, nw, nh)
    # 원본 박스의 중심점 계산 후 크롭 시작점(x1, y1)을 빼줌
    cx = (x + w/2 - x1) / new_w
    cy = (y + h/2 - y1) / new_h
    nw = w / new_w
    nh = h / new_h
    
    return cropped_img, [cx, cy, nw, nh]

def process_additional_data():
    paths.ensure_dirs()
    
    # 1. 타겟 에러 클래스 로드 (category_id 컬럼 사용)
    if not paths.ERROR_CLASS_LIST_PATH.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {paths.ERROR_CLASS_LIST_PATH}")
        return
        
    error_df = pd.read_csv(paths.ERROR_CLASS_LIST_PATH)
    target_ids = set(error_df['category_id'].unique())
    print(f"🎯 타겟 클래스 로드 완료 (ID 수: {len(target_ids)})")
    
    # 2. 전체 클래스 맵 로드 (orig_id -> yolo_id 매칭용)
    class_map_df = pd.read_csv(paths.CLASS_MAP_PATH)
    # class_map.csv는 orig_id 컬럼명을 사용한다고 하셨으므로 r.orig_id로 접근
    orig_to_yolo = {int(r.orig_id): int(r.yolo_id) for r in class_map_df.itertuples()}

    # 3. JSON 파일 리스트업
    json_files = list(paths.EDITED_ANNOTATIONS_DIR.rglob("*.json"))
    print(f"📂 스캔 시작: {len(json_files)}개의 JSON")

    count = 0
    missing_images = 0
    
    for json_path in tqdm(json_files, desc="Pill Cropping"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data.get('images'): continue
            
            img_name = data['images'][0]['file_name']
            img_path = paths.COLLECTED_IMAGES_DIR / img_name
            
            # 이미지 존재 여부 확인
            if not img_path.exists():
                missing_images += 1
                continue
                
            full_img = None # 실제 매칭될 때만 로드하기 위해 지연 할당

            # 이미지 내 모든 어노테이션 확인
            for ann in data.get('annotations', []):
                cate = ann.get('category_id') # JSON 내 키값
                
                if cate in target_ids:
                    # 매칭되는 클래스가 있으면 그제서야 이미지 로드 (속도 최적화)
                    if full_img is None:
                        full_img = cv2.imread(str(img_path))
                        if full_img is None: break

                    yolo_id = orig_to_yolo.get(cate)
                    bbox = ann.get('bbox') # [x, y, w, h]
                    
                    if not bbox or len(bbox) != 4: continue
                    
                    # 크롭 및 좌표 변환
                    cropped_img, yolo_bbox = crop_and_save(full_img, bbox)
                    
                    if cropped_img is not None:
                        # 저장 파일명: crop_{카테고리ID}_{순번}_{원본파일명}
                        base_name = f"crop_{cate}_{count:05d}_{Path(img_name).stem}"
                        save_img_path = paths.ADDITIONAL_TRAIN_IMG_DIR / f"{base_name}.png"
                        save_txt_path = paths.ADDITIONAL_TRAIN_ANN_DIR / f"{base_name}.txt"
                        
                        # 파일 저장
                        cv2.imwrite(str(save_img_path), cropped_img)
                        with open(save_txt_path, 'w', encoding='utf-8') as f_txt:
                            f_txt.write(f"{yolo_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                        
                        count += 1
                        
        except Exception as e:
            # 개별 파일 에러가 전체 중단되지 않도록 pass
            continue

    print("\n" + "="*50)
    print(f"✅ 추가 데이터 생성 완료!")
    print(f"📦 생성된 크롭 이미지/라벨: {count}쌍")
    if missing_images > 0:
        print(f"⚠️ 이미지를 찾지 못한 JSON: {missing_images}개")
    print(f"📍 위치: {paths.ADDITIONAL_TRAIN_IMG_DIR}")
    print("="*50)

if __name__ == "__main__":
    process_additional_data()