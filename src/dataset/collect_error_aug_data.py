import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths

def apply_brightness_aug(img):
    """
    이미지의 밝기를 무작위로 변경합니다. (0.7 ~ 1.3배)
    """
    # HSV 색공간으로 변환하여 V(Value, 밝기) 채널 조정
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    
    # 무작위 배수 생성 (0.7 ~ 1.3)
    ratio = random.uniform(0.7, 1.3)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    
    # 255를 넘지 않도록 제한
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def crop_and_save(img, bbox, padding=0.2):
    H, W = img.shape[:2]
    x, y, w, h = bbox
    
    pad_w = w * padding
    pad_h = h * padding
    
    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(W, int(x + w + pad_w))
    y2 = min(H, int(y + h + pad_h))
    
    cropped_img = img[y1:y2, x1:x2]
    new_w, new_h = x2 - x1, y2 - y1
    
    if new_w <= 0 or new_h <= 0:
        return None, None
    
    cx = (x + w/2 - x1) / new_w
    cy = (y + h/2 - y1) / new_h
    nw = w / new_w
    nh = h / new_h
    
    return cropped_img, [cx, cy, nw, nh]

def process_additional_data():
    paths.ensure_dirs()
    
    # 1. 설정 및 로드
    error_df = pd.read_csv(paths.ERROR_CLASS_LIST_PATH)
    target_ids = set(error_df['category_id'].unique())
    
    class_map_df = pd.read_csv(paths.CLASS_MAP_PATH)
    orig_to_yolo = {int(r.orig_id): int(r.yolo_id) for r in class_map_df.itertuples()}

    json_files = list(paths.EDITED_ANNOTATIONS_DIR.rglob("*.json"))
    
    count = 0
    # 하나의 박스당 생성할 증강 이미지 수 (원본 1 + 밝기변형 2 = 총 3장)
    AUG_COUNT = 2 

    for json_path in tqdm(json_files, desc="Pill Cropping & Augmenting"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_name = data['images'][0]['file_name']
            img_path = paths.COLLECTED_IMAGES_DIR / img_name
            if not img_path.exists(): continue
                
            full_img = None

            for ann in data.get('annotations', []):
                cate = ann.get('category_id')
                
                if cate in target_ids:
                    if full_img is None:
                        full_img = cv2.imread(str(img_path))
                        if full_img is None: break

                    yolo_id = orig_to_yolo.get(cate)
                    bbox = ann.get('bbox')
                    
                    # 2. 원본 크롭 생성
                    cropped_img, yolo_bbox = crop_and_save(full_img, bbox)
                    if cropped_img is None: continue

                    # 3. 데이터 저장 (원본 + 증강본)
                    # 원본 저장
                    versions = [("orig", cropped_img)]
                    # 밝기 증강본 추가
                    for i in range(AUG_COUNT):
                        versions.append((f"aug_br_{i}", apply_brightness_aug(cropped_img)))

                    for suffix, img_v in versions:
                        base_name = f"crop_{cate}_{count:05d}_{suffix}_{Path(img_name).stem}"
                        save_img_path = paths.ADDITIONAL_TRAIN_IMG_DIR / f"{base_name}.png"
                        save_txt_path = paths.ADDITIONAL_TRAIN_ANN_DIR / f"{base_name}.txt"
                        
                        cv2.imwrite(str(save_img_path), img_v)
                        with open(save_txt_path, 'w', encoding='utf-8') as f_txt:
                            f_txt.write(f"{yolo_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                    
                    count += 1
                        
        except Exception:
            continue

    print(f"\n✅ 완료! 총 {count}개의 타겟 박스에 대해 증강 포함 {count * (1 + AUG_COUNT)}장의 이미지를 생성했습니다.")

if __name__ == "__main__":
    process_additional_data()