'''
[추가 데이터 통합 프로세스: 수집 + 증강 + 분할 + YAML 생성]
1. error_class_list.csv 기준 AIHUB 데이터 서칭 및 크롭
2. 밝기 증강(Brightness Augmentation) 수행
3. Train/Val(8:2) 분할 및 YOLO 표준 폴더 구조(images/train, labels/train 등) 저장
4. train.txt, val.txt 생성
5. YOLOv8/v11 학습용 additional_data.yaml 자동 생성
'''

import json
import cv2
import pandas as pd
import numpy as np
import sys
import os
import random
import shutil
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 경로 설정을 위해 paths 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- 지원 함수 정의 ---

def apply_brightness_aug(img):
    """이미지의 밝기를 무작위로 변경 (0.7 ~ 1.3배)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    ratio = random.uniform(0.7, 1.3)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def crop_and_save_data(img, bbox, padding=0.2):
    """알약 박스를 패딩과 함께 크롭하고 새로운 YOLO 좌표를 생성합니다."""
    H, W = img.shape[:2]
    x, y, w, h = bbox
    
    pad_w, pad_h = w * padding, h * padding
    x1, y1 = max(0, int(x - pad_w)), max(0, int(y - pad_h))
    x2, y2 = min(W, int(x + w + pad_w)), min(H, int(y + h + pad_h))
    
    cropped_img = img[y1:y2, x1:x2]
    new_w, new_h = x2 - x1, y2 - y1
    
    if new_w <= 0 or new_h <= 0:
        return None, None
    
    # 크롭된 이미지 내에서의 상대 좌표(YOLO format)
    cx = (x + w/2 - x1) / new_w
    cy = (y + h/2 - y1) / new_h
    nw = w / new_w
    nh = h / new_h
    
    return cropped_img, [cx, cy, nw, nh]

def create_yolo_yaml(data_root, train_txt, val_txt, class_map_path):
    """YOLO 학습에 필요한 YAML 설정 파일을 생성합니다."""
    try:
        class_map = pd.read_csv(class_map_path)
        # yolo_id 기준 정렬하여 클래스 이름 리스트 생성
        df_sorted = class_map.sort_values('yolo_id')
        names = {int(r.yolo_id): str(r.class_name) for r in df_sorted.itertuples()}
        
        yaml_content = {
            'path': str(data_root.absolute()),
            'train': str(train_txt.name),  # data_root 폴더 내에 위치하므로 파일명만 기재
            'val': str(val_txt.name),
            'nc': len(names),
            'names': names
        }
        
        save_path = paths.CONFIGS_DIR / "additional_data.yaml"
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
            
        return save_path
    except Exception as e:
        logger.error(f"YAML 생성 중 오류 발생: {e}")
        return None

def run_integrated_pipeline():
    # 1. 환경 및 경로 준비
    paths.ensure_dirs()
    data_root = paths.ADDITIONAL_DATA_DIR
    img_root = data_root / "images"
    lbl_root = data_root / "labels"

    if not paths.ERROR_CLASS_LIST_PATH.exists():
        logger.error(f"에러 클래스 리스트 파일이 없습니다: {paths.ERROR_CLASS_LIST_PATH}")
        return

    # 기존 데이터 정리
    if data_root.exists():
        logger.info(f"기존 추가 데이터 정리 중: {data_root}")
        shutil.rmtree(data_root)
    
    for s in ['train', 'val']:
        (img_root / s).mkdir(parents=True, exist_ok=True)
        (lbl_root / s).mkdir(parents=True, exist_ok=True)

    # 2. 메타데이터 로드
    error_df = pd.read_csv(paths.ERROR_CLASS_LIST_PATH)
    target_ids = set(error_df['category_id'].unique())
    
    class_map_df = pd.read_csv(paths.CLASS_MAP_PATH)
    orig_to_yolo = {int(r.orig_id): int(r.yolo_id) for r in class_map_df.itertuples()}

    json_files = list(paths.EDITED_ANNOTATIONS_DIR.rglob("*.json"))
    
    temp_data_storage = []
    box_count = 0
    AUG_COUNT = 2 

    # 3. 데이터 수집 및 증강 로직
    logger.info(f"데이터 수집 및 크롭 시작 (대상 클래스: {len(target_ids)}개)...")
    for json_path in tqdm(json_files, desc="Searching & Cropping"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_name = data['images'][0]['file_name']
            img_p = paths.COLLECTED_IMAGES_DIR / img_name
            if not img_p.exists(): continue
                
            full_img = None
            for ann in data.get('annotations', []):
                cate = ann.get('category_id')
                if cate in target_ids:
                    if full_img is None:
                        full_img = cv2.imread(str(img_p))
                        if full_img is None: break

                    yolo_id = orig_to_yolo.get(cate)
                    cropped, y_bbox = crop_and_save_data(full_img, ann.get('bbox'))
                    if cropped is None: continue

                    # 파일명 베이스 생성
                    base_fn = f"crop_{cate}_{box_count:05d}_{Path(img_name).stem}"
                    
                    # 원본 추가
                    temp_data_storage.append({
                        'img': cropped, 'yolo_id': yolo_id, 'bbox': y_bbox, 'fn': f"{base_fn}_orig"
                    })
                    
                    # 밝기 증강 추가
                    for i in range(AUG_COUNT):
                        temp_data_storage.append({
                            'img': apply_brightness_aug(cropped), 
                            'yolo_id': yolo_id, 
                            'bbox': y_bbox, 
                            'fn': f"{base_fn}_aug_{i}"
                        })
                    box_count += 1
        except Exception:
            continue

    if not temp_data_storage:
        logger.warning("조건에 맞는 데이터를 찾지 못했습니다.")
        return

    # 4. 데이터 분할 및 저장
    logger.info(f"데이터 분할 및 저장 중 (총 샘플: {len(temp_data_storage)})...")
    train_data, val_data = train_test_split(temp_data_storage, test_size=0.2, random_state=42)

    def save_split_to_disk(data_list, split_name):
        full_img_paths = []
        for item in tqdm(data_list, desc=f"Saving {split_name}"):
            save_img_path = img_root / split_name / f"{item['fn']}.png"
            save_lbl_path = lbl_root / split_name / f"{item['fn']}.txt"
            
            # 이미지 저장
            cv2.imwrite(str(save_img_path), item['img'])
            # 라벨 저장
            with open(save_lbl_path, 'w', encoding='utf-8') as f:
                f.write(f"{item['yolo_id']} {' '.join(f'{x:.6f}' for x in item['bbox'])}\n")
            
            # txt 리스트를 위해 절대 경로 저장
            full_img_paths.append(str(save_img_path.absolute()))
            
        # .txt 리스트 파일 생성
        txt_list_path = data_root / f"{split_name}.txt"
        with open(txt_list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_img_paths))
        return txt_list_path

    t_txt_path = save_split_to_disk(train_data, 'train')
    v_txt_path = save_split_to_disk(val_data, 'val')

    # 5. YAML 설정 파일 자동 생성
    logger.info("YOLO 데이터셋 YAML 설정 생성 중...")
    yaml_path = create_yolo_yaml(data_root, t_txt_path, v_txt_path, paths.CLASS_MAP_PATH)

    logger.info("=" * 50)
    logger.info(f"✅ 모든 프로세스 통합 완료!")
    logger.info(f"   - 처리된 원본 박스: {box_count}개")
    logger.info(f"   - 총 생성 데이터: {len(temp_data_storage)}장 (Train: {len(train_data)}, Val: {len(val_data)})")
    logger.info(f"   - 데이터 위치: {data_root}")
    logger.info(f"   - YAML 설정: {yaml_path}")
    logger.info(f"   - 실행 가이드: yolo train data='{yaml_path}' model=yolo11n.pt")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_integrated_pipeline()