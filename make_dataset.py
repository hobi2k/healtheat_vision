import json
import os
import glob
from tqdm import tqdm  # 진행상황을 보여주는 막대기 라이브러리

# ==========================================
# 설정: 경로를 사용자 환경에 맞게 수정
# ==========================================
# 원본 노란 폴더들이 들어있는 곳
RAW_DATA_DIR = "./data/train_annotations" 
# 결과를 저장할 파일 이름
OUTPUT_FILE = "./data/train_annotations.json"

def main():
    # 1. 데이터를 담을 그릇 만들기 (COCO 포맷 기준)
    coco_format = {
        "info": {"description": "HealthEat Project Dataset"},
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 카테고리 중복 방지를 위한 사전 (예: '보령부스파정'이 여러 번 나와도 ID는 하나로 통일)
    category_map = {} 
    
    # ID가 겹치지 않게 0부터 새로 번호를 매깁니다
    image_id_counter = 0
    annotation_id_counter = 0
    category_id_counter = 0

    # 2. 모든 JSON 파일 찾기 (하위 폴더까지 싹 다 뒤짐)
    print(" 흩어진 파일들을 찾는 중입니다... 잠시만 기다려주세요.")
    # data/train_annotations 폴더 안의 모든 .json 파일을 찾음
    json_files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.json"), recursive=True)
    
    print(f"총 {len(json_files)}개의 파일을 찾았습니다! 합치기를 시작합니다.")

    # 3. 파일 하나하나 열어서 합치기
    for json_file in tqdm(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- 이미지 정보 처리 ---
            # 원본 파일에서 이미지 정보 가져오기
            raw_img = data['images'][0]
            
            # 새로운 이미지 ID 부여 (0, 1, 2...)
            current_image_id = image_id_counter
            image_id_counter += 1

            # 우리가 필요한 정보만 쏙 뽑아서 담기
            img_info = {
                "id": current_image_id,
                "file_name": raw_img['file_name'],
                "width": raw_img['width'],
                "height": raw_img['height']
            }
            coco_format['images'].append(img_info)

            # --- 카테고리(알약 이름) 처리 ---
            # 원본 데이터에 카테고리 정보가 있는 경우만
            if 'categories' in data:
                raw_cat = data['categories'][0]
                cat_name = raw_cat['name']

                # 처음 보는 알약 이름이면 등록
                if cat_name not in category_map:
                    category_map[cat_name] = category_id_counter
                    
                    # 카테고리 정보 추가
                    coco_format['categories'].append({
                        "id": category_id_counter,
                        "name": cat_name,
                        "supercategory": "pill"
                    })
                    category_id_counter += 1
                
                # 이미 등록된 알약이면 그 ID를 사용
                current_category_id = category_map[cat_name]

            # --- 라벨링(박스) 정보 처리 ---
            if 'annotations' in data:
                for raw_ann in data['annotations']:
                    ann_info = {
                        "id": annotation_id_counter,
                        "image_id": current_image_id,  # 위에서 만든 새 이미지 ID 연결
                        "category_id": current_category_id, # 위에서 찾은 카테고리 ID 연결
                        "bbox": raw_ann['bbox'],
                        "area": raw_ann['area'],
                        "iscrowd": 0,
                        "ignore": 0,
                        "segmentation": [] # segmentation 정보가 없으면 빈 리스트
                    }
                    coco_format['annotations'].append(ann_info)
                    annotation_id_counter += 1

        except Exception as e:
            print(f"에러 발생 파일: {json_file}")
            print(f"에러 내용: {e}")
            continue

    # 4. 결과 파일 저장하기
    print(f"💾 {OUTPUT_FILE} 에 저장하는 중...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, ensure_ascii=False, indent=None) # 용량 줄이려고 indent 제거
    
    print(" 성공! 모든 데이터를 하나로 합쳤습니다.")

if __name__ == "__main__":
    main()