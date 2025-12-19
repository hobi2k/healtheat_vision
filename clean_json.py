import json
import os

# ==========================================
# 설정: 경로 확인
# ==========================================
JSON_FILE = "./data/train_annotations.json"
IMAGE_DIR = "./data/train_images"

def main():
    print("🧹 데이터 대청소를 시작합니다... (없는 파일 지우기)")
    
    # 1. 장부(JSON) 열기
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_count = len(data['images'])
    valid_images = []
    valid_ids = set() # 살아남은 이미지들의 ID 모음

    # 2. 실제로 파일이 있는지 하나씩 확인
    print("🔍 파일 검사 중...")
    for img in data['images']:
        file_path = os.path.join(IMAGE_DIR, img['file_name'])
        
        # 파일이 실제로 존재하면 -> 리스트에 추가 (살려줌)
        if os.path.exists(file_path):
            valid_images.append(img)
            valid_ids.add(img['id'])
        # 파일이 없으면 -> 그냥 무시 (삭제됨)
    
    # 3. 살아남은 이미지에 해당하는 라벨(박스)만 남기기
    valid_annotations = []
    for ann in data['annotations']:
        if ann['image_id'] in valid_ids:
            valid_annotations.append(ann)

    # 4. 장부 업데이트
    data['images'] = valid_images
    data['annotations'] = valid_annotations

    # 5. 덮어쓰기 (저장)
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"\n✨ 청소 완료!")
    print(f"📝 원래 장부에 있던 개수: {original_count}개")
    print(f"✅ 실제 파일이 있어서 살아남은 개수: {len(valid_images)}개")
    print(f"🗑️ 삭제된 유령 데이터: {original_count - len(valid_images)}개")
    print("이제 다시 check_loader.py를 실행하면 될 겁니다!")

if __name__ == "__main__":
    main()