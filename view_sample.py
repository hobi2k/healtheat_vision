import json
import os
import cv2
import random
import matplotlib.pyplot as plt

# ==========================================
# 설정: 경로 확인
# ==========================================
JSON_FILE = "./data/train_annotations.json"
IMAGE_DIR = "./data/train_images"
OUTPUT_IMG = "sample_result.png"

def main():
    # 1. 실제 존재하는 이미지 파일 목록 가져오기 (이게 바뀐 부분!)
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ {IMAGE_DIR} 폴더가 없습니다.")
        return

    # png 파일만 골라내기
    real_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    
    if not real_files:
        print("❌ 폴더 안에 .png 이미지가 하나도 없습니다!")
        return

    print(f"📂 폴더 안에 있는 이미지 개수: {len(real_files)}개")

    # 2. 그 중에서 랜덤으로 하나 뽑기
    selected_filename = random.choice(real_files)
    print(f"📸 선택된 파일(실제 존재함): {selected_filename}")

    # 3. JSON 파일 읽어서 정보 찾기
    print(f"📖 JSON 장부에서 정보 찾는 중...")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 뽑은 파일 이름이랑 똑같은 정보 찾기
    target_img_id = None
    for img in data['images']:
        if img['file_name'] == selected_filename:
            target_img_id = img['id']
            break
    
    if target_img_id is None:
        print("⚠️ 주의: 이미지는 있는데 JSON 파일에 정보가 없어요! (데이터셋 짝이 안 맞음)")
        # 그래도 이미지는 띄워봅시다
        img_path = os.path.join(IMAGE_DIR, selected_filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        plt.title("No Annotation Found")
        plt.axis('off')
        plt.savefig(OUTPUT_IMG)
        print(f"⚠️ 이미지만 {OUTPUT_IMG}로 저장했습니다.")
        return

    # 4. 박스 정보 찾아서 그리기
    img_path = os.path.join(IMAGE_DIR, selected_filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    found_box = False
    for ann in data['annotations']:
        if ann['image_id'] == target_img_id:
            found_box = True
            x, y, w, h = map(int, ann['bbox'])
            # 박스 그리기
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
            
            # 이름 찾기
            cat_id = ann['category_id']
            cat_name = next((c['name'] for c in data['categories'] if c['id'] == cat_id), "Unknown")
            
            # 글씨 쓰기
            cv2.putText(img, cat_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    if not found_box:
        print("⚠️ 이 이미지에 해당하는 박스 정보가 없습니다.")

    # 5. 저장
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(OUTPUT_IMG)
    print(f"✅ 성공! {OUTPUT_IMG} 파일이 생성되었습니다. 이미지를 확인해보세요!")

if __name__ == "__main__":
    main()