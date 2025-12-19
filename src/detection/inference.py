import torch
import cv2
import os
import random
import json
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from src.detection.model import get_model

# ==========================================
# 설정
# ==========================================
MODEL_PATH = "./saved_models/faster_rcnn_final.pth"
IMAGE_DIR = "./data/train_images" 
ANNOTATION_FILE = "./data/train_annotations.json"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" 

# 기준 점수: 0.1 (10% 이상이면 표시)
SCORE_THRESHOLD = 0.1
# 중복 제거 기준
IOU_THRESHOLD = 0.4

def get_class_names():
    with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    class_map = {cat['id']: cat['name'] for cat in data['categories']}
    return class_map, len(class_map) + 1

def draw_text_korean(img, text, x, y, color=(0, 255, 0), font_size=20):
    # OpenCV 이미지(numpy)를 PIL 이미지로 변환하여 한글 출력
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    draw.text((x, y), text, font=font, fill=color)
    # 다시 OpenCV 이미지로 변환
    return np.array(img_pil)

def main():
    print(f"무작위 사진 테스트 시작 (Threshold: {SCORE_THRESHOLD})")

    class_map, num_classes = get_class_names()
    model = get_model(num_classes)
    
    if not os.path.exists(MODEL_PATH):
        print("저장된 모델 파일이 없습니다.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    if not image_files:
        # PNG 파일이 없을 경우 JPG 파일도 확인
        image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
        if not image_files:
            print("이미지 파일이 없습니다.")
            return
        
    # 무작위 이미지 선택
    file_name = random.choice(image_files)
    img_path = os.path.join(IMAGE_DIR, file_name)
    print(f"테스트 이미지: {file_name}")

    img_origin = cv2.imread(img_path)
    # PIL 처리를 위해 BGR -> RGB 변환
    img_rgb = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img_rgb).to(DEVICE)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']

    # 상위 점수 출력 (확인용)
    print("\n[상위 예측 점수 10개]")
    print(scores[:10].cpu().numpy())

    # 점수 필터링
    mask = scores > SCORE_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    # NMS (중복 제거)
    keep_indices = nms(boxes, scores, IOU_THRESHOLD)
    
    final_boxes = boxes[keep_indices].cpu().numpy()
    final_scores = scores[keep_indices].cpu().numpy()
    final_labels = labels[keep_indices].cpu().numpy()

    print(f"\n최종 검출된 알약: {len(final_boxes)}개")

    # 결과 그리기
    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = box.astype(int)
        score = final_scores[i]
        label_id = final_labels[i]
        label_name = class_map.get(label_id, f"Class {label_id}")

        # 박스 (BGR 기준 빨강)
        cv2.rectangle(img_origin, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 텍스트 (한글 출력 함수 사용)
        text = f"{label_name} ({score*100:.1f}%)"
        img_origin = draw_text_korean(img_origin, text, x1, y1 - 25, color=(0, 0, 255), font_size=16)

    save_path = "test_result_final.png"
    cv2.imwrite(save_path, img_origin)
    print(f"이미지 저장 완료: {save_path}")

if __name__ == "__main__":
    main()