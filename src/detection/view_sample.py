import torch
import cv2
import os
import json
import random
import glob
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from src.detection.model import get_model

# ================= 설정 =================
MODEL_PATH = "./saved_models/faster_rcnn_epoch_20.pth"
TEST_DIR = "./data/test_images"
ANNOTATION_DIR = "./data/train_annotations"
SAVE_PATH = "./sample_result.jpg"
SCORE_THRESHOLD = 0.35
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 설치한 한글 폰트 경로
# ========================================

def get_class_map(root_dir):
    print("[Info] Scanning annotations to map IDs to names...")
    
    all_ann_files = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
    
    id_to_name = {}
    all_ids = set()

    # ID와 이름 수집
    for ann_path in tqdm(all_ann_files, desc="Loading annotations"):
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'categories' in data:
                for cat in data['categories']:
                    cat_id = int(cat['id'])
                    cat_name = cat['name']
                    id_to_name[cat_id] = cat_name
                    all_ids.add(cat_id)
            elif 'annotations' in data:
                 for anno in data['annotations']:
                     all_ids.add(int(anno['category_id']))
        except:
            continue
            
    sorted_ids = sorted(list(all_ids))
    idx_to_name = ["Background"]
    
    for real_id in sorted_ids:
        name = id_to_name.get(real_id, f"Unknown_{real_id}")
        idx_to_name.append(name)
        
    print(f"[Info] Class map ready. Total classes: {len(idx_to_name)-1}")
    return idx_to_name

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    class_names = get_class_map(ANNOTATION_DIR)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        num_classes = checkpoint['roi_heads.box_predictor.cls_score.weight'].shape[0]
        model = get_model(num_classes)
        model.load_state_dict(checkpoint)
        model.to(device).eval()
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    image_files = glob.glob(os.path.join(TEST_DIR, "*.png")) + glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    if not image_files:
        print("[Error] No test images found.")
        return
    
    img_path = random.choice(image_files)

    transform = T.Compose([T.ToTensor()])
    original_img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()

    # OpenCV 이미지를 PIL 이미지로 변환 (한글 출력을 위해)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    # 폰트 로드
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except:
        print("[Warning] Nanum font not found. Using default.")
        font = ImageFont.load_default()

    found_count = 0
    for i, box in enumerate(boxes):
        score = scores[i]
        if score < SCORE_THRESHOLD:
            continue
        
        found_count += 1
        x1, y1, x2, y2 = map(int, box)
        label_idx = int(labels[i])
        
        if 0 <= label_idx < len(class_names):
            pill_name = class_names[label_idx]
        else:
            pill_name = f"ID:{label_idx}"

        # 박스 그리기 (Pillow 사용)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        
        # 텍스트 그리기 (한글 지원)
        text = f"{pill_name} ({score:.2f})"
        
        # 텍스트 배경 박스 (가독성 위해)
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([x1, y1 - text_height - 10, x1 + text_width + 10, y1], fill=(0, 255, 0))
        draw.text((x1 + 5, y1 - text_height - 5), text, font=font, fill=(0, 0, 0))

    # 다시 OpenCV 포맷으로 변환하여 저장
    result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(SAVE_PATH, result_img)
    
    print(f"[Result] Found {found_count} pills. Saved to {SAVE_PATH}")
    os.system(f"code {SAVE_PATH}")

if __name__ == "__main__":
    main()