import torch
import cv2
import os
import glob
import json
import pandas as pd
import re
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.ops import nms
from src.detection.model import get_model

ROOT_DIR = "./data"
TEST_IMAGE_DIR = "./data/test_images"
MODEL_PATH = "./saved_models/faster_rcnn_epoch_20.pth"
OUTPUT_FILE = "submission.csv"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SCORE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.4

def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

def get_all_category_ids(root_dir):
    json_files = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
    ids = set()
    for path in tqdm(json_files):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'categories' in data:
                for c in data['categories']: ids.add(int(c['id']))
            if 'annotations' in data:
                for a in data['annotations']: ids.add(int(a['category_id']))
        except: pass
    return sorted(list(ids))

def main():
    if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)

    # 문자열 정렬 (1 -> 10 -> 100)
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    sorted_real_ids = get_all_category_ids(ROOT_DIR)
    category_map = {i+1: real_id for i, real_id in enumerate(sorted_real_ids)}
    
    num_classes = 75
    try:
        ckpt = torch.load(MODEL_PATH, map_location='cpu')
        for k, v in ckpt.items():
            if 'cls_score.weight' in k: 
                num_classes = v.shape[0]
                break
    except: pass

    model = get_model(num_classes)
    try: model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except: model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.to(DEVICE).eval()

    results = []
    transform = T.Compose([T.ToTensor()])

    with torch.no_grad():
        for fname in tqdm(image_files):
            img_path = os.path.join(TEST_IMAGE_DIR, fname)
            img_id = extract_number(fname)
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            img_t = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(DEVICE)
            pred = model([img_t])[0]

            boxes, scores, labels = pred['boxes'], pred['scores'], pred['labels']
            
            mask = scores > SCORE_THRESHOLD
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
            
            keep = nms(boxes, scores, IOU_THRESHOLD)
            f_boxes, f_scores, f_labels = boxes[keep].cpu().numpy(), scores[keep].cpu().numpy(), labels[keep].cpu().numpy()

            for i in range(len(f_boxes)):
                x1, y1, x2, y2 = f_boxes[i]
                real_cat = category_map.get(int(f_labels[i]), int(f_labels[i]))
                
                results.append({
                    "image_id": img_id,
                    "category_id": real_cat,
                    "bbox_x": round(float(x1), 2),
                    "bbox_y": round(float(y1), 2),
                    "bbox_w": round(float(x2 - x1), 2),
                    "bbox_h": round(float(y2 - y1), 2),
                    "score": round(float(f_scores[i]), 6)
                })

    df = pd.DataFrame(results)
    if not df.empty:
        df.insert(0, 'annotation_id', range(1, len(df)+1))
        df.to_csv(OUTPUT_FILE, index=False)
        print("Done.")

if __name__ == "__main__":
    main()