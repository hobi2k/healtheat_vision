"""
YOLOv11 앙상블 추론 스크립트 (WBF 방식)

1. 모델 A (원본): 원본 이미지로 추론
2. 모델 B (FT): 전처리(CLAHE+Sharpening) 이미지로 추론
3. 두 결과를 WBF(Weighted Boxes Fusion)로 병합하여 최종 submission 생성
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

# =========================
# (1) 경로 및 설정 (하드코딩 유지)
# =========================
TEST_IMG_DIR = Path(r"E:\github\healtheat_vision\data\test_images")
CLASS_MAP_CSV = Path(r"E:\github\healtheat_vision\artifacts\class_map.csv")

# 모델 경로
MODEL_A_PATH = Path(r"E:\github\healtheat_vision\artifacts\models\yolo11s_full_train_v1_e150.pt")
MODEL_B_PATH = Path(r"E:\github\healtheat_vision\artifacts\runs\yolo11s_ft_v1_enhanced\weights\best.pt") # 결과 좋게 나오면 models로 옮기고 이름 수정

# 결과 저장
OUT_DIR = Path(r"E:\github\healtheat_vision\submissions") / f"ensemble_{datetime.now().strftime('%y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_CSV = OUT_DIR / "submission_ensemble.csv"

# 추론 하이퍼파라미터
CONF_TH = 0.5  # 앙상블 시에는 조금 낮게 잡고 WBF로 걸러내는 것이 유리
IOU_TH = 0.6
WBF_IOU_TH = 0.55 # 박스 통합 기준
WBF_SKIP_TH = 0.05 # 무시할 낮은 점수

# =========================
# (2) 전처리 및 유틸 함수
# =========================
def apply_preprocessing(img_bgr):
    """모델 B를 위한 CLAHE + Sharpening 전처리"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # YOLO 입력용 RGB

def load_class_map():
    df = pd.read_csv(CLASS_MAP_CSV)
    return {int(r.yolo_id): int(r.orig_id) for r in df.itertuples(index=False)}

def main():
    yolo_to_orig = load_class_map()
    model_a = YOLO(str(MODEL_A_PATH))
    model_b = YOLO(str(MODEL_B_PATH))
    
    img_paths = sorted(list(TEST_IMG_DIR.glob("*.png")) + list(TEST_IMG_DIR.glob("*.jpg")))
    rows = []
    ann_id = 1

    for img_path in tqdm(img_paths, desc="Ensemble Predicting"):
        image_id = int(''.join(filter(str.isdigit, img_path.stem)))
        
        # 1. 이미지 읽기
        img_array = np.fromfile(str(img_path), np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        h, w = img_bgr.shape[:2]
        
        # 2. 모델 A 추론 (원본 이미지)
        res_a = model_a.predict(img_bgr, conf=CONF_TH, iou=IOU_TH, augment=True, verbose=False)[0]
        
        # 3. 모델 B 추론 (전처리 이미지)
        img_processed = apply_preprocessing(img_bgr)
        res_b = model_b.predict(img_processed, conf=CONF_TH, iou=IOU_TH, augment=True, verbose=False)[0]
        
        # WBF를 위한 데이터 포맷 변환 ([0,1] 정규화 필요)
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for res in [res_a, res_b]:
            if len(res.boxes) > 0:
                # 박스 좌표 [x1, y1, x2, y2] 정규화
                boxes = res.boxes.xyxyn.cpu().numpy().tolist()
                scores = res.boxes.conf.cpu().numpy().tolist()
                labels = res.boxes.cls.cpu().numpy().astype(int).tolist()
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
            else:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])

        # 4. WBF 실행 (박스 병합)
        # weights=[1, 1]은 두 모델의 중요도를 동일하게 봄
        f_boxes, f_scores, f_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            weights=[1, 1], iou_thr=WBF_IOU_TH, skip_box_thr=WBF_SKIP_TH
        )

        # 5. 결과 저장 (정규화 좌표 -> 절대 좌표 변환)
        for box, score, label in zip(f_boxes, f_scores, f_labels):
            orig_id = yolo_to_orig.get(int(label))
            if orig_id is None: continue
            
            x1, y1, x2, y2 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
            rows.append({
                "annotation_id": ann_id,
                "image_id": image_id,
                "category_id": orig_id,
                "bbox_x": float(x1), "bbox_y": float(y1),
                "bbox_w": float(x2 - x1), "bbox_h": float(y2 - y1),
                "score": float(score)
            })
            ann_id += 1

    pd.DataFrame(rows).to_csv(SUBMISSION_CSV, index=False)
    print(f"\n✅ Ensemble Submission Saved: {SUBMISSION_CSV}")

if __name__ == "__main__":
    main()