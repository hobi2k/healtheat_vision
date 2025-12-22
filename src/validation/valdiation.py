"""
YOLO 모델 성능 비교 평가 스크립트

1. 평가 데이터셋 (Evaluation Dataset):
    - 출처: AIHub (TS_6번 데이터셋)
    - 특징: 모델 학습(Training)에 사용되지 않은 외부 독립 데이터셋으로, 
      모델의 일반화 성능(Generalization)을 객관적으로 검증하기 위해 사용됨.

2. 주요 평가 지표 (Evaluation Metrics):
    - Precision (정밀도): 모델이 객체라고 예측한 것 중 실제 정답(GT)인 비율. (오탐지/과검출 방지 능력)
    - Recall (재현율): 실제 정답 중 모델이 정확히 찾아낸 비율. (미탐지/누락 방지 능력)
    - F1-Score: Precision과 Recall의 조화 평균으로, 두 지표의 균형을 나타냄.
    - mAP50 (mean Average Precision): IoU 임계값 0.5 기준의 평균 정밀도. 검출 성능의 핵심 종합 지표.
    - TP (True Positive): 실제 정답을 맞게 검출한 수.
    - FP (False Positive): 정답이 아닌데 잘못 검출한 수 (오탐).
    - FN (False Negative): 실제 정답을 찾지 못하고 놓친 수 (미탐).

3. 전처리 스위치 (USE_PREPROCESSING):
    - True 설정 시: CLAHE(대비 최적화) 및 Sharpening(선명화) 필터 적용. 
      Fine-Tuning 모델의 학습 환경과 동일한 조건에서 평가하기 위해 사용됨.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime

# =========================
# (1) 설정 (모델에 따라 수정)
# =========================
MODEL_PATH = Path(r"E:\github\healtheat_vision\artifacts\models\yolo11n_full_train_v1.pt")
MODEL_NAME = "yolo11n_full_train_v1"  # 리포트에 표시될 이름
USE_PREPROCESSING = False       # FT 모델이면 True, 원본 모델이면 False

# 검증 경로
VAL_IMG_DIR = Path(r"E:\github\healtheat_vision\validation\yolo_val_ready\images")
VAL_LABEL_DIR = Path(r"E:\github\healtheat_vision\validation\yolo_val_ready\labels")
SAVE_DIR = Path(r"E:\github\healtheat_vision\validation")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 하이퍼파라미터
CONF_TH = 0.5
IOU_TH = 0.8

# =========================
# (2) 유틸리티 함수
# =========================
def apply_preprocessing(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# =========================
# (3) 메인 평가 로직
# =========================
def main():
    model = YOLO(str(MODEL_PATH))
    img_paths = list(VAL_IMG_DIR.glob("*.jpg")) + list(VAL_IMG_DIR.glob("*.png"))
    
    tp, fp, fn = 0, 0, 0
    total_gt = 0

    for img_path in tqdm(img_paths, desc=f"Evaluating {MODEL_NAME}"):
        # 1. 정답 로드
        label_path = VAL_LABEL_DIR / f"{img_path.stem}.txt"
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.split())
                    gt_boxes.append([cls, x-w/2, y-h/2, x+w/2, y+h/2]) # x1y1x2y2
        total_gt += len(gt_boxes)

        # 2. 추론
        img_array = np.fromfile(str(img_path), np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if USE_PREPROCESSING:
            img_bgr = apply_preprocessing(img_bgr)
            
        results = model.predict(img_bgr, conf=CONF_TH, iou=IOU_TH, verbose=False)[0]
        pred_boxes = results.boxes.xyxyn.cpu().numpy()
        pred_cls = results.boxes.cls.cpu().numpy()

        # 3. TP/FP 계산
        matched_gt = set()
        for p_box, p_cls in zip(pred_boxes, pred_cls):
            match_found = False
            for i, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
                if i in matched_gt: continue
                if p_cls == g_cls:
                    iou = calculate_iou(p_box, [g_x1, g_y1, g_x2, g_y2])
                    if iou >= 0.5:
                        tp += 1
                        matched_gt.add(i)
                        match_found = True
                        break
            if not match_found:
                fp += 1
        
        fn += (len(gt_boxes) - len(matched_gt))

    # 지표 산출
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # 간단하게 계산한 mAP (여기서는 IoU 0.5 기준의 Precision과 동일하게 취급되지만 분류 성능 포함)
    map50 = precision * recall 

    # 결과 리포트 생성
    report_df = pd.DataFrame([{
        "ModelName": MODEL_NAME,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "mAP50": round(map50, 4), # 실제 mAP는 PR 곡선 면적이지만 대용치로 사용
        "Total_GT": total_gt,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }])

    print("\n" + "="*60)
    print(f"📊 {MODEL_NAME} 검증 성적표")
    print("-" * 60)
    print(report_df.to_string(index=False))
    print("="*60)

    # 저장
    report_df.to_csv(SAVE_DIR / f"metrics_{MODEL_NAME}.csv", index=False)

if __name__ == "__main__":
    main()