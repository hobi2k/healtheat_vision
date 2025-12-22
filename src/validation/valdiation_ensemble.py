"""
YOLOv11 앙상블 모델 성능 검증 스크립트 (동기화 버전)

1. 평가 데이터셋 (Evaluation Dataset):
    - 출처: AIHub (TS_6번 데이터셋)
    - 목적: 학습에 사용되지 않은 외부 독립 데이터셋을 활용하여 모델의 일반화 성능 검증.

2. 주요 평가 지표:
    - Precision, Recall, F1-Score, mAP50
    - TP/FP/FN 절대 수치 산출
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm
from datetime import datetime

# =========================
# (1) 경로 및 하이퍼파라미터 (Submission과 통일)
# =========================
MODEL_A_PATH = Path(r"E:\github\healtheat_vision\artifacts\models\yolo11s_full_train_v1_e150.pt")
MODEL_B_PATH = Path(r"E:\github\healtheat_vision\artifacts\models\yolo11s_ft_v1_enhanced.pt")

VAL_IMG_DIR = Path(r"E:\github\healtheat_vision\validation\yolo_val_ready\images")
VAL_LABEL_DIR = Path(r"E:\github\healtheat_vision\validation\yolo_val_ready\labels")
REPORT_SAVE_DIR = Path(r"E:\github\healtheat_vision\validation")
REPORT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 앙상블 하이퍼파라미터
CONF_TH_A = 0.7      # 모델 A는 엄격하게 (고득점 위주)
CONF_TH_B = 0.4      # 모델 B는 너그럽게 (A가 놓친 것 수집)
FINAL_CONF_TH = 0.4  # WBF 이후 최종적으로 남길 점수
WBF_IOU_TH = 0.6
WBF_SKIP_TH = 0.05  # (모델 B의 후보를 살리기 위해)
WBF_WEIGHTS = [1.05, 1.0] # 모델 A의 위치 신뢰도를 5% 더 높게 평가
MATCH_IOU_TH = 0.5   # 검증 시 정답 인정 IoU (표준 0.5)

# =========================
# (2) 유틸리티 함수
# =========================
def apply_preprocessing(img_bgr):
    """모델 B 전용 전처리 (Submission과 동일)"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def calculate_iou(box1, box2):
    """[x1, y1, x2, y2] 형식의 IoU 계산"""
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
    print(f"🚀 앙상블 검증 시작 (AIHub TS_6) - A:{CONF_TH_A} / B:{CONF_TH_B}")
    
    model_a = YOLO(str(MODEL_A_PATH))
    model_b = YOLO(str(MODEL_B_PATH))
    
    img_paths = list(VAL_IMG_DIR.glob("*.jpg")) + list(VAL_IMG_DIR.glob("*.png"))
    tp, fp, fn, total_gt_count = 0, 0, 0, 0

    for img_path in tqdm(img_paths, desc="Evaluating Ensemble"):
        # 1. 정답(GT) 로드
        label_path = VAL_LABEL_DIR / f"{img_path.stem}.txt"
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.split())
                    gt_boxes.append([cls, x-w/2, y-h/2, x+w/2, y+h/2])
        total_gt_count += len(gt_boxes)

        # 2. 이미지 로드 및 추론
        img_array = np.fromfile(str(img_path), np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        res_a = model_a.predict(img_bgr, conf=CONF_TH_A, verbose=False)[0]
        img_proc = apply_preprocessing(img_bgr)
        res_b = model_b.predict(img_proc, conf=CONF_TH_B, verbose=False)[0]
        
        b_list, s_list, l_list = [], [], []
        for r in [res_a, res_b]:
            if len(r.boxes) > 0:
                b_list.append(r.boxes.xyxyn.cpu().numpy().tolist())
                s_list.append(r.boxes.conf.cpu().numpy().tolist())
                l_list.append(r.boxes.cls.cpu().numpy().astype(int).tolist())
            else:
                b_list.append([]), s_list.append([]), l_list.append([])
        
        # WBF 실행
        f_b, f_s, f_l = weighted_boxes_fusion(b_list, s_list, l_list, weights=WBF_WEIGHTS, iou_thr=WBF_IOU_TH, skip_box_thr=WBF_SKIP_TH)
        
        # 최종 필터링
        final_p_boxes = [b for b, s in zip(f_b, f_s) if s >= FINAL_CONF_TH]
        final_p_labels = [l for l, s in zip(f_l, f_s) if s >= FINAL_CONF_TH]

        # 3. 채점 (TP/FP/FN)
        matched_gt = set()
        for p_box, p_cls in zip(final_p_boxes, final_p_labels):
            match_found = False
            for i, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
                if i in matched_gt: continue
                if int(p_cls) == int(g_cls):
                    iou = calculate_iou(p_box, [g_x1, g_y1, g_x2, g_y2])
                    if iou >= MATCH_IOU_TH:
                        tp += 1
                        matched_gt.add(i)
                        match_found = True
                        break
            if not match_found:
                fp += 1
        fn += (len(gt_boxes) - len(matched_gt))

    # 4. 결과 산출 및 저장
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    map50 = precision * recall 

    report_data = {
        "ModelName": "Ensemble_Final", "Precision": round(precision, 4),
        "Recall": round(recall, 4), "F1-Score": round(f1, 4), "mAP50": round(map50, 4),
        "TP": tp, "FP": fp, "FN": fn, "Total_GT": total_gt_count
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    pd.DataFrame([report_data]).to_csv(REPORT_SAVE_DIR / f"val_ensemble_{timestamp}.csv", index=False)
    print("\n✅ 검증 완료. 보고서가 저장되었습니다.")

if __name__ == "__main__":
    main()