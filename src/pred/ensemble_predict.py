"""
YOLOv11 앙상블 모델 기반 최종 제출(Submission) 생성 스크립트

[최적화 리포트 - Public Score: 0.98930]
1. 검출 전략 (Detection Strategy):
    - Model A (Original): CONF 0.7 적용. 고신뢰도 객체를 안정적으로 검출.
    - Model B (Enhanced): CONF 0.4 적용. CLAHE+Sharpening 전처리를 통해 저화질 이미지의 미탐(FN) 보완.
2. 박스 병합 전략 (WBF Fusion):
    - IoU Threshold 0.6: 알약이 밀집된 환경에서 박스가 하나로 뭉치는 현상을 방지하기 위해 임계값 상향.
    - Weights [1.05, 1.0]: 기본 모델(A)의 위치 정보를 소폭 우선시함.
    - Skip Box Threshold 0.05: 저신뢰도 후보군을 최대한 포함하여 WBF 연산의 기여도 상승 유도.
3. 최종 필터링:
    - 앙상블 이후 합산 점수 0.4(FINAL_CONF_TH) 이상만 기록하여 최종 재현율(Recall) 극대화.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# =========================
# (1) 경로 및 설정
# =========================
TEST_IMG_DIR = Path(r"E:\github\healtheat_vision\data\test_images")
CLASS_MAP_CSV = Path(r"E:\github\healtheat_vision\artifacts\class_map.csv")
MODEL_A_PATH = Path(r"E:\github\healtheat_vision\artifacts\models\yolo11s_full_train_v1_e150.pt")
MODEL_B_PATH = Path(r"E:\github\healtheat_vision\artifacts\models\yolo11s_ft_v1_enhanced.pt")

# 결과 저장 경로 설정
TIME_STR = datetime.now().strftime('%y%m%d_%H%M%S')
OUT_DIR = Path(r"E:\github\healtheat_vision\submissions") / f"ensemble_{TIME_STR}"
VIZ_DIR = OUT_DIR / "viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

SUBMISSION_CSV = OUT_DIR / "submission.csv"


# 앙상블 하이퍼파라미터
CONF_TH_A = 0.7      # 모델 A는 엄격하게 (고득점 위주)
CONF_TH_B = 0.4      # 모델 B는 너그럽게 (A가 놓친 것 수집)
FINAL_CONF_TH = 0.4  # WBF 이후 최종적으로 남길 점수
WBF_IOU_TH = 0.6
WBF_SKIP_TH = 0.05  # (모델 B의 후보를 살리기 위해)
WBF_WEIGHTS = [1.05, 1.0] # 모델 A의 위치 신뢰도를 5% 더 높게 평가

# =========================
# (2) 유틸리티 함수
# =========================

def apply_preprocessing(img_bgr):
    """모델 B를 위한 CLAHE + Sharpening"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def load_class_map():
    df = pd.read_csv(CLASS_MAP_CSV)
    id_to_orig = {int(r.yolo_id): int(r.orig_id) for r in df.itertuples(index=False)}
    id_to_name = {int(r.yolo_id): str(r.class_name) for r in df.itertuples(index=False)}
    return id_to_orig, id_to_name

def draw_hangeul_label(img, text, position, font_path="malgun.ttf", font_size=18, color=(0, 255, 0)):
    """OpenCV 이미지에 PIL을 이용해 한글 텍스트를 그리는 함수"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    # 텍스트 배경(검은색)을 살짝 넣어 가독성 확보
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =========================
# (3) 메인 추론 로직
# =========================

def main():
    yolo_to_orig, yolo_to_name = load_class_map()
    model_a = YOLO(str(MODEL_A_PATH))
    model_b = YOLO(str(MODEL_B_PATH))
    
    img_paths = sorted(list(TEST_IMG_DIR.glob("*.png")) + list(TEST_IMG_DIR.glob("*.jpg")))
    rows = []
    ann_id = 1

    for img_path in tqdm(img_paths, desc="앙상블 추론 중"):
        image_id = int(''.join(filter(str.isdigit, img_path.stem)))
        
        # 이미지 로드 (한글 경로 대응)
        img_array = np.fromfile(str(img_path), np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None: continue
        h, w = img_bgr.shape[:2]
        
        # 1. 모델별 추론
        res_a = model_a.predict(img_bgr, conf=CONF_TH_A, verbose=False)[0]
        img_processed = apply_preprocessing(img_bgr)
        res_b = model_b.predict(img_processed, conf=CONF_TH_B, verbose=False)[0]
        
        # 2. WBF 포맷 변환
        boxes_list, scores_list, labels_list = [], [], []
        for res in [res_a, res_b]:
            if len(res.boxes) > 0:
                boxes_list.append(res.boxes.xyxyn.cpu().numpy().tolist())
                scores_list.append(res.boxes.conf.cpu().numpy().tolist())
                labels_list.append(res.boxes.cls.cpu().numpy().astype(int).tolist())
            else:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])

        # 3. WBF 실행
        f_boxes, f_scores, f_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            weights=WBF_WEIGHTS,  # [1.2, 1.0] 적용
            iou_thr=WBF_IOU_TH, 
            skip_box_thr=WBF_SKIP_TH      # WBF 내부에서는 낮은 점수도 일단 합치도록 설정
        )

        # 4. 결과 저장 및 시각화
        viz_img = img_bgr.copy()
        for box, score, label in zip(f_boxes, f_scores, f_labels):

            # [추가된 부분] WBF 이후 최종 점수가 임계값(예: 0.5)보다 낮으면 버림
            if score < FINAL_CONF_TH:
                continue

            orig_id = yolo_to_orig.get(int(label))
            class_name = yolo_to_name.get(int(label), "Unknown")
            if orig_id is None: continue
            
            x1, y1, x2, y2 = int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)
            
            # Submission 데이터 추가
            rows.append({
                "annotation_id": ann_id,
                "image_id": image_id,
                "category_id": orig_id,
                "bbox_x": float(x1), "bbox_y": float(y1),
                "bbox_w": float(x2 - x1), "bbox_h": float(y2 - y1),
                "score": float(score)
            })
            ann_id += 1

            # 시각화 (박스 + 한글 라벨)
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{class_name} {score:.2f}"
            viz_img = draw_hangeul_label(viz_img, label_text, (x1, y1 - 25))

        # 이미지 저장 (한글 경로 대응)
        viz_path = VIZ_DIR / f"{img_path.stem}_ensemble.jpg"
        _, res_img = cv2.imencode(".jpg", viz_img)
        res_img.tofile(str(viz_path))

    # 최종 CSV 저장
    pd.DataFrame(rows).to_csv(SUBMISSION_CSV, index=False)
    print("\n" + "="*60)
    print(f"📊 {MODEL_NAME} 검증 성적표")
    print("-" * 60)
    print(report_df.to_string(index=False))
    print("="*60)
    print(f"\n✅ 앙상블 완료!")
    print(f"📊 제출 파일: {SUBMISSION_CSV}")
    print(f"🖼️ 시각화 폴더: {VIZ_DIR}")

if __name__ == "__main__":
    main()