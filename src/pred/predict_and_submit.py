# src/pred/predict_and_submit.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re

import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# =========================
# (1) 경로 설정: 여기만 주로 바꾸면 됨
# =========================
REPO_ROOT = Path(__file__).resolve().parents[2]

# ✅ 테스트 이미지 폴더 (대회/과제 구조에 맞게 수정)
# 예: data/test_images, data/test, data/images/test 등
TEST_IMG_DIR = REPO_ROOT / "data" / "test_images"

# ✅ 학습 run 폴더명 (artifacts/runs 아래)
RUN_NAME = "yolo11s_full_train_v1_e150_"
RUN_DIR = REPO_ROOT / "artifacts" / "runs" / RUN_NAME

# ✅ 보통 제출은 best.pt 권장 (last.pt는 마지막 상태라 불안정할 수 있음)
MODEL_PT = RUN_DIR / "weights" / "best.pt"
# MODEL_PT = RUN_DIR / "weights" / "last.pt"  # 필요 시 변경

# ✅ class_map.csv (orig_id <-> yolo_id)
CLASS_MAP_CSV = REPO_ROOT / "artifacts" / "class_map.csv"

# ✅ 출력 폴더
OUT_DIR = REPO_ROOT / "submissions" / f"{RUN_NAME}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
VIS_DIR = OUT_DIR / "viz"
SUBMISSION_CSV = OUT_DIR / "submission.csv"


# =========================
# (2) 제출 포맷/ID 변환: 여기만 바꾸면 규격 대응 가능
# =========================
def transform_image_id(img_path: Path) -> int:
    """
    submission의 image_id는 int여야 함(기존 submission_251210.csv 기준).
    기본: 파일명(확장자 제외)에서 숫자만 뽑아 int로 변환.
    예)
      - "000123.png" -> 123
      - "123.jpg" -> 123
      - "test_123.png" -> 123  (가장 마지막 숫자 덩어리 사용)
    규칙이 다르면 여기만 수정.
    """
    stem = img_path.stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"Cannot parse numeric image_id from filename: {img_path.name}")
    return int(nums[-1])


def load_yolo_to_orig_class_map(class_map_csv: Path) -> dict[int, int]:
    """
    class_map.csv 컬럼: orig_id, yolo_id (기존 submission_251210.csv 기준)
    예측 cls는 yolo_id이므로 -> orig_id(category_id)로 역변환 필요
    """
    df = pd.read_csv(class_map_csv)
    # yolo_id -> orig_id
    return {int(r.yolo_id): int(r.orig_id) for r in df.itertuples(index=False)}


# =========================
# (3) 시각화 유틸
# =========================
def draw_boxes(image: Image.Image, boxes, labels, scores) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # 폰트는 환경마다 다르니 애플 기본 폰트 fallback
    try:
        font = ImageFont.truetype("AppleGothic.ttf", 18)
    except:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        draw.rectangle([x1, y1, x2, y2], width=3)
        draw.text((x1, max(0, y1 - 18)), f"{lab} {sc:.2f}", font=font)
    return img


# =========================
# (4) 메인
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_IMG_DIR.exists():
        raise FileNotFoundError(f"TEST_IMG_DIR not found: {TEST_IMG_DIR}")

    if not MODEL_PT.exists():
        raise FileNotFoundError(f"MODEL_PT not found: {MODEL_PT}")

    yolo_to_orig = load_yolo_to_orig_class_map(CLASS_MAP_CSV)

    # ✅ 모델 로드 (학습된 가중치 사용)
    model = YOLO(str(MODEL_PT))

    # ✅ 테스트 이미지 수집
    img_paths = sorted(
        list(TEST_IMG_DIR.glob("*.png")) +
        list(TEST_IMG_DIR.glob("*.jpg")) +
        list(TEST_IMG_DIR.glob("*.jpeg"))
    )
    if not img_paths:
        raise FileNotFoundError(f"No test images found under: {TEST_IMG_DIR}")

    rows = []
    ann_id = 1

    # ✅ 추론 파라미터(필요 시 수정)
    CONF_TH = 0.6
    IOU_TH = 0.7

    for img_path in img_paths:
        image_id = transform_image_id(img_path)

        # ultralytics predict
        results = model.predict(
            source=str(img_path),
            conf=CONF_TH,
            iou=IOU_TH,
            verbose=False,
        )
        r = results[0]

        # boxes: xyxy, cls, conf
        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()   # (N,4)
        cls  = r.boxes.cls.cpu().numpy()    # (N,)
        conf = r.boxes.conf.cpu().numpy()   # (N,)

        # 시각화 저장용
        vis_boxes = []
        vis_labels = []
        vis_scores = []

        for (x1, y1, x2, y2), yolo_id, score in zip(xyxy, cls, conf):
            yolo_id = int(yolo_id)
            if yolo_id not in yolo_to_orig:
                # class_map에 없는 클래스면 스킵
                continue

            category_id = yolo_to_orig[yolo_id]

            # 제출 bbox는 (x, y, w, h) top-left 기준 (submission 샘플 기준)
            bbox_x = float(x1)
            bbox_y = float(y1)
            bbox_w = float(x2 - x1)
            bbox_h = float(y2 - y1)

            rows.append({
                "annotation_id": ann_id,
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "score": float(score),
            })
            ann_id += 1

            vis_boxes.append((x1, y1, x2, y2))
            vis_labels.append(str(category_id))
            vis_scores.append(float(score))

        # ✅ 시각화: 박스가 1개라도 있으면 저장
        if vis_boxes:
            img = Image.open(img_path)
            vis = draw_boxes(img, vis_boxes, vis_labels, vis_scores)
            vis.save(VIS_DIR / f"{img_path.stem}_pred.jpg", quality=95)

    # 저장
    sub_df = pd.DataFrame(rows, columns=[
        "annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])

    # dtype 맞추기(샘플 제출파일이 int64/float64였음)
    if not sub_df.empty:
        sub_df["annotation_id"] = sub_df["annotation_id"].astype("int64")
        sub_df["image_id"] = sub_df["image_id"].astype("int64")
        sub_df["category_id"] = sub_df["category_id"].astype("int64")

    sub_df.to_csv(SUBMISSION_CSV, index=False)
    print(f"[OK] saved submission: {SUBMISSION_CSV}")
    print(f"[OK] saved viz dir   : {VIS_DIR}")


if __name__ == "__main__":
    main()