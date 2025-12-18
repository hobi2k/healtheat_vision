"""
Auto-Correct v1

- 원본 GT(annotation)가 여러 JSON에 분산되어 있으므로 이미지 기준으로 하나로 통합
- GT의 품질이 균일하지 않음 -> YOLO detection 결과와 cross-check하여 보정
- GT를 우선적으로 존중하되 틀린 경우는 YOLO로 교체하고, 누락 항목은 YOLO로 채움(pseudo-label)
- 최종 JSON은 COCO 포맷
- 파이프라인에 필요한 매핑(category_mapping.json)을 활용해 class name 일관성 유지.

핵심 기능:
1. 기존 GT(JSON 여러 개)를 이미지 기준으로 하나로 merge
2. GT vs YOLO detection 비교
   - GT가 맞으면 유지
   - GT가 명백히 틀렸으면 YOLO 기준으로 교정
   - 누락 객체는 YOLO로 자동 추가
3. 라벨 없는 이미지 -> YOLO 기반 pseudo-label 생성
4. category_mapping.json 으로 클래스명 일관성 유지
5. 최종 구조는 unified_dataset/annotations/*.json
"""

import json
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO

from src.config import Config, init_logger

logger = init_logger()

# 경로 및 리소스 로드 (파이프라인의 입력 부분)
IMG_DIR = Config.DATA_DIR / "clean_data" / "images"
ANN_DIR = Config.DATA_DIR / "clean_data" / "annotations"
OUT_DIR = Config.DATA_DIR / "clean_dataset" / "annotations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# YOLO 모델 로드
YOLO_MODEL = YOLO(str(Config.BASE_DIR / "outputs/models/hobi_yolo11m_pill4/weights/best.pt"))

# category_mapping.json: YOLO class id -> COCO id + name 딕셔너리로 가져오기
CATEGORY_MAP_PATH = ANN_DIR / "category_mapping.json"
with CATEGORY_MAP_PATH.open("r", encoding="utf-8") as f:
    raw_map = json.load(f)
    YOLO2COCO = {
        int(k): {
            "coco_id": int(v["coco_id"]),
            "name": v["name"],
        }
        for k, v in raw_map.items()
    }

logger.info(f"[INFO] Loaded category_mapping.json → {len(YOLO2COCO)} classes")


# IoU 계산 함수
# IoU는 '얼마나 많이 겹치는가'를 0~1로 정량화한다.
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# 기존 GT(JSON) -> 이미지 기준으로 통합
# '이미지 기준 단일 구조체'로 합치기
def merge_original_annotations(ann_root: Path):
    """
    반환 구조:
    merged[img_name] = {
        "annotations": [...], # 원본 GT annotation list
        "categories": {}, # category_id -> category_name
        "meta": {...},   # 원본 json 저장
    }
    """
    merged = defaultdict(lambda: {"annotations": [], "categories": {}, "meta": {}})

    json_files = list(ann_root.rglob("*.json"))
    for jpath in json_files:
        # category_mapping.json은 어노테이션이 아니라 매핑 정보이므로 제외
        if jpath.name == "category_mapping.json":
            continue
        try:
            with jpath.open("r", encoding="utf-8") as f:
                coco = json.load(f)
        except Exception as e:
            logger.warning(f"[WARN] JSON 로드 실패: {jpath} ({e})")
            continue

        imgs = coco.get("images", [])
        if not imgs:
            continue

        img_info = imgs[0]
        fname = img_info.get("file_name")
        if fname is None:
            continue

        img_id = img_info["id"]

        # annotation 모으기
        for ann in coco.get("annotations", []):
            if ann.get("image_id") == img_id and "bbox" in ann:
                merged[fname]["annotations"].append(ann)

        # category 이름 매핑
        for c in coco.get("categories", []):
            cid = int(c["id"])
            cname = c.get("name", f"class_{cid}")
            merged[fname]["categories"][cid] = cname

        # meta에 원본 json 전체 저장
        merged[fname]["meta"][jpath.name] = coco

    return merged


# 3. YOLO 모델로 박스 검출 -? COCO category_id/name 붙여 반환
# object detection 모델 output은 "class index" 기반
# COCO는 "category_id" 기반
# class id를 category id로 변환
def detect_boxes(img_path: Path, conf_th: float = 0.1):
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"[WARN] 이미지 로드 실패: {img_path}")
        return []

    results = YOLO_MODEL(img, conf=conf_th)[0]

    dets = []
    for (x1, y1, x2, y2), cls_id in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.cls.cpu().numpy(),
    ):
        cls_id = int(cls_id)
        mapped = YOLO2COCO.get(cls_id)
        if mapped is None:
            # mapping.json에 없는 class는 데이터셋 정의에서 벗어난 것이다
            continue

        dets.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "category_id": mapped["coco_id"],
                "name": mapped["name"],
            }
        )
    return dets


# 4. GT vs YOLO 자동 교정
def auto_correct(gt, det, match_th: float = 0.4, replace_th: float = 0.1):
    """
    규칙 설명:

    1. IoU >= match_th -> GT 유지  
       "둘이 충분히 겹치면 GT를 신뢰한다"

    2. IoU < replace_th -> GT가 완전히 틀렸다고 판단  
       "YOLO 박스를 GT 대체로 사용"

    3. 그 외는 GT 선택

    4) YOLO에만 있고 GT에 없는 박스는 '누락 객체'로 보고 추가
    """
    used = set()
    corrected = []

    # GT 하나당 YOLO 중 가장 IoU 높은 후보 찾기
    for g in gt:
        g_box = g["bbox"]
        best_iou = 0.0
        best_det = None
        best_idx = -1

        for j, d in enumerate(det):
            if j in used:
                continue
            iou_val = iou_xyxy(g_box, d["bbox"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_det = d
                best_idx = j

        if best_iou >= match_th:
            corrected.append(g)
            used.add(best_idx)
        elif best_iou < replace_th and best_det is not None:
            corrected.append(best_det)
            used.add(best_idx)
        else:
            corrected.append(g)

    # YOLO에만 있는 객체들 추가
    for j, d in enumerate(det):
        if j not in used:
            corrected.append(d)

    return corrected


# 중복 박스 제거
def remove_duplicate(boxes, iou_th=0.5):
    out = []
    removed = set()

    for i in range(len(boxes)):
        if i in removed:
            continue
        bi = boxes[i]

        for j in range(i + 1, len(boxes)):
            if j in removed:
                continue
            bj = boxes[j]

            if iou_xyxy(bi["bbox"], bj["bbox"]) >= iou_th:
                # GT를 우선해서 반영
                if "name" not in bi or bi["name"] == "":
                    removed.add(i)
                else:
                    removed.add(j)

        if i not in removed:
            out.append(bi)

    return out


# 최종 COCO JSON 생성
def build_unified_json(img_path: Path, merged_info, det_boxes):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"이미지 로드 실패: {img_path}")
    h, w = img.shape[:2]

    img_name = img_path.name
    img_id = 1  # 한 JSON 파일 안에서 unique면 충분

    # GT bbox를 COCO xywh -> xyxy 로 변환
    gt = []
    for ann in merged_info.get("annotations", []):
        bbox = ann.get("bbox", [])

        # 필수 방어: bbox 비어 있거나 길이 이상하면 skip
        if not bbox or len(bbox) != 4:
            logger.warning(f"[WARN] invalid GT bbox skipped: {bbox}")
            continue

        x, y, bw, bh = bbox
        gt.append(
            {
                "bbox": [x, y, x + bw, y + bh],
                "category_id": int(ann["category_id"]),
            }
        )

    # GT vs YOLO 교정 + 중복 제거
    corrected = auto_correct(gt, det_boxes)
    corrected = remove_duplicate(corrected)

    # 사용된 category 수집
    used_cat_ids = sorted({int(c["category_id"]) for c in corrected})

    # category id-> name 매핑
    catname_map = dict(merged_info.get("categories", {}))

    # YOLO mapping 보완
    for _, rec in YOLO2COCO.items():
        cid = rec["coco_id"]
        if cid not in catname_map:
            catname_map[cid] = rec["name"]

    categories = [
        {"id": cid, "name": catname_map[cid], "supercategory": "pill"}
        for cid in used_cat_ids
    ]

    # annotation 생성 (xyxy -> COCO xywh)
    anns = []
    ann_id = 1
    for c in corrected:
        x1, y1, x2, y2 = c["bbox"]
        anns.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(c["category_id"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
            }
        )
        ann_id += 1

    # 최종 JSON
    unified = {
        "images": [{"id": img_id, "file_name": img_name, "width": w, "height": h}],
        "annotations": anns,
        "categories": categories,
        "meta": merged_info.get("meta", {}),
    }

    return unified


# 메인 파이프라인
def main():
    # 기존 GT merge
    merged = merge_original_annotations(ANN_DIR)
    logger.info(f"[INFO] 기존 JSON에서 merge된 이미지 수: {len(merged)}")

    # 전체 이미지 순회
    imgs = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    logger.info(f"[INFO] 전체 train 이미지 수: {len(imgs)}")

    for img_path in imgs:
        fname = img_path.name

        # YOLO detection
        det_boxes = detect_boxes(img_path)

        # 기존 GT 여부에 따라 처리 분기
        if fname in merged:
            uni_json = build_unified_json(img_path, merged[fname], det_boxes)
        else:
            # GT 없는 이미지 -> YOLO만으로 pseudo label 생성
            uni_json = build_unified_json(
                img_path,
                {"annotations": [], "categories": {}, "meta": {}},
                det_boxes,
            )

        out_path = OUT_DIR / f"{img_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(uni_json, f, indent=2, ensure_ascii=False)

        logger.info(f"[SAVE] {out_path}")

    logger.info("[DONE] Unified dataset 생성 완료!")


if __name__ == "__main__":
    main()
