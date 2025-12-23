from pathlib import Path
from PIL import Image
from .yolo_wrapper import YoloDetector
from .clip_category_mapping import PillClassifier

def build_annotations_for_image(
    image_path: Path,
    image_id: int,
    yolo: YoloDetector,
    pill_classifier: PillClassifier,
    next_ann_id: int
):
    boxes, scores = yolo.detect(str(image_path))
    img = Image.open(image_path).convert("RGB")

    annotations = []

    for box in boxes:
        x1, y1, x2, y2 = box
        crop = img.crop((x1, y1, x2, y2))

        # 1) local_id 예측 (int 또는 str일 수 있음)
        local_id = pill_classifier.classify_pill(crop)

        # ★ 반드시 문자열로 변환해서 mapping과 타입 일치시킴
        local_id_str = str(local_id)

        # 3) local_id → COCO ID 변환
        coco_id = pill_classifier.mapping[local_id_str]["coco_id"]

        w = x2 - x1
        h = y2 - y1

        ann = {
            "id": next_ann_id,
            "image_id": image_id,
            "category_id": coco_id,  # COCO ID 저장
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0,
        }

        annotations.append(ann)
        next_ann_id += 1

    return annotations, next_ann_id
