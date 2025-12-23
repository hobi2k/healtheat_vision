# clip/run_pseudo_labeling.py
from pathlib import Path
from .mapping_loader import load_category_mapping
from .yolo_wrapper import YoloDetector
from .clipping_wrapper import CLIPHelper
from .clip_category_mapping import PillClassifier
from .build_annotations import build_annotations_for_image
import sys

from PIL import Image
import json
import tqdm

# 패키지 루트 등록
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    # 1) 경로 설정
    images_dir = Path("clip/unified_dataset/images")
    mapping_path = Path("clip/unified_dataset/annotations/category_mapping.json")
    yolo_weight = "clip/best.pt"

    # 2) category mapping 로드
    category_mapping = load_category_mapping(mapping_path)

    # 3) YOLO + CLIP 초기화
    yolo = YoloDetector(yolo_weight)
    clip = CLIPHelper(device="cuda")
    pill_classifier = PillClassifier(
        clip_helper=clip,
        category_mapping=category_mapping
    )

    # 4) 이미지 리스트 수집
    image_paths = [
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in [".jpg", ".png"]
    ]

    # 5) COCO JSON 구조
    images = []
    annotations = []
    next_ann_id = 1

    for img_id, img_path in enumerate(tqdm.tqdm(image_paths), start=1):
        pil_img = Image.open(img_path)
        w, h = pil_img.size

        # ★ COCO 규격: file_name은 파일명만 저장해야 함
        images.append({
            "id": img_id,
            "file_name": img_path.name,   # ← 수정됨
            "width": w,
            "height": h
        })

        # YOLO detection + CLIP classification
        anns, next_ann_id = build_annotations_for_image(
            image_path=img_path,
            image_id=img_id,
            yolo=yolo,
            pill_classifier=pill_classifier,
            next_ann_id=next_ann_id
        )
        annotations.extend(anns)

    # 6) COCO categories 구성 (정답 구조)
    categories = []
    for cid, item in category_mapping.items():
        categories.append({
            "id": item["coco_id"],          # COCO ID
            "name": item["name"],
            "supercategory": "pill"
        })

    # 7) 최종 JSON 저장
    out = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open("clip/pseudo_labels.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("저장 완료: pseudo_labels.json")


if __name__ == "__main__":
    main()
