"""
category_mapping.json -> CVAT Labels (Raw) JSON 생성기

- 입력: YOLO class id -> { coco_id, name } 구조의 JSON
- 출력: CVAT Raw Labels에 그대로 붙여넣을 JSON 배열
"""

import json
from pathlib import Path
from src.config import Config

INPUT_JSON = Config.DATA_DIR / "unified_dataset_v2/annotations/category_mapping.normalized.json"
OUTPUT_JSON = Config.DATA_DIR / "cvat_labels.json"
SORT_BY = "coco_id"  # "coco_id" 또는 "name"

def main():
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.values())
    items.sort(key=lambda x: x["coco_id"])

    cvat_labels = [
        {
            "name": item["name"],
            "type": "rectangle",
            "attributes": []
        }
        for item in items
    ]

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(cvat_labels, f, indent=2, ensure_ascii=False)

    print(f"[DONE] CVAT labels 생성 완료: {OUTPUT_JSON}")
    print(f"총 라벨 수: {len(cvat_labels)}")

if __name__ == "__main__":
    main()
