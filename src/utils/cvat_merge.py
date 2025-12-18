"""
per-image COCO JSON → CVAT import용 단일 COCO
- images / annotations 키가 없는 파일은 자동 skip
"""

import json
from pathlib import Path
from src.config import Config

ANN_DIR = Config.DATA_DIR / "cvat_import" / "annotations"
OUT_FILE = Config.DATA_DIR / "cvat_import" / "coco_instances.json"

images = []
annotations = []
categories = {}

next_img_id = 1
next_ann_id = 1
skipped = []

for json_path in sorted(ANN_DIR.glob("*.json")):
    coco = json.loads(json_path.read_text(encoding="utf-8"))

    # 🔒 COCO 형식 검증
    if "images" not in coco or "annotations" not in coco:
        skipped.append(json_path.name)
        continue

    if not coco["images"]:
        skipped.append(json_path.name)
        continue

    # ---- image ----
    img = coco["images"][0]
    img["id"] = next_img_id
    images.append(img)

    # ---- categories (merge by id) ----
    for c in coco.get("categories", []):
        categories[c["id"]] = c

    # ---- annotations ----
    for ann in coco.get("annotations", []):
        ann["id"] = next_ann_id
        ann["image_id"] = next_img_id
        annotations.append(ann)
        next_ann_id += 1

    next_img_id += 1

merged = {
    "images": images,
    "annotations": annotations,
    "categories": sorted(categories.values(), key=lambda x: x["id"]),
}

OUT_FILE.write_text(
    json.dumps(merged, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"[DONE] CVAT import COCO 생성 완료 → {OUT_FILE}")
print(f"images: {len(images)}, annotations: {len(annotations)}")

if skipped:
    print("[SKIPPED FILES]")
    for name in skipped:
        print(f" - {name}")
