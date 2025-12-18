"""
CVAT export per-image COCO → 원래 coco_id 복원

전제:
- CVAT import 단계에서 coco_id → cvat_id (1..N)로 변환됨
- cvat_id는 category_mapping.json에 없고, import 시점에 생성된 값
- 따라서 여기서도 동일한 규칙(coco_id 기준 정렬)으로 cvat_id를 재생성해야 함
"""

import json
from pathlib import Path
from src.config import Config

# ================== 경로 설정 ==================
ANN_DIR = Config.DATA_DIR / "cvat_export" / "annotations"
OUT_DIR = Config.DATA_DIR / "restored_dataset" / "annotations"
MAPPING = Config.DATA_DIR / "unified_dataset_v2/annotations/category_mapping.normalized.json"
# ===============================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------
# 1. category_mapping 로드
# ------------------------------------------------
mapping = json.loads(MAPPING.read_text(encoding="utf-8"))

# ------------------------------------------------
# 2. coco_id 기준 정렬 → cvat_id 재생성
#    (import 단계와 반드시 동일해야 함)
# ------------------------------------------------
items = sorted(mapping.values(), key=lambda x: x["coco_id"])

# cvat_id -> coco_id, name
cvatid_to_coco = {}

for cvat_id, item in enumerate(items, start=1):
    cvatid_to_coco[cvat_id] = {
        "coco_id": item["coco_id"],
        "name": item["name"],
    }

# ------------------------------------------------
# 3. per-image COCO JSON 복원
# ------------------------------------------------
for json_path in ANN_DIR.glob("*.json"):
    coco = json.loads(json_path.read_text(encoding="utf-8"))

    # ---- annotations: cvat_id → coco_id ----
    for ann in coco.get("annotations", []):
        cvat_id = ann["category_id"]
        if cvat_id not in cvatid_to_coco:
            raise KeyError(
                f"[ERROR] cvat_id {cvat_id} not found in mapping "
                f"(file: {json_path.name})"
            )
        ann["category_id"] = cvatid_to_coco[cvat_id]["coco_id"]

    # ---- categories: 이 이미지에 등장한 coco_id만 재작성 ----
    used_coco_ids = sorted({ann["category_id"] for ann in coco.get("annotations", [])})

    coco["categories"] = [
        {
            "id": coco_id,
            "name": next(
                v["name"] for v in cvatid_to_coco.values()
                if v["coco_id"] == coco_id
            ),
            "supercategory": "pill",
        }
        for coco_id in used_coco_ids
    ]

    # ---- 저장 ----
    out_path = OUT_DIR / json_path.name
    out_path.write_text(
        json.dumps(coco, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

print("[DONE] per-image COCO coco_id 복원 완료")
print(f"입력: {ANN_DIR}")
print(f"출력: {OUT_DIR}")
