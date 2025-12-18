"""
per-image COCO JSON → CVAT import용 COCO 변환

- 입력:
  - unified_dataset_v2/annotations/*.json (이미지당 1개 COCO)
  - category_mapping.json (coco_id, name만 포함)

- 출력:
  - cvat_import/annotations/*.json
  - CVAT가 요구하는 연속 cvat_id (1..N) 사용

※ category_mapping.json에는 cvat_id가 없어도 됨
※ cvat_id는 이 스크립트에서 "파생 생성"함
"""

import json
from pathlib import Path
from src.config import Config

# ================== 경로 설정 ==================
ANN_DIR = Config.DATA_DIR / "unified_dataset_v2" / "annotations"
OUT_DIR = Config.DATA_DIR / "cvat_import" / "annotations"
MAPPING = Config.DATA_DIR / "unified_dataset_v2/annotations/category_mapping.normalized.json"
# ===============================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------
# 1. category_mapping 로드
#    구조 예:
#    {
#      "0": { "coco_id": 1899, "name": "보령부스파정 5mg" },
#      "1": { "coco_id": 27732, "name": "트윈스타정 40/5mg" }
#    }
# ------------------------------------------------
mapping = json.loads(MAPPING.read_text(encoding="utf-8"))

# ------------------------------------------------
# 2. coco_id 기준 정렬 → cvat_id 자동 생성
#    (CVAT는 1부터 연속 정수만 안정적으로 처리)
# ------------------------------------------------
items = sorted(mapping.values(), key=lambda x: x["coco_id"])

# coco_id -> { cvat_id, name }
cocoid_to_cvat = {}
# cvat_id -> name (categories 재작성용)
cvatid_to_name = {}

for cvat_id, item in enumerate(items, start=1):
    cocoid_to_cvat[item["coco_id"]] = {
        "cvat_id": cvat_id,
        "name": item["name"],
    }
    cvatid_to_name[cvat_id] = item["name"]

# ------------------------------------------------
# 3. per-image COCO JSON 처리
# ------------------------------------------------
for json_path in ANN_DIR.glob("*.json"):
    if json_path.name == "category_mapping.json":
        continue

    coco = json.loads(json_path.read_text(encoding="utf-8"))

    # ---- annotations: coco_id → cvat_id ----
    for ann in coco.get("annotations", []):
        coco_id = ann["category_id"]
        if coco_id not in cocoid_to_cvat:
            raise KeyError(
                f"[ERROR] coco_id {coco_id} not found in category_mapping.json "
                f"(file: {json_path.name})"
            )
        ann["category_id"] = cocoid_to_cvat[coco_id]["cvat_id"]

    # ---- categories: 이 이미지에 등장한 cvat_id만 재작성 ----
    used_cvat_ids = sorted({ann["category_id"] for ann in coco.get("annotations", [])})

    coco["categories"] = [
        {
            "id": cvat_id,
            "name": cvatid_to_name[cvat_id],
            "supercategory": "pill",
        }
        for cvat_id in used_cvat_ids
    ]

    # ---- 저장 ----
    out_path = OUT_DIR / json_path.name
    out_path.write_text(
        json.dumps(coco, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

print("[DONE] per-image CVAT import용 COCO 생성 완료")
print(f"입력: {ANN_DIR}")
print(f"출력: {OUT_DIR}")
