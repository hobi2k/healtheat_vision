import json
import re
from pathlib import Path
from src.config import Config

MAPPING = Config.DATA_DIR / "unified_dataset_v2/annotations/category_mapping.json"
OUT = Config.DATA_DIR / "unified_dataset_v2/annotations/category_mapping.normalized.json"

def normalize(text: str) -> str:
    # NBSP 제거 + trim + 다중 공백 정리
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

mapping = json.loads(MAPPING.read_text(encoding="utf-8"))

for k, v in mapping.items():
    v["name"] = normalize(v["name"])

OUT.write_text(
    json.dumps(mapping, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"[DONE] 라벨 정규화 완료 → {OUT}")
