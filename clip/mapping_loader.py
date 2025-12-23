from pathlib import Path
import json

def load_category_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k):v for k, v in raw.items()}

def list_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]