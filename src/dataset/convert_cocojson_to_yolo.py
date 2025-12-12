from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Dict, List, Tuple

import pandas as pd

# --- 경로 규칙: repo_root 기준 ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TRAIN_ANN_DIR    = DATA_DIR / "train_annotations"
SPLITS_DIR       = DATA_DIR / "splits"

YOLO_DIR         = DATA_DIR / "yolo"
YOLO_IMAGES_DIR  = YOLO_DIR / "images"
YOLO_LABELS_DIR  = YOLO_DIR / "labels"

CLASS_MAP_PATH   = REPO_ROOT / "artifacts" / "class_map.csv"

# 기본: 복사(Colab 호환). 로컬에서 빠르게 하려면 symlink=True로 바꿔도 됨.
USE_SYMLINK = False

# bbox clip 여부 (추천 True)
CLIP_BBOX = True


def read_stems(split_file: Path) -> List[str]:
    stems = []
    for line in split_file.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s:
            stems.append(s)
    return stems


def load_class_map(path: Path) -> Dict[int, int]:
    df = pd.read_csv(path)
    # orig_id -> yolo_id
    return {int(r.orig_id): int(r.yolo_id) for r in df.itertuples()}


def build_ann_index(ann_root: Path) -> Dict[str, Path]:
    """
    train_annotations가 깊게 들어가 있으니, stem -> json_path 인덱스를 한번 만든다.
    (속도/안정성 위해)
    """
    idx = {}
    for jp in ann_root.rglob("*.json"):
        idx[jp.stem] = jp
    return idx


def find_image(stem: str) -> Path | None:
    # 지금은 png만 있다고 했으니 png 우선
    p = TRAIN_IMAGES_DIR / f"{stem}.png"
    if p.exists():
        return p
    # 혹시 몰라서 jpg/jpeg도 대비
    for ext in ["jpg", "jpeg", "png"]:
        p2 = TRAIN_IMAGES_DIR / f"{stem}.{ext}"
        if p2.exists():
            return p2
    return None


def ensure_dirs():
    for split in ["train", "val"]:
        (YOLO_IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (YOLO_LABELS_DIR / split).mkdir(parents=True, exist_ok=True)


def clip_xywh(x: float, y: float, w: float, h: float, W: float, H: float) -> Tuple[float, float, float, float]:
    # xywh -> clip
    x2 = x + w
    y2 = y + h
    x  = max(0.0, min(x, W))
    y  = max(0.0, min(y, H))
    x2 = max(0.0, min(x2, W))
    y2 = max(0.0, min(y2, H))
    w  = x2 - x
    h  = y2 - y
    return x, y, w, h


def xywh_to_yolo(x: float, y: float, w: float, h: float, W: float, H: float) -> Tuple[float, float, float, float]:
    # COCO xywh (top-left) -> YOLO normalized (cx, cy, w, h)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx / W, cy / H, w / W, h / H


def write_label_txt(out_path: Path, lines: List[str]):
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_or_link(src: Path, dst: Path, symlink: bool = False):
    if dst.exists():
        return
    if symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def convert_split(split_name: str, stems: List[str], ann_index: Dict[str, Path], orig_to_yolo: Dict[int, int]):
    n_img_ok = 0
    n_ann_ok = 0
    n_missing_img = 0
    n_missing_ann = 0
    n_bad_box = 0

    for stem in stems:
        img_path = find_image(stem)
        if img_path is None:
            n_missing_img += 1
            continue

        ann_path = ann_index.get(stem)
        if ann_path is None:
            n_missing_ann += 1
            continue

        data = json.loads(ann_path.read_text(encoding="utf-8", errors="replace"))

        images = data.get("images", [])
        if not images:
            n_missing_ann += 1
            continue

        img_info = images[0]
        W = float(img_info.get("width", 0))
        H = float(img_info.get("height", 0))
        if W <= 0 or H <= 0:
            n_missing_ann += 1
            continue

        # 라벨 생성
        label_lines = []
        for ann in data.get("annotations", []):
            orig_id = int(ann.get("category_id"))
            if orig_id not in orig_to_yolo:
                # class_map에 없는 클래스면 스킵(경고 대신 카운트만)
                continue
            yolo_id = orig_to_yolo[orig_id]

            bbox = ann.get("bbox", None)

            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                n_bad_box += 1
                continue

            x, y, w, h = bbox
            x, y, w, h = float(x), float(y), float(w), float(h)

            if CLIP_BBOX:
                x, y, w, h = clip_xywh(x, y, w, h, W, H)

            if w <= 0 or h <= 0:
                n_bad_box += 1
                continue

            cx, cy, nw, nh = xywh_to_yolo(x, y, w, h, W, H)

            # YOLO 포맷: class cx cy w h (normalized)
            label_lines.append(f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # 이미지 복사/링크
        out_img = YOLO_IMAGES_DIR / split_name / img_path.name
        copy_or_link(img_path, out_img, symlink=USE_SYMLINK)
        n_img_ok += 1

        # 라벨 저장
        out_lbl = YOLO_LABELS_DIR / split_name / f"{stem}.txt"
        write_label_txt(out_lbl, label_lines)
        n_ann_ok += 1

    print(f"[{split_name}] stems={len(stems)} img_ok={n_img_ok} ann_ok={n_ann_ok} "
          f"missing_img={n_missing_img} missing_ann={n_missing_ann} bad_box={n_bad_box}")


def main():
    ensure_dirs()

    # split stems
    train_stems = read_stems(SPLITS_DIR / "train.txt")
    val_stems   = read_stems(SPLITS_DIR / "val.txt")

    # class map
    orig_to_yolo = load_class_map(CLASS_MAP_PATH)
    print("class_map:", len(orig_to_yolo))

    # annotation index
    print("indexing annotations...")
    ann_index = build_ann_index(TRAIN_ANN_DIR)
    print("ann_index:", len(ann_index))

    convert_split("train", train_stems, ann_index, orig_to_yolo)
    convert_split("val",   val_stems,   ann_index, orig_to_yolo)

    # 결과 요약
    n_imgs = len(list((YOLO_IMAGES_DIR / "train").glob("*"))) + len(list((YOLO_IMAGES_DIR / "val").glob("*")))
    n_lbls = len(list((YOLO_LABELS_DIR / "train").glob("*.txt"))) + len(list((YOLO_LABELS_DIR / "val").glob("*.txt")))
    print("done. yolo images:", n_imgs, "labels:", n_lbls)


if __name__ == "__main__":
    main()