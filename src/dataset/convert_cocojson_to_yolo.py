'''
[FILE] convert_cocojson_to_yolo.py
[DESCRIPTION]
    - 초기 학습 데이터셋(AIHub)의 COCO JSON 어노테이션을 YOLO txt 형식으로 변환합니다.
    - 데이터셋을 train/val split 리스트(txt)에 따라 물리적으로 분류합니다.
    - paths.py의 통합 경로 시스템을 사용하여 데이터 정합성을 유지합니다.
[STATUS]
    - 2025-12-19: paths.py 통합 및 리팩토링 완료
    - 특이사항: 이미지 파일은 png를 우선 탐색하며, 필요 시 symlink 사용 가능
'''

from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict
import pandas as pd
import sys
import os

# 프로젝트 루트 경로 추가 및 paths 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths, logger


# 설정 상수
USE_SYMLINK = False # 로컬 환경에서는 True로 변경 시 속도 향상 및 용량 절약 가능
CLIP_BBOX = True

def read_stems(split_file: Path) -> List[str]:
    """split용 txt 파일에서 이미지 파일명(stem) 목록을 읽어옵니다."""
    if not split_file.exists():
        logger.warning(f"Split 파일을 찾을 수 없습니다: {split_file}")
        return []
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

def load_class_map(path: Path) -> Dict[int, int]:
    """orig_id -> yolo_id 매핑 딕셔너리 생성"""
    df = pd.read_csv(path)
    return {int(r.orig_id): int(r.yolo_id) for r in df.itertuples()}

def build_ann_index(ann_root: Path) -> Dict[str, List[Path]]:
    """파일명(stem)을 키로 하여 해당되는 모든 JSON 경로를 인덱싱합니다."""
    idx: DefaultDict[str, List[Path]] = defaultdict(list)
    for jp in ann_root.rglob("*.json"):
        idx[jp.stem].append(jp)
    return dict(idx)

def find_image(stem: str) -> Path | None:
    """파일명과 확장자를 조합하여 실제 이미지 파일 경로를 찾습니다."""
    # paths.py에 정의된 원본 이미지 디렉토리 사용
    for ext in ["png", "jpg", "jpeg"]:
        p = paths.TRAIN_IMAGES_DIR / f"{stem}.{ext}"
        if p.exists():
            return p
    return None

def clip_xywh(x: float, y: float, w: float, h: float, W: float, H: float) -> Tuple[float, float, float, float]:
    """이미지 범위를 벗어나는 bbox를 경계면에 맞게 자릅니다."""
    x2, y2 = x + w, y + h
    x, y = max(0.0, min(x, W)), max(0.0, min(y, H))
    x2, y2 = max(0.0, min(x2, W)), max(0.0, min(y2, H))
    return x, y, x2 - x, y2 - y

def xywh_to_yolo(x: float, y: float, w: float, h: float, W: float, H: float) -> Tuple[float, float, float, float]:
    """COCO(top-left xywh) 형식을 YOLO(center xywh normalized) 형식으로 변환합니다."""
    return (x + w / 2.0) / W, (y + h / 2.0) / H, w / W, h / H

def convert_split(split_name: str, stems: List[str], ann_index: Dict[str, List[Path]], orig_to_yolo: Dict[int, int]):
    """특정 split(train/val)에 대해 변환 및 파일 복사를 수행합니다."""
    
    # YOLO 표준 하위 디렉토리 정의 (이미지/라벨 저장용)
    split_img_dir = paths.YOLO_DIR / "images" / split_name
    split_lbl_dir = paths.YOLO_DIR / "labels" / split_name
    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_lbl_dir.mkdir(parents=True, exist_ok=True)

    counts = {"img_ok": 0, "ann_ok": 0, "miss_img": 0, "miss_ann": 0, "bad_box": 0}

    for stem in tqdm(stems, desc=f"Converting {split_name}"):
        img_path = find_image(stem)
        if not img_path:
            counts["miss_img"] += 1; continue

        ann_paths = ann_index.get(stem)
        if not ann_paths:
            counts["miss_ann"] += 1; continue

        label_lines = []
        W = H = None

        for ann_path in ann_paths:
            try:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
                if not data.get("images"): continue
                
                img_info = data["images"][0]
                W, H = float(img_info.get("width", 0)), float(img_info.get("height", 0))
                
                for ann in data.get("annotations", []):
                    orig_id = int(ann.get("category_id"))
                    if orig_id not in orig_to_yolo: continue
                    
                    bbox = ann.get("bbox")
                    if not bbox or len(bbox) != 4:
                        counts["bad_box"] += 1; continue

                    x, y, w, h = map(float, bbox)
                    if CLIP_BBOX: x, y, w, h = clip_xywh(x, y, w, h, W, H)
                    if w <= 0 or h <= 0: continue

                    cx, cy, nw, nh = xywh_to_yolo(x, y, w, h, W, H)
                    label_lines.append(f"{orig_to_yolo[orig_id]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            except Exception: continue

        if label_lines:
            # 이미지 복사/링크
            out_img = split_img_dir / img_path.name
            if not out_img.exists():
                if USE_SYMLINK: os.symlink(img_path, out_img)
                else: shutil.copy2(img_path, out_img)
            
            # 라벨 저장
            out_lbl = split_lbl_dir / f"{stem}.txt"
            out_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            
            counts["img_ok"] += 1
            counts["ann_ok"] += 1

    logger.info(f"[{split_name}] 완료: {counts}")

def main():
    paths.ensure_dirs()
    
    # 1. 메타데이터 로드 (paths.py 활용)
    orig_to_yolo = load_class_map(paths.CLASS_MAP_PATH)
    # 기존 코드에서 사용하던 data/splits/train.txt 등의 경로 반영
    train_stems = read_stems(paths.DATA_DIR / "splits" / "train.txt")
    val_stems = read_stems(paths.DATA_DIR / "splits" / "val.txt")

    # 2. 어노테이션 인덱싱
    logger.info("어노테이션 인덱싱 중...")
    ann_index = build_ann_index(paths.TRAIN_ANNOTATIONS_DIR)

    # 3. 변환 실행
    convert_split("train", train_stems, ann_index, orig_to_yolo)
    convert_split("val", val_stems, ann_index, orig_to_yolo)

    logger.info("✅ 모든 변환 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()