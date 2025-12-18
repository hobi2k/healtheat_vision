"""
COCO 어노테이션 입출력 및 ID 관리 모듈

책임:
- COCO JSON 로드 / 저장
- image_id, annotation_id 관리
- image_id -> annotation 묶기

비책임:
- 이미지 처리
- 모델 호출
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_coco(path: Path) -> Dict:
    """
    COCO 데이터를 로드한다.

    - path가 파일인 경우:
        단일 COCO JSON으로 간주하고 그대로 로드한다.
    - path가 디렉토리인 경우:
        디렉토리 내부의 모든 per-image COCO JSON을 순회하여
        하나의 dataset-level COCO로 병합한다.

    반환 포맷은 항상 COCO dataset format:
    {
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }
    """
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if not path.is_dir():
        raise ValueError(f"COCO 경로가 파일도 디렉토리도 아님: {path}")

    return _load_coco_from_directory(path)


def _load_coco_from_directory(dir_path: Path) -> Dict:
    """
    per-image COCO JSON 디렉토리를 읽어
    ID 충돌 없이 하나의 dataset-level COCO로 병합한다.
    """
    images: List[Dict] = []
    annotations: List[Dict] = []
    categories: List[Dict] = []

    next_image_id = 0
    next_ann_id = 0

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"COCO JSON 파일이 없습니다: {dir_path}")

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        if not coco.get("images"):
            continue

        # per-image COCO 전제: images는 1개
        src_image = coco["images"][0]
        src_image_id = src_image["id"]

        # 새 image_id 부여
        next_image_id += 1
        new_image_id = next_image_id

        new_image = dict(src_image)
        new_image["id"] = new_image_id
        images.append(new_image)

        # annotations 재맵
        for ann in coco.get("annotations", []):
            if ann["image_id"] != src_image_id:
                continue  # 방어적 처리

            next_ann_id += 1
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = new_image_id
            annotations.append(new_ann)

        # categories는 최초 한 번만
        if not categories:
            categories = coco.get("categories", [])

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

def save_coco(coco: dict, path: Path):
    """COCO JSON 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)


def group_annotations_by_image(annotations: list) -> dict:
    """
    annotation 리스트를 image_id 기준으로 묶는다.

    반환:
        { image_id: [ann1, ann2, ...] }
    """
    grouped = defaultdict(list)
    for ann in annotations:
        grouped[ann["image_id"]].append(ann)
    return grouped


def get_next_ids(coco: dict):
    """
    COCO에서 다음에 사용할 image_id, annotation_id 계산
    """
    max_image_id = max(img["id"] for img in coco["images"])
    max_ann_id = max(ann["id"] for ann in coco["annotations"])
    return max_image_id, max_ann_id