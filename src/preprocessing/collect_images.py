#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_images.py
- 지정한 여러 root 폴더 아래에 흩어진 PNG들을 한 폴더로 복사(모으기)
- *_index.png 는 제외
- 파일명 충돌 시 _dupXXXX suffix로 회피
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Tuple


SRC_DIRS = [
    # Path("/Users/youuchul/Documents/github/03_projects/01_HealthEat Pill Detection Model/healtheat_vision/data/aihub_downloads/images_raw1_3/TS_3_조합"),
    # Path("/Users/youuchul/Documents/github/03_projects/01_HealthEat Pill Detection Model/healtheat_vision/data/aihub_downloads/images_raw1_3/TS_1_조합"),
    # Path("/Users/youuchul/Documents/github/03_projects/01_HealthEat Pill Detection Model/healtheat_vision/data/aihub_downloads/images_raw1_3/VS_1_조합"),
    Path("/Users/youuchul/Documents/github/03_projects/01_HealthEat Pill Detection Model/healtheat_vision/data/aihub_downloads/images_raw4_5/TS_4_조합"),
    Path("/Users/youuchul/Documents/github/03_projects/01_HealthEat Pill Detection Model/healtheat_vision/data/aihub_downloads/images_raw4_5/TS_5_조합"),
]

DST_DIR = Path("/Users/youuchul/Documents/github/03_projects/01_HealthEat Pill Detection Model/healtheat_vision/data/aihub_downloads/images4_5")

EXCLUDE_SUFFIX = "_index.png"   # 파일명이 이걸로 끝나면 제외
DRY_RUN = False                # True로 바꾸면 실제 복사 없이 카운트만 확인


def iter_pngs(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            print(f"[WARN] not found: {root}")
            continue
        for p in root.rglob("*.png"):
            if p.name.endswith(EXCLUDE_SUFFIX):
                continue
            yield p


def unique_dst_path(dst_dir: Path, filename: str) -> Path:
    """
    dst_dir/filename 이 이미 존재하면 filename에 _dupXXXX 붙여서 유니크하게 만든다.
    """
    candidate = dst_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    i = 1
    while True:
        new_name = f"{stem}_dup{i:04d}{suffix}"
        candidate = dst_dir / new_name
        if not candidate.exists():
            return candidate
        i += 1


def copy_images(src_paths: Iterable[Path], dst_dir: Path) -> Tuple[int, int]:
    """
    return: (copied_count, skipped_count)
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for src in src_paths:
        if not src.is_file():
            skipped += 1
            continue

        dst = unique_dst_path(dst_dir, src.name)

        # 동일 파일(경로 다름)인데 파일명 충돌 나는 케이스를 대비해 항상 dst를 유니크하게 만듦
        if DRY_RUN:
            copied += 1
            continue

        shutil.copy2(src, dst)  # 메타데이터 포함 복사
        copied += 1

        if copied % 2000 == 0:
            print(f"[INFO] copied: {copied}")

    return copied, skipped


def main():
    print("[INFO] Source roots:")
    for d in SRC_DIRS:
        print(f"  - {d}")

    print(f"[INFO] Destination: {DST_DIR}")
    print(f"[INFO] Exclude: *{EXCLUDE_SUFFIX}")
    print(f"[INFO] DRY_RUN: {DRY_RUN}")

    src_paths = list(iter_pngs(SRC_DIRS))
    print(f"[INFO] found pngs (excluded applied): {len(src_paths)}")

    copied, skipped = copy_images(src_paths, DST_DIR)
    print(f"[DONE] copied={copied}, skipped={skipped}, dst={DST_DIR}")


if __name__ == "__main__":
    main()