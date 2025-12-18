#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import sys
import hashlib
from pathlib import Path

# 프로젝트 루트를 path에 추가하여 src 패키지를 인식하게 함
FILE_PATH = Path(__file__).resolve()
ROOT = FILE_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import logger, RAW_IMAGES_DIR, COLLECTED_IMAGES_DIR

class ImageCollector:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = logger
        self.src_root = RAW_IMAGES_DIR
        self.dst_dir = COLLECTED_IMAGES_DIR
        self.exclude_suffix = "_index.png"
        self.existing_hashes = {} # 중복 방지용 해시 장부

    def get_file_hash(self, path: Path) -> str:
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def run(self):
        if not self.src_root.exists():
            self.logger.error(f"원본 폴더가 없습니다: {self.src_root}")
            return

        self.dst_dir.mkdir(parents=True, exist_ok=True)
        
        # 탐색 대상 폴더
        sub_dirs = [p for p in self.src_root.iterdir() if p.is_dir() and (p.name.startswith("TS_") or p.name.startswith("VS_"))]
        
        self.logger.info(f"수집 시작...")

        moved, skipped = 0, 0

        for s_dir in sub_dirs:
            self.logger.info(f"스캔 중: {s_dir.name}")
            for img_path in s_dir.rglob("*.png"):
                if img_path.name.endswith(self.exclude_suffix):
                    continue

                # 핵심 수정: 목적지에 같은 이름의 파일이 이미 있다면? 
                # 내용 비교(해시)도 하지 말고 그냥 스킵합니다. (이름 보존 + 용량 아끼기)
                target_path = self.dst_dir / img_path.name
                if target_path.exists():
                    skipped += 1
                    continue

                if not self.dry_run:
                    # 복사가 아닌 이동(용량 부족 해결)
                    shutil.move(str(img_path), str(target_path))
                
                moved += 1
                if moved % 2000 == 0:
                    self.logger.info(f"진행 상황: {moved}장 완료...")

        self.logger.info(f"최종 완료! 새 파일: {moved}, 이름 중복 제외: {skipped}")

if __name__ == "__main__":
    # 안전하게 테스트해보고 싶다면 dry_run=True로 먼저 실행하세요.
    collector = ImageCollector(dry_run=False)
    collector.run()