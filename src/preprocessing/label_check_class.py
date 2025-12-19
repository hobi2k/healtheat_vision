#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT  = REPO_ROOT / "data" / "aihub_downloads" / "raw_label_zip"

# 혹시 폴더가 더 있어도 자동 처리 (이름이 TL_*_조합 형태면 전부)
FOLDER_GLOB = "TL_*_조합"


def safe_read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    except Exception:
        return None


def extract_pairs_from_json(d: dict) -> list[tuple[str, str]]:
    """
    JSON 내 images 배열에서 (dl_idx, dl_name) 리스트 추출
    """
    pairs: list[tuple[str, str]] = []
    imgs = d.get("images", [])
    if not isinstance(imgs, list):
        return pairs

    for it in imgs:
        if not isinstance(it, dict):
            continue

        dl_idx = it.get("dl_idx")
        dl_name = it.get("dl_name")

        if dl_idx is None or dl_name is None:
            continue

        dl_idx_s = str(dl_idx).strip()
        dl_name_s = str(dl_name).strip()

        if not dl_idx_s or not dl_name_s:
            continue

        pairs.append((dl_idx_s, dl_name_s))

    return pairs


def process_folder(folder: Path) -> pd.DataFrame:
    json_paths = sorted(folder.rglob("*.json"))

    # 고유쌍 + 통계(등장 파일 수)
    pair_file_count = Counter()           # (dl_idx, dl_name) -> file count
    dl_idx_to_names = defaultdict(set)    # dl_idx -> set(names)
    sample_path = {}                      # (dl_idx, dl_name) -> first json path

    bad_json = 0

    for jp in json_paths:
        d = safe_read_json(jp)
        if d is None:
            bad_json += 1
            continue

        pairs = extract_pairs_from_json(d)
        if not pairs:
            continue

        # 한 json에서 같은 pair가 여러 번 나오면 1번만 카운트
        for pair in set(pairs):
            pair_file_count[pair] += 1
            dl_idx_to_names[pair[0]].add(pair[1])
            sample_path.setdefault(pair, str(jp))

    rows = []
    for (dl_idx, dl_name), cnt in pair_file_count.items():
        rows.append({
            "dl_idx": dl_idx,
            "dl_name": dl_name,
            "json_file_count": cnt,
            "sample_json": sample_path.get((dl_idx, dl_name), ""),
        })

    df = pd.DataFrame(rows).sort_values(["dl_idx", "dl_name"]).reset_index(drop=True)

    # dl_idx 하나에 이름이 여러 개인 경우(이상치 확인용) 플래그
    if not df.empty:
        multi_name = {k: v for k, v in dl_idx_to_names.items() if len(v) > 1}
        df["dl_idx_has_multiple_names"] = df["dl_idx"].map(lambda x: x in multi_name)

    print(f"[INFO] {folder.name}: json={len(json_paths)} bad_json={bad_json} unique_pairs={len(df)}")
    return df


def main():
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"raw_label_zip 경로가 없습니다: {RAW_ROOT}")

    folders = sorted(RAW_ROOT.glob(FOLDER_GLOB))
    if not folders:
        print(f"[WARN] '{FOLDER_GLOB}' 패턴 폴더를 찾지 못했습니다: {RAW_ROOT}")
        return

    for folder in folders:
        if not folder.is_dir():
            continue

        df = process_folder(folder)

        out_path = RAW_ROOT / f"{folder.name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()