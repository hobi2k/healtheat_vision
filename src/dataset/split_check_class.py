#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from collections import Counter
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

LABEL_TRAIN_DIR = REPO_ROOT / "data" / "yolo" / "labels" / "train"
LABEL_VAL_DIR   = REPO_ROOT / "data" / "yolo" / "labels" / "val"

CLASS_MAP_PATH  = REPO_ROOT / "artifacts" / "class_map.csv"

OUT_DIR  = Path(__file__).resolve().parent / "csv"
OUT_PATH = OUT_DIR / "split_class_counts.csv"


def read_class_map(path: Path) -> pd.DataFrame:
    # 탭/콤마 등 구분자 자동 추정
    df = pd.read_csv(path, sep=None, engine="python")

    required = {"orig_id", "yolo_id", "class_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"class_map.csv 컬럼이 부족합니다. missing={missing}, columns={list(df.columns)}")

    df = df[["orig_id", "yolo_id", "class_name"]].copy()
    df["orig_id"] = pd.to_numeric(df["orig_id"], errors="coerce").astype("Int64")
    df["yolo_id"] = pd.to_numeric(df["yolo_id"], errors="coerce").astype("Int64")

    bad = df[df["yolo_id"].isna() | df["orig_id"].isna()]
    if len(bad) > 0:
        raise ValueError(f"class_map에 숫자로 변환 불가 행이 있습니다:\n{bad}")

    df["orig_id"] = df["orig_id"].astype(int)
    df["yolo_id"] = df["yolo_id"].astype(int)

    # yolo_id 중복 방지
    if df["yolo_id"].duplicated().any():
        dup = df[df["yolo_id"].duplicated(keep=False)].sort_values("yolo_id")
        raise ValueError(f"class_map에 yolo_id 중복이 있습니다:\n{dup}")

    return df.sort_values("yolo_id").reset_index(drop=True)


def scan_labels(label_dir: Path) -> tuple[Counter, Counter, int]:
    """
    return:
      - box_count[yolo_id]  : 해당 클래스 박스(라인) 수
      - file_count[yolo_id] : 해당 클래스가 등장한 txt 파일 수(한 파일에 여러 박스여도 1)
      - bad_lines           : 파싱 실패 라인 수
    """
    box_count = Counter()
    file_count = Counter()
    bad_lines = 0

    if not label_dir.exists():
        raise FileNotFoundError(f"라벨 폴더가 없습니다: {label_dir}")

    txt_paths = sorted(label_dir.rglob("*.txt"))
    for p in txt_paths:
        seen_in_file = set()
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue

            parts = ln.split()
            # YOLO 라벨은 보통 5개(클래스 cx cy w h) 이상
            if len(parts) < 5:
                bad_lines += 1
                continue

            try:
                yid = int(parts[0])
            except ValueError:
                bad_lines += 1
                continue

            box_count[yid] += 1
            seen_in_file.add(yid)

        for yid in seen_in_file:
            file_count[yid] += 1

    return box_count, file_count, bad_lines


def main():
    class_map = read_class_map(CLASS_MAP_PATH)

    train_box, train_file, train_bad = scan_labels(LABEL_TRAIN_DIR)
    val_box,   val_file,   val_bad   = scan_labels(LABEL_VAL_DIR)

    # 매핑 기준 전체 클래스에 대해 집계값 붙이기
    df = class_map.copy()

    df["train_box_count"]  = df["yolo_id"].map(train_box).fillna(0).astype(int)
    df["train_file_count"] = df["yolo_id"].map(train_file).fillna(0).astype(int)
    df["val_box_count"]    = df["yolo_id"].map(val_box).fillna(0).astype(int)
    df["val_file_count"]   = df["yolo_id"].map(val_file).fillna(0).astype(int)

    df["missing_in_train"] = df["train_box_count"].eq(0)
    df["missing_in_val"]   = df["val_box_count"].eq(0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    # 콘솔 요약
    missing_train = df[df["missing_in_train"]][["yolo_id", "orig_id", "class_name"]]
    missing_val   = df[df["missing_in_val"]][["yolo_id", "orig_id", "class_name"]]

    print(f"[OK] saved: {OUT_PATH}")
    print(f"[INFO] train_bad_lines={train_bad}, val_bad_lines={val_bad}")
    print(f"[INFO] missing classes in train: {len(missing_train)}")
    if len(missing_train) > 0:
        print(missing_train.to_string(index=False))
    print(f"[INFO] missing classes in val: {len(missing_val)}")
    if len(missing_val) > 0:
        print(missing_val.to_string(index=False))


if __name__ == "__main__":
    main()