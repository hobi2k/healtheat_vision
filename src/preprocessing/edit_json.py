from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import json


# =========================
# CONFIG (여기만 수정)
# =========================
@dataclass
class Config:
    # 입력 JSON 루트 폴더
    src_root: Path = Path("data") / "aihub_downloads" / "raw_val"

    # 출력 JSON 루트 폴더 (원본 보존)
    dst_root: Path = Path("data") / "aihub_downloads" / "raw_val_json_edited"

    # src_root 내부에 이런 폴더가 있으면 스킵
    skip_dir_contains: str = "json_edited"

    # 저장 여부 (False면 dry-run: 변경량만 출력)
    write_output: bool = True

    # True면 dst_root에 동일 폴더 구조로 저장
    keep_structure: bool = True

    # 처리 파일 수 제한 (디버깅용). None이면 전체
    limit: Optional[int] = None


def read_json_safe(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """JSON 읽기(UTF-8). 깨진 파일은 에러 메시지 반환."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return None, f"READ_ERROR: {e}"

    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSON_DECODE_ERROR: {e}"


def validate_schema(data: Dict[str, Any]) -> Optional[str]:
    """최소 스키마 확인."""
    images = data.get("images")
    if not isinstance(images, list) or len(images) == 0:
        return "SCHEMA_ERROR: 'images' missing or empty"
    if "dl_idx" not in images[0]:
        return "SCHEMA_ERROR: images[0].dl_idx missing"
    return None


def extract_dl_info(data: Dict[str, Any]) -> Tuple[Optional[int], str, Optional[str]]:
    """dl_idx -> int 변환, dl_name 확보."""
    img0 = data["images"][0]
    dl_idx = img0.get("dl_idx")

    dl_name = (
        img0.get("dl_name")
        or img0.get("dl_name_kor")
        or img0.get("drug_S")
        or "unknown"
    )

    try:
        dl_idx_int = int(dl_idx) if dl_idx is not None else None
    except Exception:
        dl_idx_int = None

    return dl_idx_int, str(dl_name), str(dl_idx)


def transform_categories_and_annotations(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    - categories 단일 재구성: id = images[0].dl_idx, name = dl_name
    - annotations[*].category_id = 동일 값으로 통일
    """
    dl_idx_int, dl_name, dl_idx_raw = extract_dl_info(data)

    report = {
        "dl_idx_raw": dl_idx_raw,
        "dl_idx_int": dl_idx_int,
        "dl_name": dl_name,
        "categories_before": len(data.get("categories", [])) if isinstance(data.get("categories"), list) else None,
        "annotations_before": len(data.get("annotations", [])) if isinstance(data.get("annotations"), list) else None,
        "annotations_updated": 0,
        "status": "OK",
    }

    if dl_idx_int is None:
        report["status"] = "SKIP_NO_VALID_DL_IDX"
        return data, report

    # categories 재구성
    data["categories"] = [{
        "supercategory": "pill",
        "id": dl_idx_int,
        "name": dl_name,
    }]

    # annotations 동기화
    anns = data.get("annotations", [])
    if isinstance(anns, list):
        for ann in anns:
            if isinstance(ann, dict):
                ann["category_id"] = dl_idx_int
                report["annotations_updated"] += 1

    return data, report


def should_skip(path: Path, cfg: Config) -> bool:
    return cfg.skip_dir_contains in path.parts


def build_dst_path(src_path: Path, cfg: Config) -> Path:
    """src_root 기준 상대경로 유지해서 dst_root에 저장."""
    if not cfg.keep_structure:
        return cfg.dst_root / src_path.name
    rel = src_path.relative_to(cfg.src_root)
    return cfg.dst_root / rel


def process_all(cfg: Config) -> Dict[str, Any]:
    cfg.dst_root.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(cfg.src_root.rglob("*.json"))
    if cfg.limit is not None:
        json_paths = json_paths[: cfg.limit]

    summary: Dict[str, Any] = {
        "total_found": len(json_paths),
        "processed": 0,
        "written": 0,
        "skipped": 0,
        "errors": 0,
        "error_samples": [],   # (path, err)
        "skip_samples": [],    # (path, reason)
        "reports": [],         # per-file report
    }

    for p in json_paths:
        if should_skip(p, cfg):
            summary["skipped"] += 1
            continue

        data, err = read_json_safe(p)
        if err:
            summary["errors"] += 1
            if len(summary["error_samples"]) < 10:
                summary["error_samples"].append((str(p), err))
            continue

        sch_err = validate_schema(data)
        if sch_err:
            summary["skipped"] += 1
            if len(summary["skip_samples"]) < 10:
                summary["skip_samples"].append((str(p), sch_err))
            continue

        data2, rep = transform_categories_and_annotations(data)
        summary["reports"].append({"path": str(p), **rep})
        summary["processed"] += 1

        if rep["status"] != "OK":
            summary["skipped"] += 1
            if len(summary["skip_samples"]) < 10:
                summary["skip_samples"].append((str(p), rep["status"]))
            continue

        if cfg.write_output:
            dst = build_dst_path(p, cfg)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(
                json.dumps(data2, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            summary["written"] += 1

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n==== SUMMARY ====")
    for k in ["total_found", "processed", "written", "skipped", "errors"]:
        print(f"{k}: {summary.get(k)}")

    if summary.get("error_samples"):
        print("\n[ERROR SAMPLES]")
        for p, e in summary["error_samples"]:
            print("-", p, "=>", e)

    if summary.get("skip_samples"):
        print("\n[SKIP SAMPLES]")
        for p, r in summary["skip_samples"]:
            print("-", p, "=>", r)

    print("\n[REPORT SAMPLE 3]")
    for r in summary.get("reports", [])[:3]:
        print(r)


def main() -> None:
    cfg = Config()

    print("SRC:", cfg.src_root)
    print("DST:", cfg.dst_root)
    print("write_output:", cfg.write_output)
    print("keep_structure:", cfg.keep_structure)
    print("limit:", cfg.limit)

    summary = process_all(cfg)
    print_summary(summary)


if __name__ == "__main__":
    main()