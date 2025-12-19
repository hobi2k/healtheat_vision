from pathlib import Path
import pandas as pd
import yaml

def main():
    root = Path(__file__).resolve().parents[2]  # repo root 기준 (src/dataset/..)
    data_dir = root / "data"
    artifacts_dir = root / "artifacts"
    class_map_path = artifacts_dir / "class_map.csv"

    yolo_yaml_path = root / "configs" / "yolo_data.yaml"
    yolo_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(class_map_path)
    df = df.sort_values("yolo_id")

    names = {int(r.yolo_id): str(r.class_name) for r in df.itertuples()}

    yolo_cfg = {
        "path": "data/yolo",           # 고정된 상대 경로로 변경
        "train": "images/train",
        "val": "images/val",
        "nc": int(df["yolo_id"].max() + 1),
        "names": names,
    }

    with open(yolo_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yolo_cfg, f, allow_unicode=True, sort_keys=False)

    print("✅ updated:", yolo_yaml_path)
    print("nc:", yolo_cfg["nc"])

if __name__ == "__main__":
    main()