import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import koreanize_matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.config import Config


def visualize_coco(img_dir: Path, ann_dir: Path, save_dir: Path):
    """
    unified_dataset 구조 전용 시각화:
      images/*.png
      annotations/*.json

    각 이미지별 JSON이 하나씩 존재하므로,
    JSON에서 바로 bbox + class name을 읽어 시각화한다.
    """

    img_dir = Path(img_dir)
    ann_dir = Path(ann_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 목록 가져오기
    images = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])

    print(f"[INFO] 전체 이미지 수: {len(images)}")

    for img_path in images:
        json_path = ann_dir / f"{img_path.stem}.json"
        if not json_path.exists():
            print(f"[WARN] JSON 없음 → {json_path}")
            continue

        with json_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        # 이미지 로드
        pil_img = Image.open(img_path)

        # 카테고리 매핑: {category_id → name}
        category_map = {
            c["id"]: c["name"] for c in coco.get("categories", [])
        }

        anns = coco.get("annotations", [])

        # 시각화 준비
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(pil_img)
        ax.axis("off")

        for ann in anns:
            cid = ann["category_id"]
            cname = category_map.get(cid, f"class_{cid}")

            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)

            ax.text(
                x, y - 4,
                cname,
                fontsize=10,
                color="red",
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
            )

        save_path = save_dir / f"{img_path.stem}_viz.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        print(f"[SAVE] {save_path}")


if __name__ == "__main__":
    visualize_coco(
        img_dir=Config.DATA_DIR / "outputs/aug_images",
        ann_dir=Config.DATA_DIR / "outputs/aug_annotations",
        save_dir=Config.BASE_DIR / "outputs/visualization_aug"
    )
