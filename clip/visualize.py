from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import koreanize_matplotlib 
import matplotlib.patches as patches

def load_coco_annotations(coco_path: Path):
    import json
    with coco_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # COCO images → {id → {info}}
    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]
    categories = {c["id"]: c["name"] for c in data["categories"]}

    return images, annotations, categories


def visualize_image(image_info, anns, categories, image_root: Path, output_path: Path):
    # ★ file_name은 파일명만 있으므로 image_root로 보정
    img_path = image_root / image_info["file_name"]

    img = Image.open(img_path).convert("RGB")

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    for ann in anns:
        x, y, w, h = ann["bbox"]
        cid = ann["category_id"]

        rect = patches.Rectangle(
            (x, y),
            w, h,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        label = categories.get(cid, str(cid))
        ax.text(x, y - 2, label, color='yellow', fontsize=10, backgroundcolor='black')

    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
