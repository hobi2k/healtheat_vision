from pathlib import Path
from .visualize import load_coco_annotations, visualize_image

def main():
    coco_path = Path("clip/pseudo_labels.json")
    image_root = Path("clip/unified_dataset/images")   # ★ 추가
    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)

    images, annotations, categories = load_coco_annotations(coco_path)

    # 그룹핑: image_id → annotations
    ann_map = {}
    for ann in annotations:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    for image_id, image_info in images.items():
        anns = ann_map.get(image_id, [])
        output_path = out_dir / f"{image_id}.png"

        # ★ image_root 전달
        visualize_image(
            image_info,
            anns,
            categories,
            image_root=image_root,
            output_path=output_path
        )

    print("Finished visualizing all images.")

if __name__ == "__main__":
    main()
