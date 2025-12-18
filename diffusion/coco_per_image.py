"""
per-image COCO JSON 생성 전용 모듈

전제:
- 이미지 1장 = COCO JSON 1개
- image_id = 1
- annotation_id = 1..N
"""

def build_per_image_coco(
    image_info: dict,
    annotations: list,
    categories: list,
) -> dict:
    """
    image_info: 단일 image dict
    annotations: 해당 이미지의 annotation list
    categories: 전체 category list
    """

    img = dict(image_info)
    img["id"] = 1

    anns = []
    for i, ann in enumerate(annotations, start=1):
        new_ann = dict(ann)
        new_ann["id"] = i
        new_ann["image_id"] = 1
        anns.append(new_ann)

    return {
        "images": [img],
        "annotations": anns,
        "categories": categories,
    }
