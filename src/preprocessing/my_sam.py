from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
from pathlib import Path
from src.config import Config, init_logger

logger = init_logger()

# SAM 모델 로드 (한 번만)
SAM_CKPT = Config.BASE_DIR / "weights/sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"

sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT))
sam.to("cuda")  # GPU 사용
sam_predictor = SamPredictor(sam)

def refine_boxes_with_sam(img_path: Path, det_boxes, min_area_ratio=0.3):
    """
    YOLO bbox → SAM box-prompt → mask → tight bbox 재계산

    [입력]
    - img_path: 이미지 경로
    - det_boxes: detect_boxes() 출력 리스트
    - min_area_ratio: mask 면적 / bbox 면적 최소 비율 (노이즈 제거용)

    [출력]
    - SAM으로 정제된 det_boxes (포맷 동일)
    """

    if not det_boxes:
        return []

    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"[WARN] 이미지 로드 실패 (SAM): {img_path}")
        return det_boxes

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(img_rgb)

    refined = []

    for det in det_boxes:
        x1, y1, x2, y2 = map(int, det["bbox"])

        # box prompt (SAM은 [x1,y1,x2,y2] float32 요구)
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        masks, scores, _ = sam_predictor.predict(
            box=box[None, :],
            multimask_output=True,
        )

        if masks is None or len(masks) == 0:
            refined.append(det)
            continue

        # 가장 score 높은 mask 선택
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]

        # mask → bbox 변환
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            refined.append(det)
            continue

        mx1, my1 = xs.min(), ys.min()
        mx2, my2 = xs.max(), ys.max()

        # 면적 필터링 (너무 작은 조각 제거)
        mask_area = (mx2 - mx1) * (my2 - my1)
        box_area = max(1.0, (x2 - x1) * (y2 - y1))
        if mask_area / box_area < min_area_ratio:
            refined.append(det)
            continue

        refined.append(
            {
                "bbox": [float(mx1), float(my1), float(mx2), float(my2)],
                "category_id": det["category_id"],
                "name": det.get("name", ""),
            }
        )

    return refined
