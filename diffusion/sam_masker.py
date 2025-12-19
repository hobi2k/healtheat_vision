"""
SAM 기반 알약 마스크 생성기

입력:
- RGB 이미지 (numpy)
- bbox 리스트 (xyxy)

출력:
- 알약 union mask (uint8, 0 or 255)

주의:
- 이 모듈은 '알약 픽셀 보호'가 목적이다.
"""

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


class PillMasker:
    def __init__(self, checkpoint: str, model_type="vit_h", device="cuda"):
        # SAM 모델 로드
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)

    def build_union_mask(
        self,
        image_rgb: np.ndarray,
        bboxes_xyxy: list,
        dilate_px: int = 8,
    ) -> np.ndarray:
        """
        여러 bbox에 대해 SAM mask를 생성한 뒤 OR 연산으로 합친다.

        Args:
            image_rgb: (H, W, 3) RGB 이미지
            bboxes_xyxy: [(x1,y1,x2,y2), ...]
            dilate_px: 마스크 팽창 픽셀 수 (경계 보호)

        Returns:
            pill_mask: (H, W) uint8, 255=알약
        """
        H, W = image_rgb.shape[:2]
        self.predictor.set_image(image_rgb)

        union = np.zeros((H, W), dtype=np.uint8)

        for x1, y1, x2, y2 in bboxes_xyxy:
            masks, scores, _ = self.predictor.predict(
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                multimask_output=True,
            )
            # 가장 confidence 높은 mask 선택
            best = masks[scores.argmax()].astype(np.uint8)
            union = np.maximum(union, best)

        # 경계 침범 방지용 팽창
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)
            )
            union = cv2.dilate(union, kernel, iterations=1)

        return (union * 255).astype(np.uint8)