"""
배경 증강 정책 모듈

정책:
- 알약 픽셀은 절대 변경하지 않는다
- 배경만 새로 생성한다
"""

from PIL import Image


class BackgroundAugmentor:
    def __init__(self, masker, inpainter, cfg):
        self.masker = masker
        self.inpainter = inpainter
        self.cfg = cfg

    def augment(
        self,
        image_rgb,
        image_pil: Image.Image,
        bboxes_xyxy: list,
        seed: int,
    ) -> Image.Image:
        """
        단일 이미지에 대해 배경 증강 1회 수행
        """
        pill_mask = self.masker.build_union_mask(
            image_rgb,
            bboxes_xyxy,
            dilate_px=self.cfg.mask_dilate_px,
        )

        return self.inpainter.inpaint(
            image_pil=image_pil,
            pill_mask=pill_mask,
            prompt=self.cfg.random_prompt(),
            negative_prompt=self.cfg.negative_prompt,
            seed=seed,
            strength=self.cfg.strength,
            guidance_scale=self.cfg.guidance_scale,
            steps=self.cfg.num_inference_steps,
        )