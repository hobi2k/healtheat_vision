"""
Stable Diffusion Inpainting 래퍼

중요:
- diffusion은 배경만 건드린다
- 알약 영역은 mask로 완전히 보호한다
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from .aug_config import AugConfig


class BackgroundInpainter:
    def __init__(self, model_id: str, device="cuda"):
        self.device = device
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        self.pipe.enable_attention_slicing()

    def inpaint(
        self,
        image_pil: Image.Image,
        pill_mask: np.ndarray,
        prompt: str,
        negative_prompt: str,
        seed: int,
        strength: float,
        guidance_scale: float,
        steps: int,
    ) -> Image.Image:
        """
        배경만 인페인팅 수행

        pill_mask:
            255 = 알약 (보호)
            0 = 배경
        """
        # diffusion은 255 영역을 칠하므로 mask 반전
        bg_mask = Image.fromarray(255 - pill_mask, mode="L")

        gen = torch.Generator(self.device).manual_seed(seed)

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=bg_mask,
            generator=gen,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            height=image_pil.height,
            width=image_pil.width,
        )

        # 결과 유효성 검사 (None 방어)
        if out is None or not hasattr(out, "images") or not out.images:
            raise RuntimeError("[SD Inpaint] Pipeline returned no images")

        result = out.images[0]
        if result is None:
            raise RuntimeError("[SD Inpaint] Image is None")

        # padding 제거 (캔버스 복원)
        orig_w, orig_h = image_pil.size
        res_w, res_h = result.size

        if (res_w, res_h) != (orig_w, orig_h):
            left = (res_w - orig_w) // 2
            top = (res_h - orig_h) // 2
            result = result.crop((left, top, left + orig_w, top + orig_h))

        # 최종 좌표계 검증
        if result.size != image_pil.size:
            raise RuntimeError(
                f"[SD Inpaint] Size mismatch after crop: "
                f"orig={image_pil.size}, out={result.size}"
            )

        return result