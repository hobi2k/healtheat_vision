"""
SAM + Stable Diffusion Inpainting

역할:
- 단일 이미지 증강 (인터랙티브 미리보기)
- YOLO bbox -> SAM mask 시각화
- 증강 이미지 저장
- YOLO txt 그대로 저장 (파일명만 증강 이미지에 맞춤)
"""

import time
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import numpy as np
from PIL import Image

# 프로젝트 모듈
from diffusion.aug_config import AugConfig
from diffusion.sam_masker import PillMasker
from diffusion.inpainting import BackgroundInpainter
from diffusion.model_hub import prepare_models


# YOLO 유틸
def parse_yolo_text(yolo_text: str) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO txt 문자열 파싱

    입력 예:
        0 0.5 0.5 0.2 0.3
        0 0.3 0.4 0.1 0.15
    """
    boxes = []
    for raw in yolo_text.strip().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"잘못된 YOLO 라벨 형식: {raw}")
        cls, cx, cy, w, h = parts
        boxes.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return boxes


def yolo_to_xyxy(
    yolo_boxes: List[Tuple[int, float, float, float, float]],
    img_w: int,
    img_h: int,
) -> List[List[float]]:
    """
    YOLO (cx,cy,w,h normalized) -> xyxy (pixel)
    """
    out = []
    for _, cx, cy, bw, bh in yolo_boxes:
        bw_px = bw * img_w
        bh_px = bh * img_h
        x_c = cx * img_w
        y_c = cy * img_h

        x1 = max(0.0, x_c - bw_px / 2)
        y1 = max(0.0, y_c - bh_px / 2)
        x2 = min(img_w - 1.0, x_c + bw_px / 2)
        y2 = min(img_h - 1.0, y_c + bh_px / 2)

        out.append([x1, y1, x2, y2])
    return out


# 전역 초기화 (한 번만)
cfg = AugConfig()

prepare_models(cfg)

cfg.out_image_dir.mkdir(parents=True, exist_ok=True)
cfg.out_label_dir.mkdir(parents=True, exist_ok=True)

masker = PillMasker(
    checkpoint=str(cfg.sam_checkpoint),
    model_type=cfg.sam_model_type,
    device=cfg.sam_device,
)

inpainter = BackgroundInpainter(
    model_id=cfg.sd_inpaint_model,
    device=cfg.sd_device,
)


# Gradio 실행 함수
def run_augmentation(
    image_pil: Image.Image,
    yolo_text: str,
    user_prompt: str,
    user_negative_prompt: str,
    seed: int,
    strength: float,
    guidance_scale: float,
    steps: int,
):
    """
    단일 이미지 증강 (미리보기)
    """
    if image_pil is None:
        raise ValueError("이미지를 업로드하세요.")

    yolo_boxes = parse_yolo_text(yolo_text)

    image_rgb = np.array(image_pil)
    img_w, img_h = image_pil.size

    bboxes_xyxy = yolo_to_xyxy(yolo_boxes, img_w, img_h)

    # SAM mask 생성
    pill_mask = masker.build_union_mask(
        image_rgb=image_rgb,
        bboxes_xyxy=bboxes_xyxy,
        dilate_px=cfg.mask_dilate_px,
    )

    # 프롬프트 결정 (비어 있으면 default)
    prompt = user_prompt.strip() or cfg.random_prompt()
    negative_prompt = user_negative_prompt.strip() or cfg.negative_prompt

    # Inpainting
    aug_image = inpainter.inpaint(
        image_pil=image_pil,
        pill_mask=pill_mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        steps=int(steps),
    )

    sam_mask_vis = Image.fromarray(pill_mask)
    inpaint_mask_vis = Image.fromarray(255 - pill_mask)

    return sam_mask_vis, inpaint_mask_vis, aug_image


def save_result(
    aug_image: Image.Image,
    yolo_text: str,
):
    """
    증강 이미지 + YOLO txt 저장
    """
    if aug_image is None:
        raise ValueError("저장할 이미지가 없습니다.")

    ts = int(time.time() * 1000)

    img_name = f"aug_{ts}.jpg"
    label_name = f"aug_{ts}.txt"

    img_path = cfg.out_image_dir / img_name
    label_path = cfg.out_label_dir / label_name

    aug_image.save(img_path)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write(yolo_text.strip() + "\n")

    return f"Saved:\n{img_path}\n{label_path}"


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# YOLO-only Pill Background Augmentation")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Original Image")

        with gr.Column():
            yolo_text = gr.Textbox(
                label="YOLO Labels (class cx cy w h)",
                placeholder=(
                    "0 0.5 0.5 0.2 0.3\n"
                    "0 0.3 0.4 0.1 0.15"
                ),
                lines=5,
            )

            prompt_text = gr.Textbox(
                label="Background Prompt (optional)",
                placeholder="비워두면 랜덤 프롬프트 사용",
                lines=2,
            )

            negative_prompt_text = gr.Textbox(
                label="Negative Prompt (optional)",
                value=cfg.negative_prompt,
                lines=2,
            )

            seed = gr.Number(value=cfg.seed, label="Seed")
            strength = gr.Slider(0.1, 1.0, value=cfg.strength, label="Strength")
            guidance = gr.Slider(1, 12, value=cfg.guidance_scale, label="Guidance")
            steps = gr.Slider(10, 50, value=cfg.num_inference_steps, label="Steps")

            run_btn = gr.Button("Run Augmentation")
            save_btn = gr.Button("Save Image + YOLO Label")

    with gr.Row():
        sam_mask_view = gr.Image(label="SAM Pill Mask", interactive=False)
        inpaint_mask_view = gr.Image(label="Inpaint Mask", interactive=False)
        output_image = gr.Image(label="Augmented Image", interactive=False)

    status = gr.Textbox(label="Status")

    run_btn.click(
        run_augmentation,
        inputs=[
            input_image,
            yolo_text,
            prompt_text,
            negative_prompt_text,
            seed,
            strength,
            guidance,
            steps,
        ],
        outputs=[
            sam_mask_view,
            inpaint_mask_view,
            output_image,
        ],
    )

    save_btn.click(
        save_result,
        inputs=[
            output_image,
            yolo_text,
        ],
        outputs=status,
    )

    # footer
    gr.Markdown(
        "<div style='text-align:center; font-size:12px; color:gray; margin-top:20px;'>"
        "made by hobi2k"
        "</div>"
    )


if __name__ == "__main__":
    demo.launch()
