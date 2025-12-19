"""
배경 증강 파이프라인 전용 설정 파일

원칙:
- 이 파일에는 설정만 존재한다.
- 함수 로직, 모델 호출, 파일 입출력 금지.
- 다른 모듈은 이 Config만 import해서 사용한다.
"""
from dataclasses import dataclass
from pathlib import Path
from src.dataset import convert_cocojson_to_yolo
from src.utils import paths
import random


@dataclass
class AugConfig:
    # 경로 설정
    image_dir: Path = convert_cocojson_to_yolo.YOLO_IMAGES_DIR / "train"
    label_dir: Path = convert_cocojson_to_yolo.YOLO_LABELS_DIR / "train"

    out_image_dir: Path = paths.DATA_DIR / "aug/aug_images"
    out_label_dir: Path = paths.DATA_DIR / "aug/aug_labels"

    # SAM 설정
    sam_checkpoint: Path = paths.MODELS_DIR / "sam/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    sam_device: str = "cuda"

    # Stable Diffusion Inpainting 설정
    sd_inpaint_model: str = "runwayml/stable-diffusion-inpainting"
    sd_device: str = "cuda"

    # 증강 파라미터
    num_aug_per_image: int = 3
    seed: int = 42

    # 알약 경계 보호용 마스크 팽창(px)
    mask_dilate_px: int = 8

    # 프롬프트
    prompt_pool = (
        # 일반 테이블 계열
        "wooden table, plain background, medium contrast, daylight",
        "desk surface, neutral color, soft contrast, natural light",
        "tabletop background, minimal texture, low contrast, ambient light",

        # 촬영 환경 변화용
        "neutral background, low contrast, soft shadow",
        "plain surface, balanced contrast, diffuse light",
        "wooden table, low contrast, soft lighting",
    )

    negative_prompt: str = (
        "pill, tablet, capsule, medicine, text, letters, logo, watermark, "
        "engraving, imprint, deformed, distorted"
    )

    # Diffusion 파라미터
    # `guidance_scale` (classifier-free guidance scale):
    # - 프롬프트에 얼마나 강하게 따를지 결정합니다.
    # - 값이 클수록(예: 7-15) 모델이 프롬프트를 더 엄격히 반영하지만
    #   생성물의 다양성은 감소하고 아티팩트가 생길 수 있습니다.
    # - 값이 작을수록(예: 1-4) 창의성은 증가하지만 프롬프트 준수는 약해집니다.
    guidance_scale: float = 6.5

    # `num_inference_steps` (샘플링 스텝 수):
    # - 디퓨전 모델이 노이즈를 제거하며 이미지를 생성하는 단계 수입니다.
    # - 스텝 수가 많을수록(예: 50-100) 품질과 디테일이 개선되지만 추론 시간이 증가합니다.
    # - 너무 적으면(예: <10) 결과가 거칠거나 노이즈가 남을 수 있습니다.
    num_inference_steps: int = 30

    # `strength` (inpainting strength / denoising strength):
    # - inpainting 시 마스크 영역을 얼마나 새로 생성할지 제어합니다.
    # - 1.0에 가까우면 마스크 영역을 완전히 새로 생성하고,
    #   낮은 값(예: 0.2-0.5)은 원본 구조를 더 많이 보존합니다.
    # - 마스크가 있는 원본과의 혼합 정도를 조절할 때 사용합니다.
    strength: float = 0.85

    # 유틸
    def random_prompt(self) -> str:
        """프롬프트 풀에서 하나를 랜덤 선택"""
        return random.choice(self.prompt_pool)