"""
배경 증강 파이프라인 실행 스크립트

이 파일에서 하는 것
- 전체 흐름 제어
- I/O (이미지 로드 / 저장)
- COCO image_id / annotation_id 관리
"""

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 프로젝트 모듈 import
from .aug_config import AugConfig
from .coco_io import (
    load_coco,
    save_coco,
    group_annotations_by_image,
    get_next_ids,
)
from .sam_masker import PillMasker
from .inpainting import BackgroundInpainter
from .background_aug import BackgroundAugmentor
from .coco_per_image import build_per_image_coco


# 유틸 함수
def coco_bbox_to_xyxy(bbox):
    """
    COCO bbox [x, y, w, h] -> xyxy [x1, y1, x2, y2]

    SAM은 bbox를 xyxy 형식으로 받기 때문에 변환이 필요하다.
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def load_image_rgb(image_path):
    """
    이미지 파일을 읽어서 RGB numpy array로 반환

    OpenCV는 기본이 BGR이므로 RGB로 변환한다.
    """
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없음: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# 메인 실행 로직
def main():
    # 1. 설정 로드
    cfg = AugConfig()

    cfg.out_image_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_coco_per_image_dir.mkdir(parents=True, exist_ok=True)

    # 2. COCO 로드 및 사전 처리
    coco = load_coco(cfg.coco_json)

    images = coco["images"]
    annotations = coco["annotations"]

    # image_id -> annotation 리스트
    ann_by_image = group_annotations_by_image(annotations)

    # 새로운 ID를 만들기 위한 시작점
    next_image_id, next_ann_id = get_next_ids(coco)

    # 3. 모델 / 파이프라인 초기화
    # SAM: 알약 영역 보호용
    masker = PillMasker(
        checkpoint=str(cfg.sam_checkpoint),
        model_type=cfg.sam_model_type,
        device=cfg.sam_device,
    )

    # Stable Diffusion Inpainting: 배경 생성용
    inpainter = BackgroundInpainter(
        model_id=cfg.sd_inpaint_model,
        device=cfg.sd_device,
    )

    # 정책 레이어: "알약 고정 + 배경만 증강"
    augmentor = BackgroundAugmentor(
        masker=masker,
        inpainter=inpainter,
        cfg=cfg,
    )

    # 4. 증강 결과를 담을 컨테이너
    new_images = []
    new_annotations = []

    # 5. 이미지 단위 증강 루프
    for img_info in tqdm(images, desc="Background augmentation"):
        image_id = img_info["id"]
        file_name = img_info["file_name"]

        image_path = cfg.image_dir / file_name

        # 해당 이미지에 annotation이 없는 경우는 스킵
        if image_id not in ann_by_image:
            continue

        # 5-1. 이미지 로드
        image_rgb = load_image_rgb(image_path)
        image_pil = Image.fromarray(image_rgb)

        # 5-2. bbox -> xyxy 변환
        bboxes_xyxy = [
            coco_bbox_to_xyxy(ann["bbox"])
            for ann in ann_by_image[image_id]
        ]

        # 5-3. 증강 N회 반복
        for k in range(cfg.num_aug_per_image):
            next_image_id += 1
            aug_image_id = next_image_id

            # 결과 파일명 생성
            stem = image_path.stem
            ext = image_path.suffix
            out_name = f"{stem}_aug_{k:02d}{ext}"
            out_path = cfg.out_image_dir / out_name

            # seed는 image_id 기반으로 고정 -> 재현성 확보
            seed = cfg.seed + aug_image_id * 100 + k

            # 핵심: 배경 증강 수행
            aug_image = augmentor.augment(
                image_rgb=image_rgb,
                image_pil=image_pil,
                bboxes_xyxy=bboxes_xyxy,
                seed=seed,
            )
            
            if aug_image.size != image_pil.size:
                raise RuntimeError(
                    f"[FATAL] Augmented image size mismatch: "
                    f"orig={image_pil.size}, aug={aug_image.size}"
                )

            # 결과 이미지 저장
            aug_image.save(out_path)

            # COCO image entry 복제
            new_img_info = dict(img_info)
            new_img_info["id"] = aug_image_id
            new_img_info["file_name"] = str(
                cfg.out_image_dir.name + "/" + out_name
            )

            new_images.append(new_img_info)

            # annotation 복제 (bbox / class 그대로)
            image_annotations = []  # per-image 전용

            for ann in ann_by_image[image_id]:
                next_ann_id += 1
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = aug_image_id

                new_annotations.append(new_ann) 
                image_annotations.append(new_ann)
                
            # 여기서 per-image COCO 생성
            per_image_coco = build_per_image_coco(
                image_info=new_img_info,
                annotations=image_annotations,
                categories=coco["categories"],
            )

            per_image_json_path = (
                cfg.out_coco_per_image_dir / f"{out_path.stem}.json"
            )

            save_coco(per_image_coco, per_image_json_path)

    # 6. COCO 병합 및 저장
    out_coco = dict(coco)
    out_coco["images"] = coco["images"] + new_images
    out_coco["annotations"] = coco["annotations"] + new_annotations

    save_coco(out_coco, cfg.out_coco_json)

    print(f"증강 이미지 수: {len(new_images)}")
    print(f"증강 어노테이션 수: {len(new_annotations)}")
    print(f"저장 위치: {cfg.out_coco_json}")


if __name__ == "__main__":
    main()