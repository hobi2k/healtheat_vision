"""
배경 증강 파이프라인 실행 스크립트 (YOLO-only)

이 파일에서 하는 것
- 전체 흐름 제어
- I/O (이미지 로드 / 저장)
- YOLO label(txt) 읽기 -> SAM bbox 입력 생성
- 증강 이미지 저장 + YOLO txt 복사(파일명만 증강 이미지에 맞춤)

주의:
- annotation(라벨)은 "생성/수정"하지 않는다.
- 이미지 기하(resize/crop/pad)가 변하지 않는다는 전제에서만 안전하다.
"""

import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import shutil
from typing import List, Tuple

# 프로젝트 모듈 import
from .aug_config import AugConfig
from .sam_masker import PillMasker
from .inpainting import BackgroundInpainter
from .background_aug import BackgroundAugmentor
from .model_hub import prepare_models


# 유틸 함수
def load_image_rgb(image_path: Path):
    """
    이미지 파일을 읽어서 RGB numpy array로 반환

    OpenCV는 기본이 BGR이므로 RGB로 변환한다.
    """
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없음: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def yolo_line_to_xyxy(
    line: str,
    img_w: int,
    img_h: int,
) -> List[float]:
    """
    YOLO 라벨 한 줄을 픽셀 단위 xyxy로 변환한다.

    입력 라벨 포맷:
        class cx cy w h   (모두 0~1 정규화)

    출력:
        [x1, y1, x2, y2]  (픽셀 좌표)
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"YOLO 라벨 형식 오류(토큰 수 != 5): {line!r}")

    _, cx, cy, bw, bh = parts
    cx = float(cx)
    cy = float(cy)
    bw = float(bw)
    bh = float(bh)

    # 정규화 -> 픽셀
    box_w = bw * img_w
    box_h = bh * img_h
    x_c = cx * img_w
    y_c = cy * img_h

    x1 = x_c - box_w / 2
    y1 = y_c - box_h / 2
    x2 = x_c + box_w / 2
    y2 = y_c + box_h / 2

    # 안전하게 클램프(경계 밖으로 살짝 나가는 케이스 방어)
    x1 = max(0.0, min(x1, img_w - 1.0))
    y1 = max(0.0, min(y1, img_h - 1.0))
    x2 = max(0.0, min(x2, img_w - 1.0))
    y2 = max(0.0, min(y2, img_h - 1.0))

    return [x1, y1, x2, y2]


def load_yolo_bboxes_xyxy(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> List[List[float]]:
    """
    YOLO txt를 읽어서 SAM 입력용 bbox(xyxy) 리스트로 변환한다.
    """
    if not label_path.exists():
        return []

    lines = []
    with label_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                lines.append(raw)

    bboxes = [yolo_line_to_xyxy(line, img_w, img_h) for line in lines]
    return bboxes


def is_image_file(p: Path) -> bool:
    """
    데이터셋 이미지 확장자 필터.
    필요하면 여기에 .bmp, .tif 등을 추가.
    """
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


# 메인 실행 로직
def main():
    # 1. 설정 로드
    cfg = AugConfig()

    cfg.out_image_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_label_dir.mkdir(parents=True, exist_ok=True)

    # 2. 모델 준비(다운로드/캐시 등)
    prepare_models(cfg)

    # 3. 모델 / 파이프라인 초기화
    masker = PillMasker(
        checkpoint=str(cfg.sam_checkpoint),
        model_type=cfg.sam_model_type,
        device=cfg.sam_device,
    )

    inpainter = BackgroundInpainter(
        model_id=cfg.sd_inpaint_model,
        device=cfg.sd_device,
    )

    augmentor = BackgroundAugmentor(
        masker=masker,
        inpainter=inpainter,
        cfg=cfg,
    )

    # 4. 입력 이미지 목록
    image_paths = sorted([p for p in cfg.image_dir.iterdir() if p.is_file() and is_image_file(p)])
    if not image_paths:
        raise RuntimeError(f"입력 이미지가 없습니다: {cfg.image_dir}")

    total_aug = 0
    skipped_no_label = 0

    # 5. 이미지 단위 증강 루프
    for image_path in tqdm(image_paths, desc="Background augmentation (YOLO-only)"):
        label_path = cfg.label_dir / f"{image_path.stem}.txt"

        # YOLO label이 없는 이미지는 학습 기준에서도 보통 제외하므로 스킵
        if not label_path.exists():
            skipped_no_label += 1
            continue

        # 5-1. 이미지 로드
        image_rgb = load_image_rgb(image_path)
        image_pil = Image.fromarray(image_rgb)
        img_w, img_h = image_pil.size

        # 5-2. YOLO txt -> bbox(xyxy) 변환 (SAM 입력)
        bboxes_xyxy = load_yolo_bboxes_xyxy(label_path, img_w=img_w, img_h=img_h)

        if not bboxes_xyxy:
            # 라벨 파일은 있는데 내용이 비어있는 케이스(객체 없음)
            # 이 경우도 증강은 가능하지만, 네 파이프라인 의도상 스킵하는 게 일반적이라 스킵 처리.
            skipped_no_label += 1
            continue

        # 5-3. 증강 N회 반복
        for k in range(cfg.num_aug_per_image):
            # 결과 파일명 생성 (이미지)
            out_image_name = f"{image_path.stem}_aug_{k:02d}{image_path.suffix}"
            out_image_path = cfg.out_image_dir / out_image_name

            # 결과 파일명 생성 (라벨) - 이미지 stem과 반드시 일치해야 함
            out_label_name = f"{Path(out_image_name).stem}.txt"
            out_label_path = cfg.out_label_dir / out_label_name

            # seed 고정(재현성)
            seed = cfg.seed + (hash(image_path.stem) % 100000) * 10 + k

            # 핵심: 배경 증강 수행
            aug_image = augmentor.augment(
                image_rgb=image_rgb,
                image_pil=image_pil,
                bboxes_xyxy=bboxes_xyxy,
                seed=seed,
            )

            if aug_image.size != image_pil.size:
                raise RuntimeError(
                    f"[FATAL] Augmented image size mismatch: orig={image_pil.size}, aug={aug_image.size}"
                )

            # 결과 이미지 저장
            aug_image.save(out_image_path)

            # 핵심: YOLO label txt 그대로 복사 (파일명만 맞춤)
            shutil.copy2(label_path, out_label_path)

            total_aug += 1

    print(f"완료: 증강 이미지 생성 수 = {total_aug}")
    print(f"스킵(라벨 없음/비어있음) 이미지 수 = {skipped_no_label}")


if __name__ == "__main__":
    main()