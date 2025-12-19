# Stable Diffusion + SAM 기반 알약 이미지 증강 파이프라인

## 개요

본 파이프라인은 의약품(알약) 객체 인식용 데이터셋 증강을 목적으로 설계되었다.
핵심 아이디어는 다음과 같다.

- SAM (Segment Anything Model)로 알약 영역을 정밀 분리

- Stable Diffusion Inpainting 을 사용해
    - 알약은 보존
    - 배경만 생성적으로 변경

- COCO 어노테이션 유지 -> YOLO / Detectron / RT-DETR 계열 학습에 바로 사용 가능

이 방식은 단순한 색상/노이즈 증강 대비 다음 장점을 가진다.

1. 배경 다양성 극대화
2. 실제 촬영 환경 변화(책상, 종이, 패브릭, 그림자 등) 시뮬레이션
3. 객체 형태, 텍스트, 각인 정보 보존

## 전체 파이프라인 구조

```
원본 이미지 + COCO annotation
        │
        ▼
[SAM] 알약 마스크 생성 (binary mask)
        │
        ▼
[Mask Inversion]
(알약=보호 / 배경=인페인트 대상)
        │
        ▼
[Stable Diffusion Inpainting]
(prompt 기반 배경 생성)
        │
        ▼
증강 이미지 + 기존 COCO annotation 재사용
```

## 핵심 구성 요소 설명

### SAM 기반 알약 마스킹

목표

- 알약 픽셀을 최대한 정확하게 보호
- 배경만 diffusion 대상이 되도록 mask 생성

마스크 정의 규칙

- 255: 알약 영역 (보호)
- 0: 배경 영역 (변경 대상)


SAM을 사용하는 이유

- 얇은 가장자리, 타원형 알약에 강함
- bbox 기반 -> mask 자동 확장 가능

### Stable Diffusion Inpainting 구조

Stable Diffusion Inpainting의 핵심 동작은 다음과 같다.

- 흰색(255): 새로 그릴 영역
- 검정색(0): 유지할 영역
- 따라서 SAM 마스크를 반드시 반전해야 한다.

```python
# SAM 결과: 255=알약, 0=배경
# Diffusion 입력용: 255=배경, 0=알약
bg_mask = Image.fromarray(255 - pill_mask, mode="L")
```

이 한 줄이 전체 증강 품질을 좌우한다.

### COCO 어노테이션 처리 전략

이 파이프라인의 강점 중 하나는

- Bounding box / category ID를 수정할 필요가 없다는 것

이유

- 객체 위치, 형태 불변
- 배경만 변경

따라서:

- image_id만 새로 부여
- annotation은 그대로 복사

```
images/
 ├── img_001.jpg
 ├── img_001_aug_01.jpg
 ├── img_001_aug_02.jpg

annotations.json
 ├── bbox 동일
 ├── category_id 동일
```

## 실전 적용 팁


| 항목             | 권장값           |
| -------------- | ------------- |
| 증강 비율          | 원본 1 : 증강 3~5 |
| strength       | 0.55 ~ 0.75   |
| guidance_scale | 6.0 ~ 8.5     |
| steps          | 25 ~ 40       |


주의:

- strength > 0.8 -> 그림자 번짐 위험
- guidance_scale 과다 -> 배경 인공 느낌