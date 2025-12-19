# Stable Diffusion + SAM 기반 알약 이미지 증강 파이프라인
*(YOLO Dataset + Gradio WebUI)*

## 개요

본 파이프라인은 YOLO 기반 알약 객체 인식 데이터셋의 품질과 다양성을 향상시키기 위해 설계되었다.
핵심 목표는 다음과 같다.

- 알약 객체는 절대 변경하지 않고
- 배경만 생성적으로 다양화하여
- YOLO 학습 시 일반화 성능을 향상시키는 것

이를 위해 다음 기술을 결합한다.

- SAM (Segment Anything Model)
        - YOLO bbox를 기반으로 알약 픽셀을 정밀 분리

- Stable Diffusion Inpainting
        - 알약은 보호하고 배경만 생성

- YOLO annotation(txt) 그대로 유지
        - bbox / class / 좌표계 불변

## 스테이블 디퓨전 기반 증강 이점

### 기존 YOLO 증강의 한계

- Color jitter, blur, noise 중심
- 실제 촬영 환경 변화(배경 재질, 그림자, 조명) 반영 한계
- 배경 다양성 부족 -> 과적합 위험

### 본 파이프라인의 차별점

- 배경 자체를 생성
- YOLO bbox는 그대로 유지
- 실제 촬영 환경 변화에 가까운 증강

## 전체 파이프라인 구조 (YOLO 기준)

```
원본 이미지 + YOLO label(txt)
        │
        ▼
[YOLO bbox -> pixel 좌표 변환]
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
증강 이미지 + 기존 YOLO txt 그대로 재사용
```

## 핵심 구성 요소 설명

### YOLO bbox의 SAM 마스킹

**목적**

- YOLO bbox는 거칠기 때문에
- 실제 픽셀 단위 보호를 위해 SAM segmentation 필요

**흐름**

- YOLO txt (class cx cy w h, normalized)
- 이미지 크기 기준 pixel 좌표로 변환
- SAM에 xyxy bbox로 전달
- 알약 union mask 생성

**마스크 정의 규칙**

- 255 -> 알약 영역 (절대 보호)
- 0 -> 배경 영역 (변경 대상)

**SAM을 사용하는 이유**

- 얇은 가장자리, 타원형 알약에 강함
- bbox 기반이므로 YOLO와 자연스럽게 연결됨
- 수작업 segmentation 불필요

### Stable Diffusion Inpainting 구조

Stable Diffusion Inpainting은 마스크 해석 방식이 반대이므로 주의가 필요하다.

- 흰색(255): 새로 그릴 영역
- 검정색(0): 유지할 영역

따라서 SAM 마스크는 반드시 반전해야 한다.

```python
# SAM 결과: 255=알약, 0=배경
# Diffusion 입력용: 255=배경, 0=알약
bg_mask = Image.fromarray(255 - pill_mask, mode="L")
```

이 한 줄이 알약 손상 여부, 각인 보존, 증강 품질을 결정한다.

### YOLO 어노테이션 처리 전략

본 파이프라인에서는 YOLO annotation을 새로 생성하지 않는다.

**이유**

- 객체 위치 불변
- bbox 크기, 좌표 불변
- class 불변
- 배경만 변경

따라서 처리 방식은 다음과 같다.

- 증강 이미지 저장
- YOLO txt 파일을 그대로 복사
- 파일명만 이미지에 맞게 변경

```
images/
 ├── pill_001.jpg
 ├── pill_001_aug_000.jpg
 ├── pill_001_aug_001.jpg

labels/
 ├── pill_001.txt
 ├── pill_001_aug_000.txt
 ├── pill_001_aug_001.txt
```

YOLO 학습 및 추론 파이프라인에 즉시 투입 가능

## WebUI (Gradio) 기반 실험 설계

본 파이프라인의 중요한 특징은
배치 증강 이전에 “사람이 직접 품질을 확인”할 수 있다는 점이다.

이를 위해 Gradio WebUI를 제공한다.

### Gradio WebUI의 역할

Gradio는 단순 데모가 아니라 실험, 검증 도구다.

제공 기능

- 단일 이미지 업로드
- YOLO label(txt) 직접 입력
- SAM mask 시각화
- Inpainting mask 시각화
- 증강 결과 실시간 확인
- 증강 이미지 + YOLO txt 저장

### WebUI에서 가능한 실험

- 프롬프트에 따른 배경 질감 변화 비교
- shadow / contrast / material 영향 확인
- strength / guidance / steps 파라미터 튜닝
- 실패 케이스 사전 제거
- 프롬프트 전략
- 사용자가 입력하면 해당 프롬프트 사용
- 입력하지 않으면 cfg.random_prompt() 자동 적용

통제 실험과 랜덤 실험을 하나의 UI에서 병행

### WebUI와 배치 파이프라인의 관계


| 구분           | 역할                |
| ------------ | ----------------- |
| Gradio WebUI | 실험 / 검증 / 파라미터 튜닝 |
| 배치 스크립트      | 대량 증강 / 데이터셋 생성   |


즉,

```
Gradio로 검증
        ↓
정책 확정
        ↓
배치 증강
```

이라는 흐름을 전제로 설계되었다.

6. 실전 적용 팁 (YOLO 기준)


| 항목             | 권장값           |
| -------------- | ------------- |
| 증강 비율          | 원본 1 : 증강 3~5 |
| strength       | 0.55 ~ 0.75   |
| guidance_scale | 6.0 ~ 8.5     |
| steps          | 25 ~ 40       |


주의 사항

- strength > 0.8
        - 그림자 번짐, 배경 침범 위험

- guidance_scale 과다
        - 배경 인공적 느낌 증가

## 요약

YOLO bbox + SAM으로 객체를 보호하고
Stable Diffusion으로 배경만 생성하는
“검증 가능한 생성 증강 파이프라인”

- YOLO 데이터셋 품질 향상
- 과적합 완화
- 실제 촬영 환경 일반화

를 동시에 달성하기 위한 설계다.