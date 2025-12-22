# HealthEat – AI 기반 헬스케어 알약 인식 프로젝트


## 팀 소개

헬스잇(HealthEat) AI 엔지니어링 팀은
사용자가 찍은 알약 사진을 분석하여 약 이름을 자동으로 인식하고,
약물 정보 및 주의사항을 제공하는 서비스를 개발합니다.

## 프로젝트 목표


| 항목    | 설명                              |
| ----- | ------------------------------- |
| Task  | 알약 객체 검출 + 약 이름 인식              |
| 입력    | 스마트폰 카메라 이미지                    |
| 출력    | 각 약의 클래스(이름) + 위치(Bounding Box) |
| 확장 목표 | 시각 장애인용 음성 안내, 알약 주의사항 안내          |


## 기술 스택


| 분류                | 사용 기술                                             |
| ----------------- | ------------------------------------------------- |
| Object Detection  | YOLOv8, YOLOv11, Faster R-CNN                                         |
| 협업 도구             | GitHub, Notion, Discord                             |


## 브랜치 전략


| Branch               | 용도                             |
| -------------------- | ------------------------------ |
| `main`               | 배포용 코드 |
| `dev` | 팀 개발용 통합 브랜치                   |
| `feature/개인 브랜치`          | 개인별 학습, 개발, 실험 브랜치                  |


## Git 규칙

- 직접 main, dev 브랜치에 push 금지

PR을 통해 코드가 main, dev로 들어갈 수 있으며 다음 규칙을 준수합니다.


| 규칙             | 설명                       |
| -------------- | ------------------------ |
| 1. PR 필수       | 모든 변경은 PR을 통해 main, dev에 병합   |
| 2. 코드 리뷰 필수    | 팀원 최소 1명 리뷰 필요      |
| 3. 커밋 메시지 규칙   | 한국어 or 영어 / 기능 중심 설명     |
| 4. small PR 지향 | 기능 단위로 빠른 리뷰 흐름 유지       |


## 커밋 메시지 컨벤션

- feat: 기능 추가
- fix: 버그 수정
- chore: 한 일 추가
- docs: 문서 작성/수정
- style: 코드 스타일 변경
- refactor: 기능 변화 없는 코드 개선
- test: 테스트 코드 추가/수정


예시:

feat: YOLOv8 데이터 로더 구현  
fix: OCR 전처리 크롭 좌표 오류 수정  
docs: 데이터셋 구성 설명 추가  

## 폴더 구조

personal_repository

```
healtheat_vision/
├── artifacts/               # 모델 가중치 및 학습 결과 시각화 (Confusion Matrix 등)
│   ├── runs/                # 객체 탐지 모델 저장소
│   ├── class_map.csv        # LLM 제어용 클래스 맵
├── configs/                 # YOLO 학습을 위한 YAML 설정 파일
├── data/                    # 데이터셋 (Git 제외 권장)
│   ├── aihub_downloads/     # AI Hub에서 다운로드한 원본 데이터
│   ├── train_images/        # 전처리 완료된 학습 이미지
│   ├── train_annotations/   # 전처리/수정된 JSON 어노테이션
│   ├── yolo/                # YOLO 포맷으로 변환된 최종 데이터 (images/labels)
│   └── test_images/         # 평가용 테스트 이미지
├── diffusion/               # 스테이블 디퓨전 + SAM 관련 소스 코드
├── docs/                    # 데이터 분석(EDA) 및 매칭 리스트 (Jupyter Notebooks)
├── scripts/                 # 환경 설정 및 데이터 다운로드 셸 스크립트
├── src/                     # 핵심 소스 코드
│   ├── preprocessing/       # 데이터 정제 및 수집 (collect_images.py 등)
│   ├── dataset/             # 포맷 변환 및 데이터셋 분할 (train/val split)
│   ├── train/               # YOLO 모델 학습 및 재개(resume)
│   ├── pred/                # 추론 및 제출 파일(CSV) 생성
│   └── utils/               # 공통 유틸리티 (로깅, 장치 설정, 경로 관리)
├── submissons/              # kaggle 제출용
├── tts/                     # tts 모델 스크립트
├── requirements.txt         # 설치 필요한 라이브러리 목록
└── README.md                # 프로젝트 가이드
```

## 모델 평가 기준


| 항목        | 지표                              |
| --------- | ------------------------------- |
| Detection | mAP, iou, Recall, Precision          |
| 의료 안전성    | 오검출률 최소화                        |


## 스테이블 디퓨전 이미지 증강

여기로 -> [README.md](/diffusion/README.md)


# 데모 앱

- 윈도우 유저 사용법
    - Download ZIP로 전체 개발 파일 다운로드 후 `app_augmentaition.bat`을 더블클릭하여 이미지 증강 앱 실행 가능
    - Download ZIP로 전체 개발 파일 다운로드 후 `app_speechT5.bat`을 더블클릭하여 이미지 증각 앱 실행 가능

- 파이썬에 익숙한 유저 사용법
    - git clone으로 리포지토리 복사
    - `uv venv --python 3.11` 명령어로 파이썬 가상환경 생성
    - `uv pip install -r requirements.txt` 명령어로 필요 라이브러리 설치
    - `uv run app_augmentation.py` 명령어로 이미지 증강 앱 실행 가능
    - `uv run app_speechT5.py` 명령어로 이미지 증강 앱 실행 가능

## 커뮤니케이션 규칙

- 회의록은 모두 notion에 기록
- Issue/PR 템플릿 활용