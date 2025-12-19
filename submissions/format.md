# Submission Format

## CSV Columns
- annotation_id (int): 1부터 시작하는 고유 박스 ID
- image_id (int): 테스트 이미지의 ID (파일명에서 숫자 추출 규칙 사용)
- category_id (int): 원본 클래스 ID (orig_id)  
  - 모델 출력 cls(yolo_id)를 `artifacts/class_map.csv`로 역변환해서 사용
- bbox_x, bbox_y, bbox_w, bbox_h (float): pixel 좌표, top-left 기준 (x,y,w,h)
- score (float): confidence score (0~1)

## Class Mapping
- 파일: `artifacts/class_map.csv`
- 컬럼: orig_id, yolo_id
- 변환:
  - 예측 cls = yolo_id
  - 제출 category_id = orig_id

## How to Run
```bash
# repo root(healtheat_vision)에서 실행
python -m src.pred.predict_and_submit