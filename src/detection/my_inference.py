import csv
import json
import re
from pathlib import Path
from ultralytics import YOLO
from src.config import Config, init_logger

# 로거 초기화
logger = init_logger()


def extract_image_id(filename: str) -> int:
    """
    파일 이름에서 숫자(이미지 ID)를 추출하는 헬퍼 함수.

    예시:
      - "test_000123.png" -> 123
      - 숫자가 하나도 없으면 -1 리턴

    csv 제출용으로 image_id가 필요해서,
    파일명에 포함된 숫자를 추출.
    """
    # re.findall(r"\d+", filename)
    # 정규표현식 \d+ : 연속된 숫자 하나 이상
    # filename에서 숫자 덩어리들을 전부 리스트로 추출
    numbers = re.findall(r"\d+", filename)

    # numbers[0] : 첫 번째 숫자 덩어리만 사용
    # 예: ["123"] -> int("123") == 123
    # 만약 숫자가 전혀 없으면 numbers == [] 이므로 -1 리턴
    return int(numbers[0]) if numbers else -1


def load_yolo_to_coco():
    """
    YOLO 클래스 ID -> COCO category_id 매핑 정보를 로드하는 함수.

    - cocoparser의 remap_categories 단계에서 만든 category_mapping.json을 읽어서,
      {yolo_id(int): coco_id(int)} 딕셔너리를 반환한다.
      -> {0: 1, 1: 11, 2: 24} 이렇게 int key/value로 변환.
    """
    # remap에서 만들어 둔 category_mapping.json 경로
    mapping_path = Config.DATA_DIR / "raw/train_annotations/category_mapping.json"

    # 파일이 없으면 바로 에러를 던져서 사용자가 알 수 있게 함
    if not mapping_path.exists():
        raise FileNotFoundError(f"category_mapping.json 을 찾을 수 없습니다: {mapping_path}")

    # JSON 로드 (UTF-8 인코딩)
    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON의 key, value를 int로 캐스팅해서 {int: int} 형태의 dict로
    return {int(k): int(v) for k, v in data.items()}


def inference():
    """
    테스트 이미지 폴더에 대해 Inference를 수행하고,
    결과를 COCO 형식에 맞춘 CSV(submission.csv)로 저장.

    전체 흐름:
      1. YOLO 최종 모델 로드
      2. YOLO id -> COCO category_id 매핑 로드
      3. test_images 폴더에 대해 model.predict() 실행
      4. 각 detection 결과를 순회하며 CSV 한 줄씩 작성
    """

    # YOLO 모델 로드
    model = YOLO(str(Config.BASE_DIR / "outputs/models/hobi_yolov8_pill/weights/best.pt"))

    # COCO id 매핑 파일 로드
    yolo_to_coco = load_yolo_to_coco()

    # Inference 실행
    results = model.predict(
        source=str(Config.DATA_DIR / "raw/test_images"),
        imgsz=Config.IMAGE_SIZE,
        save=True,
        project=str(Config.BASE_DIR / "outputs/test"),
        name="pill_test_results",
    )

    # CSV 저장 경로 설정
    output_csv = Config.BASE_DIR / "outputs/test/pill_test_results/submission.csv"

    # 상위 디렉토리가 없으면 생성 (parents=True: 중간 경로도 전부 생성)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # annotation_id는 COCO에서 각 박스(검출 결과)에 부여하는 고유 ID 역할
    # 1부터 시작해서 detection 하나당 1씩 증가시키며 사용
    annotation_id = 1

    # CSV 파일 쓰기 모드로 열기
    # newline="" : 윈도우에서 빈 줄 끼어드는 문제 방지
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # CSV 헤더 작성
        writer.writerow([
            "annotation_id",
            "image_id",
            "category_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "score",
        ])

        # results: YOLO가 반환한 각 이미지별 결과 객체 리스트
        for result in results:
            # result.path: 처리한 이미지의 전체 경로 문자열
            img_path = Path(result.path)
            image_id = extract_image_id(img_path.name)  # 파일명에서 숫자 ID 추출

            # result.boxes: 해당 이미지에서 검출된 bbox들의 집합
            boxes = result.boxes

            # 박스가 하나도 없거나 None이면 스킵
            if boxes is None or len(boxes) == 0:
                continue

            # 각 검출 박스에 대해 CSV 한 줄씩 작성
            for box in boxes:
                # box.cls: 예측된 클래스 인덱스 텐서 (예: tensor(0.), tensor(1.), ...)
                yolo_cls = int(box.cls.item())  # YOLO 내부 클래스 ID (0 ~ num_classes-1)

                # yolo_to_coco에서 원래 COCO category_id 조회
                category_id = yolo_to_coco[yolo_cls]

                # box.conf: confidence score 텐서 (0.0 ~ 1.0)
                score = float(box.conf.item())

                # box.xyxy: [x1, y1, x2, y2] 형식의 텐서
                # squeeze(): 불필요한 차원 제거
                # tolist(): 파이썬 리스트로 변환
                x1, y1, x2, y2 = box.xyxy.squeeze().tolist()

                # COCO 형식의 bbox는 [x, y, w, h]
                # x, y는 좌상단 좌표
                # w, h는 폭/높이
                w = x2 - x1
                h = y2 - y1

                # 한 박스(annotation)에 해당하는 한 줄을 CSV에 기록
                writer.writerow([
                    annotation_id,
                    image_id,
                    category_id,
                    x1,
                    y1,
                    w,
                    h,
                    score,
                ])

                # 다음 박스를 위해 annotation_id 증가
                annotation_id += 1

    logger.info(f"Submission CSV 생성 완료: {output_csv}")


# 실행
if __name__ == "__main__":
    inference()
