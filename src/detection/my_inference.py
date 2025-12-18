import csv
import json
import re
from pathlib import Path
from ultralytics import YOLO

from src.config import Config, init_logger

logger = init_logger()


def extract_image_id(filename: str) -> int:
    # 파일 이름에서 숫자를 추출하여 image_id로 사용
    nums = re.findall(r"\d+", filename)
    return int(nums[0]) if nums else -1


def load_yolo_to_coco():
    """
    category_mapping.json을 yolo_id(int) -> {coco_id:int, name:str} 형태로 로드.
    """
    mapping_path = Config.DATA_DIR / "unified_dataset_v2/annotations/category_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"category_mapping.json 없음: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # JSON key("0","1","2") -> int key 변환
    mapping = {int(k): v for k, v in raw.items()}

    logger.info(f"[INFO] Loaded category_mapping.json → {len(mapping)} classes")
    return mapping


def inference(include_class_name=False):
    """
    YOLO inference 수행 후 submission CSV 생성.
    category_mapping.json을 기반으로 coco_id 및 class_name을 정확히 매핑한다.
    """
    # YOLO 모델 로드
    model_path = Config.BASE_DIR / "outputs/models/hobi_yolo11s/weights/best.pt"
    model = YOLO(str(model_path))

    # category mapping 로드
    yolo_to_coco = load_yolo_to_coco()

    max_yolo_cls = max(yolo_to_coco.keys())
    logger.info(f"[INFO] YOLO class range: 0 ~ {max_yolo_cls}")

    # YOLO inference 실행
    results = model.predict(
        source=str(Config.DATA_DIR / "raw/test_images"),
        imgsz=Config.IMAGE_SIZE,
        save=True,
        project=str(Config.BASE_DIR / "outputs/test"),
        name="pill_test_results",
    )

    # CSV 저장 준비
    output_csv = (
        Config.BASE_DIR
        / "outputs/test/pill_test_results/submission.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header_with_name = [
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "score", "class_name"
    ]
    header_no_name = header_with_name[:-1]

    annotation_id = 1

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # CSV 헤더
        if include_class_name:
            writer.writerow(header_with_name)
        else:
            writer.writerow(header_no_name)


        # inference 결과 순회
        for result in results:
            img_path = Path(result.path)
            image_id = extract_image_id(img_path.name)

            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                yolo_cls = int(box.cls.item())

                # YOLO class가 매핑 범위 밖이라면 스킵
                if yolo_cls not in yolo_to_coco:
                    logger.warning(f"[WARN] 없는 클래스: {yolo_cls}")
                    continue

                coco_id = int(yolo_to_coco[yolo_cls]["coco_id"])
                class_name = yolo_to_coco[yolo_cls]["name"]

                score = float(box.conf.item())

                x1, y1, x2, y2 = map(float, box.xyxy.squeeze().tolist())
                w = x2 - x1
                h = y2 - y1

                row = [
                    annotation_id, image_id, coco_id,
                    round(x1, 2), round(y1, 2),
                    round(w, 2), round(h, 2),
                    round(score, 5)
                ]

                if include_class_name:
                    row.append(class_name)

                writer.writerow(row)
                annotation_id += 1

    logger.info(f"[DONE] CSV 생성 완료: {output_csv}")


if __name__ == "__main__":
    inference(include_class_name=False)
