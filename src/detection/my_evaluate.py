from ultralytics import YOLO
from src.config import Config, init_logger

# 로거 초기화
logger = init_logger()

def evaluate_yolov8():
    model = YOLO(str(Config.BASE_DIR / "outputs/models/hobi_yolo11s/weights/best.pt"))
    
    metrics = model.val(
        data=str(Config.DATA_DIR / "yolo_dataset/dataset.yaml"),
        imgsz=Config.IMAGE_SIZE,
        batch=Config.BATCH_SIZE,
    )
    
    logger.info("\n평가 결과\n")
    logger.info(metrics) # 모든 metric 출력
    # 박스 메트릭
    logger.info(f"mAP50: {metrics.box.map50:.4f}")
    logger.info(f"mAP50-95: {metrics.box.map:.4f}")
    logger.info(f"precision (per class): {metrics.box.p}")
    logger.info(f"recall (per class): {metrics.box.r}")


if __name__ == "__main__":
    evaluate_yolov8()