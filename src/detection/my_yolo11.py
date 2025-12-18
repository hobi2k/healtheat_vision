from ultralytics import YOLO
from src.config import Config, init_logger

# 로거 초기화
logger = init_logger()

def train_yolov11():
    model = YOLO("yolo11s.pt")

    model.train(
        data=str(Config.DATA_DIR / "yolo_dataset/dataset.yaml"),
        epochs=Config.EPOCHS,
        imgsz=Config.IMAGE_SIZE,
        batch=Config.BATCH_SIZE,
        lr0=Config.LEARNING_RATE,
        workers=8,
        device=None,
        project=str(Config.BASE_DIR / "outputs/models"),
        name="hobi_yolo11s",
        pretrained=True,

        # Early Stopping 추가
        patience=30,   # 30 epoch 동안 개선 없으면 종료
    )

    logger.info("YOLO11-S 모델 학습 및 저장 완료.")

if __name__ == "__main__":
    train_yolov11()
