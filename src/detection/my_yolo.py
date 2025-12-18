from ultralytics import YOLO
from src.config import Config, init_logger

# 로거 초기화
logger = init_logger()

def train_yolov8():
    # YOLOv8 모델 로드 (n/s/m/l 중 성택)
    model = YOLO("yolov8n.pt")
    
    # 학습 루프
    model.train(
        data=str(Config.DATA_DIR / "yolo_dataset/dataset.yaml"),
        epochs=Config.EPOCHS,
        imgsz=Config.IMAGE_SIZE,
        batch=Config.BATCH_SIZE,
        lr0=Config.LEARNING_RATE,
        workers=8,
        device=None, # GPU 아니면 CPU
        project=str(Config.BASE_DIR / "outputs/models"), # 저장 폴더
        name="hobi_yolov8_pill_2u",
        pretrained=True, # 조기 종료        
    )
    
    logger.info(f"모델 파일 저장 완료.")

# 실행
if __name__ == "__main__":
    train_yolov8()