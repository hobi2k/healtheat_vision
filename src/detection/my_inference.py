from ultralytics import YOLO
from src.config import Config, init_logger

logger = init_logger()

def inference():
    model = YOLO(str(Config.BASE_DIR / "outputs/models/hobi_yolov8_pill/weights/best.pt"))
    
    results = model.predict(
        source=str(Config.DATA_DIR / "raw/test_images"),
        imgsz=Config.IMAGE_SIZE,
        save=True,
        project=str(Config.BASE_DIR / "outputs/test"),
        name="pill_test_results"
    )

if __name__ == "__main__":
    inference()