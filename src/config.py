from pathlib import Path
import logging

class Config:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"

    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 10
    LEARNING_RATE = 0.001
    EPOCHS = 50
    
    SEED = 42
    
def init_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        encoding="utf-8",
    )
    return logging.getLogger("project_logger")