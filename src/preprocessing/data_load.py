import zipfile
from pathlib import Path
from src.config import Config, init_logger

# 압축 파일 경로
zip_path = Config.DATA_DIR / "ai06-level1-project.zip"

# 압축 해제 경로
extracted_path = Config.DATA_DIR / "raw"

# 로거 초기화
logger = init_logger()

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)
    
logger.info(f"{zip_path} 파일을 {extracted_path} 경로에 압축 해제 완료.")

