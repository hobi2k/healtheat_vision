from src import config, init_logger
from src.utils import data_matching_checker

# 로거 초기화
logger = init_logger()

def match_data(image_dir: str, label_dir: str) -> None:
    unmatched_images, unmatched_labels = data_matching_checker.check_data_matching(
        image_dir=image_dir,
        label_dir=label_dir
    )
    
    if unmatched_images or unmatched_labels:
        logger.warning("데이터 매칭에 문제가 있습니다.")
    else:
        logger.info("모든 이미지와 라벨이 일치합니다.")