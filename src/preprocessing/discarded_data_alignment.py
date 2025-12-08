# # 필요 없는 코드(annotation 평탄화용)
# from pathlib import Path
# from src.config import Config, init_logger

# # 로거 초기화
# logger = init_logger()

# # 원본 데이터 경로
# raw_data_path = Config.DATA_DIR / "raw"

# # 훈련 및 테스트 이미지 경로
# train_images_path = raw_data_path / "train_images"
# test_images_path = raw_data_path / "test_images"

# # 훈련 및 테스트 어노테이션 경로
# train_annotations_path = raw_data_path / "train_annotations"

# # 정렬된 데이터 저장 경로
# aligned_data_path = Config.DATA_DIR / "aligned"

# # 이미지 정렬 폴더 생성
# alined_train_images_path = aligned_data_path / "train_images"
# alined_test_images_path = aligned_data_path / "test_images"

# # 어노테이션 정렬 폴더 생성
# alined_train_annotations_path = aligned_data_path / "train_annotations"
# alined_test_images_path = aligned_data_path / "test_images_annotations"

# def data_alignment(input_path: Path, output_path: Path):
#     """
#     데이터 정렬 함수
#     """
#     if not output_path.exists():
#         output_path.mkdir(parents=True, exist_ok=True)
    
#     # 입력 경로의 모든 파일을 출력 경로로 복사
#     for item in input_path.iterdir():
#         if item.is_file():
#             destination = output_path / item.name
#             # rename()을 사용하여 파일 이동
#             item.rename(destination)
#             logger.info(f"{item} 를 {destination} 로 이동 완료.")
        
#         # 디렉토리인 경우 재귀적으로 처리    
#         else:
#             data_alignment(item, output_path)
            
# # 훈련 데이터 정렬
# data_alignment(train_images_path, alined_train_images_path)
# logger.info("훈련 이미지 정렬 완료.")
# data_alignment(train_annotations_path, alined_train_annotations_path)
# logger.info("훈련 어노테이션 정렬 완료.")

# # 테스트 데이터 정렬
# data_alignment(test_images_path, alined_test_images_path)
# logger.info("테스트 이미지 정렬 완료.")

