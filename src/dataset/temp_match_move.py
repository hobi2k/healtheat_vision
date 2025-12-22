import shutil
from pathlib import Path

def sync_labels():
    # 1. 절대 경로 설정
    IMAGE_DIR = Path(r"E:\github\healtheat_vision\data\ft_mix_yolo\match\images")
    SOURCE_LABEL_DIR = Path(r"E:\github\healtheat_vision\data\additional_yolo\labels\train")
    TARGET_LABEL_DIR = Path(r"E:\github\healtheat_vision\data\ft_mix_yolo\match\labels")

    # 2. 목적지 라벨 폴더가 없으면 생성
    TARGET_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 이미지 폴더 내의 파일들 순회
    # 확장자는 소문자, 대문자 모두 대응 가능하도록 처리
    extensions = ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG")
    image_files = []
    for ext in extensions:
        image_files.extend(list(IMAGE_DIR.glob(ext)))

    print(f"--- 총 {len(image_files)}개의 이미지를 감지했습니다. ---")

    success_count = 0
    fail_count = 0

    for img_path in image_files:
        # 이미지 파일명(확장자 제외)과 동일한 .txt 파일명 생성
        label_filename = img_path.stem + ".txt"
        source_label_path = SOURCE_LABEL_DIR / label_filename
        target_label_path = TARGET_LABEL_DIR / label_filename

        # 4. 소스 라벨 폴더에 파일이 있는지 확인 후 복사
        if source_label_path.exists():
            shutil.copy2(source_label_path, target_label_path)
            success_count += 1
        else:
            print(f"❌ 라벨 없음: {img_path.name} (예상 경로: {source_label_path})")
            fail_count += 1

    print("-" * 50)
    print(f"✅ 작업 완료!")
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    print(f"결과 저장 위치: {TARGET_LABEL_DIR}")

if __name__ == "__main__":
    sync_labels()