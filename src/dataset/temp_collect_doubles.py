import shutil
from pathlib import Path

def prepare_double_labels():
    # 경로 설정
    img_double_dir = Path(r"E:\github\healtheat_vision\data\ft_mix_yolo_CLASHE_sharpen\doubles\images_double")
    src_label_dir = Path(r"E:\github\healtheat_vision\data\ft_mix_yolo_CLASHE_sharpen\labels\train")
    dst_label_dir = Path(r"E:\github\healtheat_vision\data\ft_mix_yolo_CLASHE_sharpen\doubles\labels_double")

    dst_label_dir.mkdir(parents=True, exist_ok=True)

    # 1. images_double 폴더 내의 모든 이미지 확인
    img_files = list(img_double_dir.glob("*.jpg")) + list(img_double_dir.glob("*.png")) + list(img_double_dir.glob("*.jpeg"))
    
    print(f"총 {len(img_files)}개의 이미지 발견. 파일명 변경 및 라벨 복사 시작...")

    count = 0
    for img_path in img_files:
        # 원래 파일명 (예: pill_01.jpg)
        old_name = img_path.name
        
        # 2. 이미지 파일명 변경 (앞에 double_ 추가)
        # 이미 double_이 붙어있을 수도 있으니 체크 후 변경
        if not old_name.startswith("double_"):
            new_img_name = f"double_{old_name}"
            new_img_path = img_double_dir / new_img_name
            img_path.rename(new_img_path)
            current_stem = img_path.stem # 원본 이름 (라벨 매칭용)
            final_img_name = new_img_name
        else:
            final_img_name = old_name
            current_stem = old_name.replace("double_", "") # 접두어 떼고 원본 이름 찾기

        # 3. 일치하는 라벨 찾아서 double_ 붙여서 복사
        # 원본 라벨 경로 (예: labels/train/pill_01.txt)
        src_label_path = src_label_dir / f"{current_stem}.txt"
        # 대상 라벨 경로 (예: doubles/labels_double/double_pill_01.txt)
        dst_label_path = dst_label_dir / f"double_{current_stem}.txt"

        if src_label_path.exists():
            shutil.copy2(src_label_path, dst_label_path)
            count += 1
        else:
            print(f"경고: 라벨 파일을 찾을 수 없음 -> {src_label_path}")

    print(f"완료! {count}개의 라벨 파일이 {dst_label_dir}로 복사되었습니다.")

if __name__ == "__main__":
    prepare_double_labels()