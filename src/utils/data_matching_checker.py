from pathlib import Path
import json
from src.config import Config

def check_data_matching(image_dir: Path, label_dir: Path) -> tuple:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    image_files = sorted([f.stem for f in image_dir.iterdir() if f.suffix.lower() == ".png"])
    label_files = sorted([f.stem for f in label_dir.iterdir() if f.suffix.lower() == ".json"])
                         
    unmatched_images = set(image_files) - set(label_files)
    unmatched_labels = set(label_files) - set(image_files)
    
    if unmatched_images:
        print("라벨 없는 이미지:")
        for img in unmatched_images:
            print(f"- {img}.png\n")
    
    if unmatched_labels:
        print("이미지 없는 라벨:")
        for lbl in unmatched_labels:
            print(f"- {lbl}.json\n")
            
    if not unmatched_images and not unmatched_labels:
        print("모든 이미지와 라벨이 일치합니다.")
        
    return (unmatched_images, unmatched_labels)

# 실행
if __name__ == "__main__":
    check_data_matching(
        image_dir=Config.DATA_DIR / "aligned/train_images",
        label_dir=Config.DATA_DIR / "aligned/train_annotations"
    )