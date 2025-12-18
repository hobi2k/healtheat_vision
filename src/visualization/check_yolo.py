import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse  # 터미널 명령어 처리 도구

# 프로젝트 루트 경로 설정
FILE_PATH = Path(__file__).resolve()
ROOT = FILE_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import paths

def draw_yolo_box(img_path, label_path):
    """이미지에 YOLO 박스 그리기 (기존 로직 동일)"""
    if not img_path.exists(): return None
    image = cv2.imread(str(img_path))
    if image is None: return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cls_id = int(parts[0])
                x_c, y_c, bw, bh = map(float, parts[1:])
                x1, y1 = int((x_c - bw/2)*w), int((y_c - bh/2)*h)
                x2, y2 = int((x_c + bw/2)*w), int((y_c + bh/2)*h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"ID:{cls_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def show_images(mode, target, count):
    """시각화 실행 함수"""
    img_dir = paths.YOLO_IMAGES_DIR / "train" 
    lbl_dir = paths.YOLO_LABELS_DIR / "train"
    all_images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    
    selected_imgs = []

    if mode == 'random':
        selected_imgs = random.sample(all_images, min(count, len(all_images)))
    
    elif mode == 'filename':
        selected_imgs = [p for p in all_images if target in p.name]
    
    elif mode == 'class':
        target_id = int(target)
        for img_p in all_images:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                with open(lbl_p, 'r') as f:
                    if any(int(line.split()[0]) == target_id for line in f if line.strip()):
                        selected_imgs.append(img_p)
            if len(selected_imgs) >= count: break

    # 결과 출력
    if not selected_imgs:
        print(f"❌ '{target}'에 해당하는 결과를 찾을 수 없습니다.")
        return

    n = len(selected_imgs[:count])
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(20, 4 * rows))
    for i, img_path in enumerate(selected_imgs[:count]):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        res = draw_yolo_box(img_path, lbl_path)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(res); plt.title(img_path.name, fontsize=8, pad=10); plt.axis('off')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    # 1. argparse 설정
    parser = argparse.ArgumentParser(description="YOLO 데이터셋 시각화 도구")
    parser.add_argument("--mode", type=str, choices=['random', 'filename', 'class'], help="시각화 모드")
    parser.add_argument("--target", type=str, help="검색 대상 (클래스번호 또는 파일명)")
    parser.add_argument("--count", type=int, help="표시할 이미지 개수")
    
    args = parser.parse_args()

    # 2. 파라미터가 없으면 input()으로 물어보기
    mode = args.mode
    if not mode:
        print("=== YOLO 시각화 도구 ===")
        mode = input("모드를 선택하세요 (random, filename, class): ").strip()

    target = args.target
    if not target and mode in ['filename', 'class']:
        target = input(f"검색할 {mode} 값을 입력하세요: ").strip()
    
    count = args.count
    if count is None:
        # 입력이 없으면 기본값 1, 있으면 숫자로 변환
        count_input = input("3) 표시할 이미지 개수를 입력하세요 (기본값 1): ").strip()
        count = int(count_input) if count_input.isdigit() else 1

    # 실행
    show_images(mode, target, count)