'''
흐릿한 알약 이미지에 CLAHE와 Sharpening을 적용하여, 
FT(Fine-Tuning) 학습에 최적화된 데이터셋을 구성하는 코드입니다.
경로 혼선을 방지하기 위해 절대경로로 작성되었으므로, 실행 전 사용자 환경에 맞춰 경로를 수정해 주세요.
'''
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def apply_enhancement(img_path, dst_path):
    """CLAHE와 Sharpening을 적용하여 이미지를 저장합니다."""
    # 한글 경로 대응 읽기
    img_array = np.fromfile(str(img_path), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return False

    # 1. CLAHE (Contrast 최적화)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. Sharpening (경계선 강화)
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]])
    img_processed = cv2.filter2D(img_processed, -1, kernel)

    # 3. 한글 경로 대응 저장
    extension = img_path.suffix
    _, im_buf_arr = cv2.imencode(extension, img_processed)
    im_buf_arr.tofile(str(dst_path))
    return True

def main():
    # --- 경로 설정 (절대경로) ---
    paths = [
        {
            "src": r"E:\github\healtheat_vision\data\ft_mix_yolo_CLASHE_sharpen\before_Fliltered_collected\images\train",
            "dst": r"E:\github\healtheat_vision\data\ft_mix_yolo_CLASHE_sharpen\images\train"
        },
        {
            "src": r"E:\github\healtheat_vision\data\ft_mix_yolo_CLASHE_sharpen\before_Fliltered_collected\images\val",
            "dst": r"E:\github\healtheat_vision\data\ft_mix_yolo_CLASHE_sharpen\images\val"
        }
    ]

    for item in paths:
        src_dir = Path(item["src"])
        dst_dir = Path(item["dst"])
        
        # 대상 폴더 생성
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 목록 추출
        img_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            img_files.extend(list(src_dir.glob(ext)))
        
        if not img_files:
            print(f"경고: {src_dir} 에 이미지가 없습니다.")
            continue

        print(f"\n🚀 {src_dir.name} 폴더 전처리 시작 (총 {len(img_files)}개)")
        
        for img_path in tqdm(img_files):
            dst_path = dst_dir / img_path.name
            success = apply_enhancement(img_path, dst_path)
            if not success:
                print(f"실패: {img_path.name}")

    print("\n✅ 모든 데이터셋의 전처리가 완료되었습니다!")

if __name__ == "__main__":
    main()