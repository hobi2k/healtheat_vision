import cv2
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
import os

# 프로젝트 루트 경로 추가 및 utils 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths

def get_korean_font_path():
    """운영체제별 한글 폰트 경로 반환 (viz.py 로직 참고)"""
    import platform
    os_name = platform.system()
    if os_name == "Darwin":  # macOS
        return "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    elif os_name == "Windows":
        return "C:/Windows/Fonts/malgun.ttf"
    else:  # Linux
        return "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

def load_class_map():
    """class_map.csv 로드"""
    if not paths.CLASS_MAP_PATH.exists():
        return {}
    df = pd.read_csv(paths.CLASS_MAP_PATH)
    mapping = {}
    for _, row in df.iterrows():
        y_id = int(row['yolo_id'])
        o_id = int(row.get('orig_id', row.get('category_id', 0)))
        name = row.get('class_name', 'Unknown')
        mapping[y_id] = (o_id, name)
    return mapping

def draw_info_with_hangeul(img, label_path, class_info):
    """PIL을 사용하여 텍스트를 2줄로 나누어 그립니다."""
    H, W = img.shape[:2]
    font_path = get_korean_font_path()
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype(font_path, 14) # 폰트 사이즈 살짝 조정
    except:
        font = ImageFont.load_default()

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            y_id, cx, cy, nw, nh = map(float, parts)
            y_id = int(y_id)
            orig_id, class_name = class_info.get(y_id, ("?", "알수없음"))

            x1 = int((cx - nw/2) * W)
            y1 = int((cy - nh/2) * H)
            x2 = int((cx + nw/2) * W)
            y2 = int((cy + nh/2) * H)

            # 1. 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

            # 2. 텍스트 구성 (2줄)
            line1 = f"ID: {y_id} (Orig:{orig_id})"
            line2 = f"{class_name}"
            
            # 텍스트 위치 계산 (박스 바로 위)
            # line_spacing을 고려하여 배경 박스 크기 계산
            bbox1 = draw.textbbox((x1, y1), line1, font=font)
            bbox2 = draw.textbbox((x1, y1), line2, font=font)
            
            text_w = max(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])
            text_h = (bbox1[3] - bbox1[1]) + (bbox2[3] - bbox2[1]) + 4 # 여백 포함
            
            # 텍스트 배경 그리기
            draw.rectangle([x1, y1 - text_h - 5, x1 + text_w + 4, y1], fill=(0, 255, 0))
            
            # 텍스트 쓰기
            draw.text((x1 + 2, y1 - text_h - 2), line1, font=font, fill=(0, 0, 0))
            draw.text((x1 + 2, y1 - (bbox2[3] - bbox2[1]) - 2), line2, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    class_info = load_class_map()
    img_files = list(paths.ADDITIONAL_TRAIN_IMG_DIR.glob("*.png"))
    
    if not img_files:
        print("❌ 이미지가 없습니다.")
        return

    samples = random.sample(img_files, min(5, len(img_files)))

    for img_p in samples:
        label_p = paths.ADDITIONAL_TRAIN_ANN_DIR / f"{img_p.stem}.txt"
        if not label_p.exists(): continue
            
        img = cv2.imread(str(img_p))
        if img is None: continue
        
        # 한글 처리된 이미지 가져오기
        img_result = draw_info_with_hangeul(img, label_p, class_info)
        
        win_name = f"Check: {img_p.name}"
        cv2.imshow(win_name, img_result)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()