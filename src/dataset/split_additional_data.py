import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths

def split_additional_data(val_size=0.2, seed=42):
    print("🚀 추가 데이터 스플릿 작업을 시작합니다...")
    
    # 1. 추가 데이터 경로 확인
    img_dir = paths.ADDITIONAL_TRAIN_IMG_DIR
    if not img_dir.exists():
        print(f"❌ 데이터 폴더가 없습니다: {img_dir}")
        return

    # 2. 이미지 파일 리스트 확보
    all_images = list(img_dir.glob("*.png"))
    
    # 3. 그룹화 (Data Leakage 방지)
    # 파일명 예시: err_303_00001_orig_IMG_123.png -> 'err_303_00001'을 키로 묶음
    # 이렇게 하면 동일 알약에서 나온 증강본들이 한 세트로 묶임
    data_groups = {}
    for img_p in all_images:
        parts = img_p.name.split('_')
        # 접두어(err) + 카테고리ID + 순번까지를 그룹 키로 사용
        group_key = "_".join(parts[:3]) 
        
        if group_key not in data_groups:
            data_groups[group_key] = []
        data_groups[group_key].append(str(img_p.absolute()))

    group_keys = list(data_groups.keys())
    print(f"📦 총 알약 객체 수: {len(group_keys)}개 (증강 포함 전체 이미지: {len(all_images)}장)")

    # 4. 그룹 단위로 Train / Val 분할
    train_keys, val_keys = train_test_split(
        group_keys, 
        test_size=val_size, 
        random_state=seed,
        shuffle=True
    )

    # 5. 최종 이미지 경로 리스트 생성
    train_list = []
    for k in train_keys:
        for img_path in data_groups[k]:
            # PROJECT_ROOT 기준 상대 경로로 변환하여 저장
            rel_path = Path(img_path).relative_to(paths.PROJECT_ROOT)
            train_list.append(str(rel_path))
        
    val_list = []
    for k in val_keys:
        for img_path in data_groups[k]:
            rel_path = Path(img_path).relative_to(paths.PROJECT_ROOT)
            val_list.append(str(rel_path))

    # 6. 결과 저장 (YOLO 학습용 txt 파일)
    # paths.ADDITIONAL_SPLITS_DIR가 없다면 생성 (paths.py에 정의되어 있어야 함)
    save_dir = paths.DATA_DIR / "additional_splits"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_txt = save_dir / "train.txt"
    val_txt = save_dir / "val.txt"

    with open(train_txt, 'w') as f:
        f.write("\n".join(train_list))
    
    with open(val_txt, 'w') as f:
        f.write("\n".join(val_list))

    print(f"✅ 스플릿 완료!")
    print(f"   - Train: {len(train_list)}장 (그룹: {len(train_keys)}개)")
    print(f"   - Val: {len(val_list)}장 (그룹: {len(val_keys)}개)")
    print(f"📍 저장 위치: {save_dir}")

if __name__ == "__main__":
    split_additional_data()