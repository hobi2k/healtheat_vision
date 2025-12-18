import yaml
import os
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import paths

def create_additional_yaml():
    print("📝 추가 학습용 YAML 생성을 시작합니다...")
    
    # 1. 클래스 맵 로드 (names 리스트 생성용)
    if not paths.CLASS_MAP_PATH.exists():
        print(f"❌ 클래스 맵 파일이 없습니다: {paths.CLASS_MAP_PATH}")
        return
    
    class_map_df = pd.read_csv(paths.CLASS_MAP_PATH).sort_values('yolo_id')
    class_names = class_map_df['class_name'].tolist()
    
    # 2. YAML 데이터 구성
    # YOLO는 절대 경로를 권장하므로 .absolute() 사용
    data_config = {
        # 'path'를 빈칸이나 '.'으로 두면 현재 실행 위치(학습 스크립트 위치) 기준이 됩니다.
        # 데스크탑에서 학습을 실행할 때 해당 폴더 안에서 실행한다면 '.'이 가장 안전합니다.
        'path': '.', 
        'train': 'data/additional_splits/train.txt',
        'val': 'data/additional_splits/val.txt',
        'nc': len(class_names),
        'names': class_names
    }
    
    # 3. YAML 파일 저장
    # 기존 파일과 구분하기 위해 이름을 다르게 설정
    save_path = paths.CONFIGS_DIR / "additional_data.yaml"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)
    
    print(f"✅ YAML 생성 완료: {save_path}")
    print(f"   - 학습 클래스 수: {len(class_names)}개")
    print(f"   - 참조 Train 리스트: {data_config['train']}")

if __name__ == "__main__":
    create_additional_yaml()