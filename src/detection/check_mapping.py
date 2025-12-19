import json
import os
import glob
from tqdm import tqdm

# 경로가 맞는지 꼭 확인해야 합니다!
TRAIN_ANNOTATION_DIR = "./data/train_annotations"

def check_map():
    print(f"[검사 시작] '{TRAIN_ANNOTATION_DIR}' 폴더에서 json 파일을 찾습니다...")
    
    files = glob.glob(os.path.join(TRAIN_ANNOTATION_DIR, "**", "*.json"), recursive=True)
    if not files:
        print(" [치명적 오류] json 파일을 하나도 못 찾았습니다! 경로를 확인해주세요.")
        return

    print(f" json 파일 {len(files)}개를 찾았습니다. ID를 추출합니다...")
    
    all_ids = set()
    for path in tqdm(files):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # categories가 있으면 거기서, 없으면 annotations에서 추출
                if 'categories' in data:
                    for c in data['categories']:
                        all_ids.add(int(c['id']))
                elif 'annotations' in data:
                    for a in data['annotations']:
                        all_ids.add(int(a['category_id']))
        except:
            pass

    sorted_ids = sorted(list(all_ids))
    print("\n" + "="*30)
    print(f" [결과 확인] 총 {len(sorted_ids)}개의 진짜 ID를 찾았습니다.")
    print(f" 첫 5개 ID: {sorted_ids[:5]} ...")
    print(f" 마지막 5개 ID: ... {sorted_ids[-5:]}")
    print("="*30)
    
    if len(sorted_ids) == 0:
        print(" ID를 하나도 못 찾았습니다. 데이터 폴더 구조가 예상과 다릅니다.")
    else:
        print(" 성공! 이제 제출 코드를 돌려도 됩니다.")

if __name__ == "__main__":
    check_map()