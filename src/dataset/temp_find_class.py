import os
from pathlib import Path

# 1. 경로 설정
train_label_dir = Path(r"E:\github\healtheat_vision\data\ft_mix_yolo\labels\train")
target_class = "4"  # 찾고자 하는 클래스 ID

def find_files_with_class(label_path, class_id):
    found_files = []
    
    # 폴더 내 모든 .txt 파일 탐색
    label_files = list(label_path.glob("*.txt"))
    
    for txt_file in label_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    # 줄의 첫 번째 항목(class_id)이 대상과 일치하는지 확인
                    parts = line.split()
                    if parts and parts[0] == str(class_id):
                        found_files.append(txt_file.name)
                        break # 한 파일에 여러 개 있어도 파일명은 한 번만 추가
        except Exception as e:
            print(f"파일 읽기 오류 ({txt_file.name}): {e}")
            
    return found_files

# 2. 실행
results = find_files_with_class(train_label_dir, target_class)

# 3. 결과 출력
print(f"--- 클래스 {target_class} 탐색 결과 ---")
print(f"찾은 파일 개수: {len(results)}개")
print("-" * 40)

# 상위 10개만 먼저 보여주기
for fileName in results[:10]:
    print(fileName)

if len(results) > 10:
    print(f"... 외 {len(results) - 10}개 더 있음")