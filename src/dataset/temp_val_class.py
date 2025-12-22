import os
from collections import Counter
from pathlib import Path

# 1. 경로 설정
val_label_dir = Path(r"E:\github\healtheat_vision\data\ft_mix_yolo\labels\val")

def analyze_yolo_labels(label_path):
    class_counts = Counter()
    file_count = 0
    
    # 폴더 내의 모든 .txt 파일 탐색
    label_files = list(label_path.glob("*.txt"))
    
    for txt_file in label_files:
        file_count += 1
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    # YOLO 포맷: <class_id> <x_center> <y_center> <width> <height>
                    class_id = line.split()[0]
                    class_counts[int(class_id)] += 1
                    
    return class_counts, file_count

# 2. 분석 실행
counts, total_files = analyze_yolo_labels(val_label_dir)

# 3. 결과 출력
print(f"--- 분석 결과 ({val_label_dir.name}) ---")
print(f"전체 라벨 파일 수: {total_files}개")
print(f"발견된 클래스 종류: {len(counts)}종")
print("-" * 30)
print(f"{'Class ID':<10} | {'개수(Object Count)':<15}")
print("-" * 30)

# 클래스 번호 순으로 정렬해서 출력
for class_id in sorted(counts.keys()):
    print(f"{class_id:<10} | {counts[class_id]:<15}")

print("-" * 30)