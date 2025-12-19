import torch
from torch.utils.data import DataLoader
from src.preprocessing.dataset import PillDataset

# 경로 설정
root_dir = "./data/train_images"
annotation_file = "./data/train_annotations.json"

def main():
    # 데이터셋 인스턴스 생성
    dataset = PillDataset(root_dir=root_dir, annotation_file=annotation_file, img_size=640)
    
    print(f"데이터셋 전체 크기: {len(dataset)}")

    # 데이터로더 생성 (배치 단위로 데이터를 뽑아주는 역할)
    # collate_fn은 다양한 크기의 데이터를 묶을 때 필요하지만, 여기서는 단순 테스트
    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 첫 번째 배치를 가져와서 형태(Shape) 확인
    images, targets = next(iter(data_loader))

    print(f"\n--- 배치(Batch) 정보 확인 ---")
    print(f"배치 크기: {len(images)}")
    
    # 첫 번째 이미지 텐서 확인
    print(f"이미지 텐서 형태 (Channels, Height, Width): {images[0].shape}")
    print(f"이미지 값 범위: {images[0].min()} ~ {images[0].max()}")
    
    # 첫 번째 라벨 정보 확인
    print(f"라벨(박스) 개수: {len(targets[0]['boxes'])}")
    print(f"라벨(박스) 좌표 예시: \n{targets[0]['boxes']}")
    
    print("\n✅ 데이터셋 클래스가 정상적으로 작동합니다.")

if __name__ == "__main__":
    main()