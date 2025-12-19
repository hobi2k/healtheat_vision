import torch
from src.detection.model import get_model
from src.preprocessing.dataset import PillDataset
from torch.utils.data import DataLoader

# 경로 설정
root_dir = "./data/train_images"
annotation_file = "./data/train_annotations.json"

def main():
    # 1. 데이터셋 및 로더 준비
    # 배경(0) + 실제 알약 클래스 개수를 알아야 함
    # 우선 데이터셋을 로드하여 클래스 개수 파악
    dataset = PillDataset(root_dir=root_dir, annotation_file=annotation_file, img_size=640)
    
    # 데이터셋 내부의 coco 객체에서 카테고리 정보 확인
    num_categories = len(dataset.coco['categories'])
    print(f"데이터셋의 알약 클래스 개수: {num_categories}")
    
    # 모델에 들어갈 클래스 개수 = 알약 종류 + 1 (배경)
    num_classes = num_categories + 1
    print(f"모델 설정 클래스 개수(배경 포함): {num_classes}")

    # 데이터 로더 (배치 사이즈 2로 테스트)
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # 2. 모델 생성
    model = get_model(num_classes)
    model.eval() # 평가 모드로 설정 (테스트용)

    # 3. 데이터 주입 테스트
    images, targets = next(iter(data_loader))
    
    print("모델에 데이터를 주입합니다...")
    
    # 모델 포워드 패스 (Forward Pass)
    # 학습 시에는 loss를 반환하고, 평가 시에는 predictions를 반환함
    predictions = model(images, targets)
    
    print("모델 출력 결과 형식 확인:")
    print(f"예측된 박스 개수(첫 번째 이미지): {len(predictions[0]['boxes'])}")
    print(f"예측된 점수(Scores) 예시: {predictions[0]['scores'][:5]}")
    
    print("모델 테스트 완료. 정상 작동합니다.")

if __name__ == "__main__":
    main()