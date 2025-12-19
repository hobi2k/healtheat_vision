import torch
import os
from torch.utils.data import DataLoader
from src.detection.dataset import PillDataset
from src.detection.model import get_model
from src.detection.utils import collate_fn

# 경로 설정
DATA_ROOT = "./data"
SAVE_DIR = "./saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # 1. 데이터셋 객체 생성
    full_dataset = PillDataset(root=DATA_ROOT)
    
    # 2. 클래스 개수 설정 (배경 포함)
    num_classes = len(full_dataset.sorted_cat_ids) + 1
    print(f"학습 시작: 총 {num_classes - 1}개의 알약 클래스를 감지했습니다.")

    # 3. 모델 생성
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes)
    model.to(device)

    # 4. 데이터 로더 설정 (배치 사이즈는 사양에 따라 2~4 권장)
    train_loader = DataLoader(
        full_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=collate_fn
    )

    # 5. 옵티마이저 및 학습 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # 에포크 설정
    num_epochs = 50

    print("학습 루프 시작")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss/len(train_loader):.4f}")

        # 5 에포크마다 모델 저장
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(SAVE_DIR, f"faster_rcnn_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"모델 체크포인트 저장됨: {save_path}")

    # 최종 모델 저장
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "faster_rcnn_final.pth"))
    print("최종 학습 완료 및 모델 저장 성공")

if __name__ == "__main__":
    main()