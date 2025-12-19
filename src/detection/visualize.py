import torch
import cv2
import os
import random
import glob
import torchvision.transforms as T
from src.detection.model import get_model

# ================= 설정 =================
MODEL_PATH = "./saved_models/faster_rcnn_epoch_20.pth" # 20 에포크 모델
TEST_DIR = "./data/test_images"
SAVE_DIR = "./visualized_results"
SCORE_THRESHOLD = 0.5  # 0.5점 이상인 것만 표시
# ========================================

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 모델 로드
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    num_classes = checkpoint['roi_heads.box_predictor.cls_score.weight'].shape[0]
    model = get_model(num_classes)
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    # 2. 이미지 가져오기
    image_files = glob.glob(os.path.join(TEST_DIR, "*.png")) + glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    sample_files = random.sample(image_files, 5) # 5장 랜덤 추출

    transform = T.Compose([T.ToTensor()])
    print(f">>> 추론 시작! (결과는 {SAVE_DIR} 폴더에 JPG로 저장됨)")

    with torch.no_grad():
        for img_path in sample_files:
            original_img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img_rgb).to(device)

            prediction = model([img_tensor])[0]

            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()

            found = False
            for i, box in enumerate(boxes):
                score = scores[i]
                if score < SCORE_THRESHOLD:
                    continue
                
                found = True
                x1, y1, x2, y2 = map(int, box)
                
                # 초록색 박스 그리기
                cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # 점수와 라벨 표시
                label_id = labels[i]
                text = f"ID:{label_id} ({score:.2f})"
                cv2.putText(original_img, text, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # [핵심] 무조건 JPG로 저장 (호환성 해결)
            filename = os.path.basename(img_path).split('.')[0] + ".jpg"
            save_path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(save_path, original_img)
            
            if found:
                print(f" 알약 검출 성공! -> {save_path}")
            else:
                print(f" 검출 실패 (기준 점수 미달) -> {save_path}")

    print("\n[확인 방법]")
    print("1. 왼쪽 파일 탐색기에서 'visualized_results' 폴더를 클릭하세요.")
    print("2. 안에 있는 .jpg 파일을 클릭하면 오른쪽 화면에 바로 뜹니다.")

if __name__ == "__main__":
    main()