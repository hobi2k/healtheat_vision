import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    # COCO 데이터셋으로 사전 학습된 모델 로드
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # 분류기(헤드) 교체
    # 모델의 입력 특징(feature) 크기 가져오기
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 미리 학습된 헤드를 제거하고, 사용자의 클래스 개수에 맞는 새로운 헤드로 교체
    # num_classes는 배경(background) 클래스를 포함해야 함
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model