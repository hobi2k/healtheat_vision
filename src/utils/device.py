import torch

def get_device():
    """
    사용 가능한 장치를 탐지하여 Ultralytics YOLO에 적합한 문자열로 반환합니다.
    """
    if torch.cuda.is_available():
        return "0"  # CUDA:0 (여러 개일 경우를 대비해 인덱스 번호 문자열 추천)
    
    if torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"