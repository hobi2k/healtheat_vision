import torch

def get_device():
    """
    사용 가능한 연산 장치를 자동으로 탐지하여 반환하는 함수.

    동작 우선순위:
    1. CUDA GPU가 존재하면 'cuda' 장치를 반환
    2. (Mac 전용) Apple Silicon MPS 가속기가 가능하면 'mps' 장치를 반환
    3. 위 두 옵션이 모두 없을 경우 CPU 장치를 반환

    Returns:
        torch.device: 선택된 연산 장치 (cuda / mps / cpu)
    """
    # 1) CUDA (Colab or PC)
    # (코랩 설정 필요: 메뉴 - 런타임 - 런타임 유형 변경 - 하드웨어 가속기)
    if torch.cuda.is_available():
        print("CUDA GPU detected.")
        return torch.device("cuda")

    # 2) Apple Silicon (Local Mac)
    if torch.backends.mps.is_available():
        print("Apple MPS GPU detected.")
        return torch.device("mps")

    # 3) CPU fallback
    print("No GPU detected. Using CPU.")
    return torch.device("cpu")