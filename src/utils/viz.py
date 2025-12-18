import platform
import matplotlib.pyplot as plt

def set_korean_font():
    """
    운영체제별로 적절한 한국어 폰트를 설정하고 
    마이너스 기호 깨짐 현상을 방지합니다.
    """
    os_name = platform.system()
    
    if os_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif os_name == "Darwin":  # macOS
        plt.rcParams["font.family"] = "AppleGothic"
    else:  # Linux/Ubuntu 등
        plt.rcParams["font.family"] = "NanumGothic"
        
    # 마이너스 기호 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False
    print(f"Font system set for: {os_name}")