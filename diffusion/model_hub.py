"""
모델 다운로드 및 캐시 보장 모듈

원칙:
- AugConfig를 읽기만 한다
- 모델 로딩은 절대 하지 않는다
- '존재 보장'만 책임진다
"""

from pathlib import Path
import urllib.request
from huggingface_hub import snapshot_download


# SAM
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


def ensure_sam_checkpoint(checkpoint_path: Path) -> None:
    """
    SAM checkpoint 파일 존재 보장

    checkpoint_path:
        cfg.sam_checkpoint
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        print(f"[OK] SAM checkpoint exists: {checkpoint_path}")
        return

    print("[DOWNLOAD] SAM checkpoint...")
    urllib.request.urlretrieve(SAM_URL, checkpoint_path)
    print("[DONE] SAM checkpoint downloaded")


# Stable Diffusion (HF)
def ensure_sd_model(model_id: str) -> None:
    """
    Stable Diffusion 모델을 HuggingFace cache에 다운로드

    주의:
    - 실제 파이프라인 로딩은 하지 않는다
    - snapshot_download는 weight / config만 캐시한다
    """
    print(f"[DOWNLOAD] Stable Diffusion model: {model_id}")

    snapshot_download(
        repo_id=model_id,
        resume_download=True,
    )

    print("[DONE] Stable Diffusion model cached")


# 통합 엔트리
def prepare_models(cfg) -> None:
    """
    증강 실행 전에 반드시 호출

    역할:
    - SAM checkpoint 보장
    - Stable Diffusion weight 캐시
    """
    ensure_sam_checkpoint(cfg.sam_checkpoint)
    ensure_sd_model(cfg.sd_inpaint_model)