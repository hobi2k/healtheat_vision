import torch
import numpy as np
import importlib.util, sys
import unicodedata
import soundfile as sf
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    PreTrainedTokenizerFast,
)
import gradio as gr

from src.utils import paths

# YOLO 모델 로드 (모델 필요함)
yolo = YOLO(paths.ARTIFACTS_DIR / "need models")

QWEN_ID = "Qwen/Qwen2.5-3B-Instruct"  # 적절한 한국어 실력의 LM 사용

qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    dtype=torch.float16,
    device_map="auto",
).eval()


def qwen_generate(prompt: str) -> str:
    """
    Qwen으로 최종 설명 문장 생성
    """
    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)

    with torch.no_grad():
        output = qwen_model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    text = qwen_tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split(prompt)[-1].strip()

def clean_label(obj: str) -> str:
    """
    YOLO 클래스명 정리
    - 첫 토큰만 사용
    - 한글만 유지
    """
    first = obj.split()[0]
    return "".join(ch for ch in first if "가" <= ch <= "힣")


def box_to_position(x1, y1, x2, y2, img_w, img_h) -> str:
    """
    bounding box 좌표 -> 말로 설명 가능한 위치
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 가로 방향
    if cx < img_w / 3:
        horiz = "왼쪽"
    elif cx < img_w * 2 / 3:
        horiz = "가운데"
    else:
        horiz = "오른쪽"

    # 세로 방향
    if cy < img_h / 3:
        vert = "위쪽"
    elif cy < img_h * 2 / 3:
        vert = "중앙"
    else:
        vert = "아래쪽"

    return f"{horiz} {vert}"


def extract_objects_with_position(results, image):
    """
    YOLO 결과 -> 라벨 + 위치 정보 추출
    """
    h, w = image.shape[:2]
    objects = []

    for b in results.boxes:
        cls_name = results.names[int(b.cls)]
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        pos = box_to_position(x1, y1, x2, y2, w, h)

        objects.append({
            "label": clean_label(cls_name),
            "position": pos,
        })

    return objects


def yolo_to_prompt_with_position(objects: list[dict]) -> str:
    """
    위치 정보를 포함한 Qwen 프롬프트 생성
    """
    lines = [
        f"- {obj['label']} ({obj['position']})"
        for obj in objects
    ]

    return f"""
너는 시각 장애인을 위한 음성 안내 AI다.
아래는 이미지에서 탐지된 객체와 그 위치다.

객체 목록:
{chr(10).join(lines)}

규칙:
- 좌표나 수치는 말하지 말 것
- 위치는 '왼쪽/가운데/오른쪽', '위/중앙/아래' 표현만 사용할 것
- 의학적 판단이나 추측은 하지 말 것
- 보이는 사실만 말할 것
- 마지막에 전체 객체 개수를 말할 것

자연스럽고 간결한 한국어 문장으로 설명하라.
""".strip()

# SpeechT5 한국어 모델 ID
MODEL_ID = "ahnhs2k/speecht5-korean-jamo"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000


# 한글 데이터 전처리 유틸 가져오기
utils_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="korean_text_utils.py"
)
spec = importlib.util.spec_from_file_location("korean_text_utils", utils_path)
korean_text_utils = importlib.util.module_from_spec(spec)
sys.modules["korean_text_utils"] = korean_text_utils
spec.loader.exec_module(korean_text_utils)

# utils 로드
inject_tokens_for_training = korean_text_utils.inject_tokens_for_training
decompose_jamo_with_placeholders = korean_text_utils.decompose_jamo_with_placeholders
normalize_korean = korean_text_utils.normalize_korean
prosody_split = korean_text_utils.prosody_split
prosody_pause = korean_text_utils.prosody_pause

# SpeechT5 모델 요소 로드 (tokenizer, vocoder, embedding)
tts_model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID).to(DEVICE).eval()
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)
vocoder = SpeechT5HifiGan.from_pretrained(
    "microsoft/speecht5_hifigan"
).to(DEVICE).eval()

spk_path = hf_hub_download(repo_id=MODEL_ID, filename="speaker_embedding.pth")
speaker_embeddings = torch.load(spk_path, map_location=DEVICE)

if speaker_embeddings.dim() == 1:
    speaker_embeddings = speaker_embeddings.unsqueeze(0)
speaker_embeddings = speaker_embeddings.to(DEVICE)


# TTS 생성 함수
def tts(text: str):
    """
    YOLO 결과 텍스트 -> 한국어 음성 생성
    STST 코드 기반으로 안정성 강화한 버전.
    """
    if text is None or text.strip() == "":
        return (SR, np.zeros(int(SR * 0.5), dtype=np.float32))  # 0.5초 무음

    norm = normalize_korean(text)

    segs_raw = prosody_split(norm)

    # 빈 세그먼트 제거
    segments = [s for s in segs_raw if s is not None and s.strip() != ""]
    if len(segments) == 0:
        return (SR, np.zeros(int(SR * 0.5), dtype=np.float32))

    audio_chunks = []

    for seg in segments:
        print(f"[TTS] Process segment: {seg}")

        # placeholder-aware jamo decomposition
        jamo_seq = decompose_jamo_with_placeholders(seg)

        if len(jamo_seq) == 0:
            print("[TTS] Empty jamo_seq. Skip this segment.")
            continue

        enc = tokenizer(
            jamo_seq,
            is_split_into_words=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            wav = tts_model.generate_speech(
                enc["input_ids"],
                speaker_embeddings=speaker_embeddings,
                vocoder=vocoder,
            )

        audio_chunks.append(wav.cpu().numpy())

        # punctuation pause 적용
        pause = prosody_pause(seg)
        if pause > 0:
            audio_chunks.append(np.zeros(int(SR * pause), dtype=np.float32))

    if len(audio_chunks) == 0:
        print("[TTS] No audio chunks generated. Return silence.")
        return (SR, np.zeros(int(SR * 0.5), dtype=np.float32))

    full = np.concatenate(audio_chunks, axis=0)
    return (SR, full)
 
def clean_label(obj: str) -> str:
    # 띄어쓰기 기준 첫 토큰만
    first = obj.split()[0]
    # 숫자/영문/특수문자 제거 (한글만 유지)
    only_korean = "".join(ch for ch in first if '가' <= ch <= '힣')
    return only_korean   

# YOLO -> 텍스트 -> 음성 전체 파이프라인
def pipeline(image):
    """
    image ->
    YOLO ->
    위치 추출 ->
    Qwen 설명 ->
    VITS 음성
    """
    # YOLO 추론
    results = yolo.predict(image, imgsz=640)[0]
    annotated_image = results.plot()

    objects = extract_objects_with_position(results, image)

    # 텍스트 생성
    if not objects:
        final_text = "이미지에서 탐지된 알약이 없습니다."
    else:
        prompt = yolo_to_prompt_with_position(objects)
        final_text = qwen_generate(prompt)

    sr, wav = tts(final_text)

    return annotated_image, final_text, (sr, wav)


# Gradio 인터페이스
demo = gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="YOLO 탐지 결과"),
        gr.Text(label="설명 텍스트"),
        gr.Audio(label="음성 출력"),
    ],
    title="YOLO + Qwen + SpeechT5 알약 위치 음성 안내 데모",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
