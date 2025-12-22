import torch
import numpy as np
import pandas as pd
from collections import Counter

import gradio as gr
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer
from tts.vits_tts import VITSTTS

from src.utils import paths

# 1. YOLO 로드 (모델 필요함)
yolo = YOLO(paths.ARTIFACTS_DIR / "runs/yolo11n_main_v2/weights/best.pt")

# class_map.csv 파일을 읽어옵니다. 
df_class = pd.read_csv(paths.ARTIFACTS_DIR / "class_map.csv")
# -------------------------

# 2. Qwen LLM 로드
QWEN_ID = "Qwen/Qwen2.5-3B-Instruct"

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

def remove_ack_sentences(text: str) -> str:
    """
    LLM이 습관적으로 출력하는 응답형 문장 제거
    (네, 이해했습니다 / 알겠습니다 등)
    """
    blacklist = [
        "네",
        "네.",
        "네,",
        "이해했습니다",
        "알겠습니다",
        "확인했습니다",
        "네 이해했습니다",
        "네, 이해했습니다",
        "네 알겠습니다",
    ]

    # 문장 단위로 분리
    sentences = (
        text.replace("\n", " ")
            .replace("!", ".")
            .replace("?", ".")
            .split(".")
    )

    filtered = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # 블랙리스트 포함 문장 제거
        if any(b in s for b in blacklist):
            continue
        filtered.append(s)

    if not filtered:
        return ""

    return ". ".join(filtered) + "."


# 3. VITS TTS 로드
tts_engine = VITSTTS(
    speaker_id=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


# 4. 유틸 함수들
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
    YOLO 결과 -> 라벨 + 위치 정보 + 임산부 주의 정보 추출
    """
    h, w = image.shape[:2]
    objects = []

    for b in results.boxes:
        # 1. 변수 정의: cls_id를 먼저 정의해야 합니다!
        cls_id = int(b.cls) 
        cls_name = results.names[cls_id]
        
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        pos = box_to_position(x1, y1, x2, y2, w, h)

        # 2. CSV에서 정보 매칭
        row = df_class[df_class['yolo_id'] == cls_id]
        is_pregnant_warning = False
        if not row.empty:
            is_pregnant_warning = bool(row.iloc[0]['is_pregnant'])

        objects.append({
            "label": clean_label(cls_name),
            "position": pos,
            "is_pregnant": is_pregnant_warning,
        })

    return objects


def yolo_to_prompt_with_position(objects: list[dict]) -> str:
    """
    위치 정보를 포함한 Qwen 프롬프트 생성
    """
    lines = [] # 빈 리스트를 먼저 만들고
    for obj in objects: # 하나씩 꺼내서
        # 1. 임산부 약인지 검사해서 글자를 정함
        warning_tag = "[임산부 주의 약물]" if obj.get('is_pregnant') else ""
    
        # 2. 이름 + 위치 + 아까 정한 글자를 합쳐서 리스트에 넣음
        lines.append(f"- {obj['label']} ({obj['position']}) {warning_tag}")

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
- 목록에 '[임산부 주의 약물]' 표시가 있는 약은 반드시 "이 약은 임산부에게 위험할 수 있으니 주의하세요"라는 내용을 문장에 포함할 것
- 마지막에 전체 객체 개수를 말할 것
- "네", "알겠습니다", "이해했습니다" 같은 응답형 문장은 절대 포함하지 말 것

자연스럽고 간결한 한국어 문장으로 설명하라.
""".strip()


# 5. 전체 파이프라인
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
        raw_text = qwen_generate(prompt)
        final_text = remove_ack_sentences(raw_text)

    # 음성 합성
    sr, wav = tts_engine.synthesize(final_text)

    return annotated_image, final_text, (sr, wav)


# 6. Gradio UI
demo = gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="YOLO 탐지 결과"),
        gr.Text(label="설명 텍스트"),
        gr.Audio(label="음성 출력"),
    ],
    title="YOLO + Qwen + VITS 알약 위치 음성 안내 데모",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
