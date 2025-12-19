"""
VITS TTS wrapper

- 현재 데모용에서는 hobi2k/vits를 읽고 의존성 설치 후 실행 가능

- Codebase: hobi2k/vits
- Checkpoint: ORI-Muchim/VITS_multi_speaker_fine_tuning
- This file is the ONLY place that knows VITS internals.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from huggingface_hub import hf_hub_download

# 1. VITS 코드베이스 경로 등록
# 현재 파일: healtheat_vision__/tts/vits_tts.py
# VITS 코드: healtheat_vision__/tts/vits/
VITS_CODEBASE = Path(__file__).resolve().parent / "vits"

if not VITS_CODEBASE.exists():
    raise RuntimeError(f"[VITS] codebase not found: {VITS_CODEBASE}")

# VITS 원본 코드 그대로 import 하기 위함
sys.path.insert(0, str(VITS_CODEBASE))

# 2. hobi2k/vits 원본 모듈 import
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


# 3. VITS TTS 클래스
class VITSTTS:
    """
    VITS TTS engine (single instance)

    - multi-speaker supported
    - inference-only
    """

    def __init__(
        self,
        speaker_id: int = 0,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.speaker_id = speaker_id

        # HF 체크포인트 다운로드
        repo_id = "ORI-Muchim/VITS_multi_speaker_fine_tuning"
        subdir = "ko_fine_tuning_22050hz"

        self.config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subdir}/config.json",
        )
        self.ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subdir}/G_172000.pth",
        )

        # hparams 로드
        self.hps = utils.get_hparams_from_file(self.config_path)
        self.sample_rate = int(self.hps.data.sampling_rate)

        if not (0 <= speaker_id < self.hps.data.n_speakers):
            raise ValueError(
                f"speaker_id {speaker_id} out of range "
                f"(0 ~ {self.hps.data.n_speakers - 1})"
            )

        # VITS Generator 로드 (원본 코드 그대로)
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        ).to(self.device)

        self.net_g.eval()
        utils.load_checkpoint(self.ckpt_path, self.net_g, None)

    # 4. 원본 get_text 로직
    def _get_text(self, text: str) -> torch.LongTensor:
        """
        text -> token id sequence
        (exactly same as original inference code)
        """
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)

        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)

        return torch.LongTensor(text_norm)

    # 5. 외부에서 호출하는 메인 API
    def synthesize(self, text: str):
        """
        text -> (sample_rate, waveform)

        This function is the ONLY public interface.
        """
        if not text or not text.strip():
            return (
                self.sample_rate,
                np.zeros(int(self.sample_rate * 0.5), dtype=np.float32),
            )

        # 너무 긴 문장 방지: 간단한 문장 분할
        segments = [
            s.strip()
            for s in text.replace("\n", " ").split(".")
            if s.strip()
        ]

        audio_chunks = []

        for seg in segments:
            stn = self._get_text(seg).to(self.device)

            with torch.no_grad():
                x = stn.unsqueeze(0)
                x_len = torch.LongTensor([stn.size(0)]).to(self.device)
                sid = torch.LongTensor([self.speaker_id]).to(self.device)

                audio = self.net_g.infer(
                    x,
                    x_len,
                    sid=sid,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0,
                )[0][0, 0].cpu().float().numpy()

            audio_chunks.append(audio)

            # 문장 간 짧은 무음
            audio_chunks.append(
                np.zeros(int(self.sample_rate * 0.15), dtype=np.float32)
            )

        return self.sample_rate, np.concatenate(audio_chunks)
