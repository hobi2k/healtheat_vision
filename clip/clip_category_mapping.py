import torch
from PIL import Image
from .clipping_wrapper import CLIPHelper

class PillClassifier:
    def __init__(self, clip_helper: CLIPHelper, category_mapping: dict):
        self.clip = clip_helper
        self.mapping = category_mapping

        # 정수 기반 정렬 (중요!)
        items = sorted(self.mapping.items(), key=lambda x: int(x[0]))

        names = [v["name"] for k, v in items]
        self.category_ids = [int(k) for k, v in items]

        self.text_emb = self.clip.encode_texts(names)

    def classify_pill(self, pil_image: Image.Image) -> int:
        img_emb = self.clip.encode_images([pil_image])  # (1, D)
        # cosine similarity
        sim = (img_emb @ self.text_emb.T)[0]  # (C,)
        idx = int(torch.argmax(sim).item())
        return self.category_ids[idx]
