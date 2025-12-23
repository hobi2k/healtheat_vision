import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPHelper:
    def __init__(self, model_name: str="openai/clip-vit-base-patch32", device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)
    
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)