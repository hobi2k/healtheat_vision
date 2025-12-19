import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset

class PillDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None, img_size=640):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transforms = transforms

        # JSON 파일 로드
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco = json.load(f)

        # image_id를 키로 사용하여 이미지 정보를 빠르게 찾을 수 있도록 매핑
        self.images = {img['id']: img for img in self.coco['images']}
        
        # image_id를 키로 사용하여 어노테이션 정보를 빠르게 찾을 수 있도록 매핑
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.ids = list(self.images.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.images[img_id]
        file_name = img_info['file_name']
        
        # 경로 결합
        img_path = os.path.join(self.root_dir, file_name)
        
        # 이미지 로드 시도
        image = cv2.imread(img_path)

        # [수정된 부분] 이미지가 제대로 안 읽혔으면 에러 메시지 띄우기
        if image is None:
            print(f"\n[CRITICAL ERROR] 이미지를 찾을 수 없습니다!")
            print(f"👉 찾는 경로: {img_path}")
            print(f"👉 실제 폴더에 이 파일이 있는지 확인해보세요.")
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        boxes = []
        labels = []
        
        if img_id in self.annotations:
            for ann in self.annotations[img_id]:
                x_min, y_min, w_box, h_box = ann['bbox']
                
                x_scale = self.img_size / w
                y_scale = self.img_size / h
                
                x_min = x_min * x_scale
                y_min = y_min * y_scale
                w_box = w_box * x_scale
                h_box = h_box * y_scale
                x_max = x_min + w_box
                y_max = y_min + h_box
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])

        target = {}
        if len(boxes) > 0:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            
        target['image_id'] = torch.tensor([img_id])

        return image_tensor, target

    def __len__(self):
        return len(self.ids)