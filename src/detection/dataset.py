import os
import json
import glob
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class PillDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        print(">>> [데이터 무결성 검사] 이미지 존재 여부를 전수 조사합니다...")
        
        all_ann_files = glob.glob(os.path.join(root, "train_annotations", "**", "*.json"), recursive=True)
        
        self.ann_files = []
        self.cat_to_idx = {}
        all_ids = set()

        valid_count = 0
        missing_count = 0
        
        # 1. 이미지 파일이 실존하는지 1차 검증
        for ann_path in tqdm(all_ann_files, desc="검증 진행"):
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                img_name = data['images'][0]['file_name']
                img_path = os.path.join(root, "train_images", img_name)
                
                if os.path.exists(img_path):
                    self.ann_files.append(ann_path)
                    valid_count += 1
                    
                    # 카테고리 ID 수집
                    for cat in data.get('categories', []):
                        all_ids.add(int(cat['id']))
                    if not data.get('categories') and 'annotations' in data:
                         for anno in data['annotations']:
                             all_ids.add(int(anno['category_id']))
                else:
                    missing_count += 1
            except:
                missing_count += 1
                continue

        print(f"검증 완료: 총 {len(all_ann_files)}개 중 {valid_count}개 학습 가능 (이미지 누락: {missing_count}개)")

        if valid_count == 0:
            raise RuntimeError("[오류] 학습 가능한 이미지가 0장입니다. 경로를 확인하세요.")

        self.sorted_cat_ids = sorted(list(all_ids))
        self.cat_to_idx = {cat_id: i + 1 for i, cat_id in enumerate(self.sorted_cat_ids)}

    def __getitem__(self, idx):
        ann_path = self.ann_files[idx]
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_info = data['images'][0]
        img_path = os.path.join(self.root, "train_images", img_info['file_name'])
        
        # 이미지 로드 및 텐서 변환
        img = Image.open(img_path).convert("RGB")
        img = T.ToTensor()(img)
        
        boxes = []
        labels = []
        
        for anno in data['annotations']:
            # [핵심 방어 코드] bbox 키가 없거나, 비어있거나, 숫자가 4개가 아니면 무시
            if 'bbox' not in anno or not anno['bbox'] or len(anno['bbox']) != 4:
                continue

            x, y, w, h = anno['bbox']
            
            # [핵심 방어 코드] 너비나 높이가 0 이하인 비정상 박스 무시
            if w <= 0 or h <= 0:
                continue
                
            # [핵심 방어 코드] 좌표가 이미지 크기를 벗어나는 경우 클리핑 (선택적 안전장치)
            # 여기서는 일단 정상적인 데이터만 담습니다.
            boxes.append([x, y, x + w, y + h])
            
            real_id = int(anno['category_id'])
            if real_id in self.cat_to_idx:
                labels.append(self.cat_to_idx[real_id])
            
        # 데이터가 모두 불량이라 boxes가 비어버린 경우 처리 (배경 이미지로 처리)
        if len(boxes) > 0:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx])
            }
        else:
            # 박스가 하나도 없으면(불량 포함), 빈 텐서를 반환하여 학습이 멈추지 않게 함
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx])
            }

        if self.transforms:
            pass 

        return img, target

    def __len__(self):
        return len(self.ann_files)