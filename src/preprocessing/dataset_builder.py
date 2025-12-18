import shutil
import random
from PIL import Image
from pathlib import Path

from src.config import Config
from src.preprocessing.cocoparser import CocoParser

class YoloDatasetBuilder:
    def __init__(self, merged_data: dict, out_dir: Path, val_ratio: float = 0.2):
        """
        merged_data : COCO 파서가 만들어낸 "이미지 파일명: annotation 묶음" 구조의 딕셔너리
        out_dir : YOLO 학습용 데이터셋이 생성될 최상위 디렉토리
        val_ratio : 전체 데이터 중 validation에 사용할 비율

        이 클래스는 COCO 데이터를 YOLOv8이 필요로 하는 폴더 구조로 재정렬하고,
        YOLO 포맷(annotation txt)을 자동 생성하는 역할을 한다.
        """

        self.data = merged_data
        self.out_dir = Path(out_dir)
        self.val_ratio = val_ratio

        # YOLOv8에서 요구하는 폴더 구조를 미리 정의해 둔다.
        # images/train, images/val, labels/train, labels/val
        self.img_train = self.out_dir / "images/train"
        self.img_val = self.out_dir / "images/val"
        self.lbl_train = self.out_dir / "labels/train"
        self.lbl_val = self.out_dir / "labels/val"

        # 디렉토리를 실제로 생성
        # parents=True -> 중간 폴더도 자동 생성
        # exist_ok=True -> 이미 존재해도 에러 없이 통과
        for p in [self.img_train, self.img_val, self.lbl_train, self.lbl_val]:
            p.mkdir(parents=True, exist_ok=True)

    def _convert_bbox(self, bbox, img_w, img_h):
        """
        COCO bbox: [x, y, w, h]  (왼쪽 상단 기준)
        YOLO bbox: [cx, cy, w, h] (중심점 기준, 정규화(nomalized))

        COCO -> YOLO 변환 공식:
            cx = x + w/2
            cy = y + h/2

        YOLO는 모든 값이 이미지 크기로 나누어진 정규화된 값이어야 함.
        """

        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2

        return cx / img_w, cy / img_h, w / img_w, h / img_h

    def build(self):
        """
        COCO 파서에서 얻은 merged_data를 기반으로:

        1. train / val split 수행
        2. 이미지를 YOLO 폴더 구조로 복사
        3. YOLO annotation(txt) 생성

        이 함수가 실행되면 YOLOv8 학습에 필요한 이미지/라벨이 완성된다.
        """
        # merged_data의 key는 이미지 파일명이다.
        keys = list(self.data.keys())

        # 데이터를 섞어서 train/val에 랜덤하게 배정
        # random.shuffle은 seed가 없으면 매번 다른 split이 되므로
        # 재현성을 위해 외부에서 seed를 설정해야 한다.
        random.shuffle(keys)

        # train/val split index 계산
        split_idx = int(len(keys) * (1 - self.val_ratio))

        # 앞부분은 train, 뒷부분은 val
        train_keys = keys[:split_idx]
        val_keys = keys[split_idx:]

        # 두 그룹(train/val)을 묶어서 루프를 돌린다
        for key, group in [("train", train_keys), ("val", val_keys)]:
            for fname in group:
                item = self.data[fname]

                img_path = item["img_path"] # 실제 이미지 Path 객체
                anns = item["annotations"] # COCO annotation 리스트

                # 이미지 열기: bbox scaling을 위해 width/height가 필요함
                img = Image.open(img_path)
                w, h = img.size

                # key(train/val)에 따라 라벨, 이미지 저장 위치를 결정
                if key == "train":
                    lbl_path = self.lbl_train / f"{img_path.stem}.txt"
                    dst_img = self.img_train / img_path.name
                else:
                    lbl_path = self.lbl_val / f"{img_path.stem}.txt"
                    dst_img = self.img_val / img_path.name

                # YOLO 학습 폴더로 이미지 복사
                shutil.copy(img_path, dst_img)

                # YOLO annotation txt 생성
                # 한 줄 형식: <class_id> <cx> <cy> <w> <h>
                with lbl_path.open("w") as f:
                    for ann in anns:
                        cid = ann["category_id"]  # 클래스 ID
                        bbox = ann["bbox"] # COCO bbox

                        # 중심 기반 YOLO bbox로 변환
                        cx, cy, bw, bh = self._convert_bbox(bbox, w, h)

                        # YOLO 형식으로 저장
                        f.write(f"{cid} {cx} {cy} {bw} {bh}\n")

    def write_yaml(self, yaml_path: Path):
        """
        YOLOv8 학습용 dataset.yaml 파일 생성

        - YOLO는 이 YAML을 기준으로 학습 시 이미지/라벨 경로와 클래스 이름을 로드한다.
        """
        # 모든 이미지에서 카테고리 뽑아오기
        all_categories = {}
        for item in self.data.values():
            for cid, cname in item["categories"].items():
                all_categories[cid] = cname

        # YAML 텍스트 생성
        yaml_text = (
            f"path: {self.out_dir}\n"
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"names:\n"
        )

        # category_id 0~N-1 순서대로 출력
        for cid in sorted(all_categories.keys()):
            yaml_text += f"  {cid}: '{all_categories[cid]}'\n"

        yaml_path.write_text(yaml_text, encoding="utf-8")
        print("[INFO] dataset.yaml 생성 완료:", yaml_path)


# 실행
if __name__ == "__main__":
    # 원본 이미지 및 annotation JSON 위치
    img_dir = Config.DATA_DIR / "clean_data/images"
    ann_root = Config.DATA_DIR / "clean_data/annotations"

    # YOLO dataset 출력 위치
    out_dir = Config.DATA_DIR / "yolo_dataset"

    # COCO 파서 실행 -> 이미지 단위 병합 데이터 생성
    parser = CocoParser(img_dir, ann_root)
    merged = parser.load()
    merged = parser.remap_categories(merged) 

    # YOLO dataset builder 실행
    random.seed(Config.SEED)  # train/val split 재현성 확보
    builder = YoloDatasetBuilder(merged, out_dir)
    builder.build()
    

    # dataset.yaml 생성
    builder.write_yaml(out_dir / "dataset.yaml")