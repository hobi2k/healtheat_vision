from pathlib import Path
import json
from collections import defaultdict

class CocoParser:
    """
    COCO 형식의 어노테이션을 여러 JSON 파일에서 읽어 이미지 파일 단위로 annotation을 모두 합쳐주는 파서(Parser) 클래스.

    원시 데이터 특성
      - COCO 형식에서는 (images, annotations, categories)가 하나의 JSON에 들어있다.
      - 이미지 하나당 JSON이 따로 있거나, 여러 JSON에 나누어져 있는 경우가 많다.
      - 모델을 학습하려면 결국 "이미지 파일명 <-> 그 이미지의 모든 annotation" 구조여야 하므로
        이를 다시 모아주는 전처리 과정이 필수적이다.
    """

    def __init__(self, img_dir: Path, ann_root: Path):
        """
        img_dir : 이미지들이 저장된 폴더 경로
        ann_root : COCO annotation JSON 파일들이 들어 있는 상위 폴더

        Path 객체를 사용하면 OS에 따라 달라지는 경로 구분자를 자동 처리할 수 있어 안정적.
        """
        self.img_dir = Path(img_dir)
        self.ann_root = Path(ann_root)

    def load(self):
        """
        1. 전체 JSON을 읽고
        2. 이미지 파일명을 기준으로 annotation을 통합한 뒤
        3. {filename: {annotations:[], categories:{}, img_path:Path}} 형태로 반환한다.

        이 함수의 핵심 로직:
          1. 이미지 목록을 수집한다.
          2. JSON 파일을 하나씩 읽는다.
          3. 각 JSON의 images, categories, annotations를 연결한다.
          4. 이미지 파일이 없는 annotation은 제외한다(데이터 정합성을 위해 매우 중요).
        """

        # 이미지 파일 목록 만들기
        # 딕셔너리 형태: { "이미지.png": Path(".../이미지.png"), "이미지2.jpg": Path(".../이미지.jpg") }
        images = {
            p.name: p
            for p in self.img_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }

        # 이미지 단위로 annotation을 모으기
        # defaultdict를 쓰는 이유: key(이미지 파일명)가 처음 등장해도 자동으로 기본 dict이 생성되도록 함.
        merged = defaultdict(lambda: {"annotations": [], 
                                      "categories": {}, 
                                      "img_path": None})

        # annotation JSON 파일들을 전부 수집
        # rglob("*.json"): 하위 폴더까지 전부 포함해 JSON 탐색
        json_files = list(self.ann_root.rglob("*.json"))

        # JSON 파일을 하나씩 읽기
        for jpath in json_files:
            with jpath.open("r", encoding="utf-8") as f:
                coco = json.load(f)

            # category 정보 처리
            # categories: {id -> name} 형태의 매핑을 만든다.
            # get("name")이 없다면 class_{id}라는 기본 이름을 만들기.
            categories = {
                c["id"]: c.get("name", f"class_{c['id']}")
                for c in coco.get("categories", [])
            }

            # images 섹션 읽기
            for img_info in coco.get("images", []):
                fname = img_info["file_name"]

                # 이미지 파일이 실제로 존재하지 않으면 무시
                if fname not in images:
                    continue

                # 실제 이미지 경로 저장
                merged[fname]["img_path"] = images[fname]

                # 카테고리 이름 기록
                # (각 JSON마다 categories가 있을 수 있으므로 반복적으로 합치기)
                for cid, cname in categories.items():
                    merged[fname]["categories"][cid] = cname

                # annotation을 연결하기 위해 image_id가 필요하다
                image_id = img_info["id"]

                # annotations 섹션 읽기
                for ann in coco.get("annotations", []):
                    # 이 annotation이 현재 이미지에 속하는지 확인
                    if ann.get("image_id") == image_id:

                        # bbox가 없거나 4개 값이 아니면 제외 (데이터 정합성 체크)
                        if "bbox" in ann and len(ann["bbox"]) == 4:
                            merged[fname]["annotations"].append(ann)

        # annotation이 없는 이미지는 제거
        merged = {
            fname: data
            for fname, data in merged.items()
            if len(data["annotations"]) > 0
        }

        # 최종 병합 결과 반환
        return merged
    
    def remap_categories(self, merged):
        """
        YOLO 학습을 위해 category_id를 0~N-1 로 재매핑하고,
        yolo_id -> {coco_id, name} 구조의 mapping.json을 생성한다.
        """
        # 모든 COCO category id 수집
        all_classes = set()
        cid_to_name = {}

        for item in merged.values():
            for cid, cname in item["categories"].items():
                all_classes.add(cid)
                cid_to_name[cid] = cname  # name 정보 저장
        
        # old_id -> new_id
        new_ids = {old_id: new_id for new_id, old_id in enumerate(sorted(all_classes))}

        # YOLO ID -> {COCO ID, class_name}
        yolo_mapping = {
            new_id: {
                "coco_id": old_id,
                "name": cid_to_name[old_id]
            }
            for old_id, new_id in new_ids.items()
        }

        # 재매핑 적용
        for item in merged.values():
            # categories 재생성
            new_cat_data = {}
            for old_cid, cname in item["categories"].items():
                new_cat_data[new_ids[old_cid]] = cname
            item["categories"] = new_cat_data

            # annotation category_id 재매핑
            for anno in item["annotations"]:
                anno["category_id"] = new_ids[anno["category_id"]]

        # 최종 매핑 저장
        mapping_path = self.ann_root / "category_mapping.json"
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(yolo_mapping, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved category_mapping to: {mapping_path}")

        return merged

        