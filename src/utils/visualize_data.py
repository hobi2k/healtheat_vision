import json
from pathlib import Path 
import matplotlib
matplotlib.use("Agg")
import koreanize_matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from PIL import Image

from src.config import Config


def visualize_coco(img_dir: Path, ann_root: Path, save_dir: Path):
    """
    AiHub 경구약제처럼,
    - '이미지 하나당 여러 JSON(annotation)이 존재하는' COCO 스타일 데이터셋을 시각화하는 함수.

    매개변수:
        img_dir  : 이미지들이 들어 있는 최상위 폴더 (예: data/raw/train_images)
        ann_root : 어노테이션(JSON)들이 들어 있는 최상위 폴더 (예: data/raw/train_annotations)
                   - 하위에 여러 폴더가 있어도 rglob 로 모두 탐색
        save_dir : 시각화 결과(.png)를 저장할 폴더
    """

    # 혹시 문자열이 들어와도 안전하게 Path 객체로 변환
    img_dir = Path(img_dir)
    ann_root = Path(ann_root)
    save_dir = Path(save_dir)

    # save_dir가 존재하지 않으면, 부모 디렉터리까지 포함해서 생성
    # parents=True : 상위 폴더도 같이 생성
    # exist_ok=True : 이미 존재해도 에러 내지 않음
    save_dir.mkdir(parents=True, exist_ok=True)

    # train_images 안의 모든 이미지 파일 탐색
    images = sorted([
        p
        for p in img_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])
    print(f"[INFO] 전체 이미지 수: {len(images)}")

    # train_annotations 안의 모든 JSON 파일 탐색
    # ann_root.rglob("*.json"): ann_root 아래의 모든 하위 디렉터리를 재귀적으로 뒤져서 확장자가 .json 인 파일을 전부 찾는다.
    json_files = list(ann_root.rglob("*.json"))
    print(f"[INFO] 전체 annotation JSON 수: {len(json_files)}")

    # JSON을 "이미지 파일명" 기준으로 빠르게 찾아올 수 있게 매핑
    # - key : COCO JSON 내부의 images[].file_name 값 (실제 이미지 파일명)
    # - value : (image 정보, coco 전체 dict) 쌍들의 리스트 같은 이미지가 여러 JSON에 등장할 수 있으므로 리스트로 저장
    img_to_jsons = {}

    # 모든 JSON 파일을 순회하면서, 내부의 'images' 정보를 읽어온다.
    for jpath in json_files:
        with jpath.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        # coco.get("images", []) :
        # coco 딕셔너리에서 "images" 키에 해당하는 값을 가져오고 없으면 기본값으로 빈 리스트 [] 사용
        for img_info in coco.get("images", []):
            fname = img_info.get("file_name") 

            # 해당 파일명이 딕셔너리에 없는 key라면 먼저 빈 리스트로 초기화
            if fname not in img_to_jsons:
                img_to_jsons[fname] = []

            # 이 이미지 정보와 전체 coco JSON 을 함께 저장
            # 나중에 이 쌍을 통해:
            #   - img_info["id"] 로 image_id를 얻고
            #   - coco["annotations"] 에서 해당 image_id의 annotation만 골라낼 수 있다.
            img_to_jsons[fname].append((img_info, coco))

    # 4) 모든 train 이미지에 대해 반복
    for img_path in images:
        fname = img_path.name   # 예: Path(".../K-001900.png").name → "K-001900.png"

        # 이 이미지 파일명에 해당하는 JSON 정보가 하나도 없는 경우
        if fname not in img_to_jsons:
            print(f"[WARN] 해당 이미지에 대한 JSON 없음: {fname}")
            continue

        print(f"[INFO] 이미지 {fname} 에 대한 JSON 개수: {len(img_to_jsons[fname])}")

        # 이미지 파일 로드 (Pillow 사용)
        pil_img = Image.open(img_path)

        # 이 이미지에 대해 모든 JSON에서 모은 annotation을 저장할 리스트
        merged_annotations = []

        # category_id → category_name 매핑 딕셔너리
        # 예: {1: "pill_red", 2: "pill_blue", ...}
        category_map = {}


        # 5) 이 이미지와 관련된 '모든' JSON에서 annotation 수집
        #    img_to_jsons[fname] 는 (img_info, coco_dict) 튜플들의 리스트
        for img_info, coco in img_to_jsons[fname]:
            image_id = img_info["id"]           # COCO에서 이 이미지의 고유 id

            # categories 정보에서 id → name 매핑 만들어두기
            categories = coco.get("categories", [])
            for c in categories:
                # c["id"] : category_id (정수)
                # c.get("name", f"class_{c['id']}") : name이 없으면 "class_카테고리ID"로 대체
                category_map[c["id"]] = c.get("name", f"class_{c['id']}")

            # coco.get("annotations", []) :
            #   이 JSON에 들어있는 모든 annotation 리스트
            for ann in coco.get("annotations", []):
                # ann.get("image_id") == image_id :
                #   이 annotation이 우리가 보고 있는 이미지에 속하는지 확인
                # "bbox" in ann and len(ann["bbox"]) == 4 :
                #   bbox 키가 존재하고, [x, y, w, h] 4개 값이 있는지 확인
                if (
                    ann.get("image_id") == image_id
                    and "bbox" in ann
                    and len(ann["bbox"]) == 4
                ):
                    merged_annotations.append(ann)

        print(f"[INFO] {fname} merged annotations: {len(merged_annotations)}")


        # 6) 시각화
        #    - 하나의 figure(그림)와 axes(좌표축)를 생성
        #    - figsize=(10, 10) : 인치 단위 크기 (너비 10, 높이 10)

        fig, ax = plt.subplots(figsize=(10, 10))

        # 배경으로 실제 이미지를 깔기
        ax.imshow(pil_img)

        # 축 눈금, 테두리 등을 숨기기 (이미지만 보이게)
        ax.axis("off")

        # 수집된 모든 annotation에 대해 박스 + 클래스 이름 시각화
        for ann in merged_annotations:
            # COCO bbox 형식: [x_min, y_min, width, height]
            x, y, w, h = ann["bbox"]

            # category_id를 category_map에서 찾아서 사람이 읽을 수 있는 이름으로 변환
            cat_name = category_map.get(ann["category_id"], "unknown")

            # patches.Rectangle:
            #  - (x, y) 좌표에서 시작하는 width=w, height=h 인 사각형 도형
            #  - edgecolor="red" : 테두리 색을 빨간색으로
            #  - facecolor="none": 내부는 투명
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)   # 실제 그림에 사각형 추가

            # bbox 위에 카테고리 이름 텍스트 표시
            ax.text(
                x,               # 텍스트 x 좌표 (bbox 왼쪽 위 근처)
                y - 3,           # 텍스트 y 좌표 (박스보다 조금 위)
                cat_name,        # 실제 표시할 문자열
                fontsize=10,
                color="red",
                weight="bold",
                bbox=dict(       # 텍스트 배경 상자 스타일 지정
                    facecolor="white",  # 배경 흰색
                    alpha=0.6,          # 투명도 (0~1)
                    edgecolor="none",   # 테두리 없음
                ),
            )


        # 7) 그림을 파일로 저장
        #
        #    img_path.stem : 확장자 제외한 파일명 (예: "K-001900.png" → "K-001900")
        #    f"{img_path.stem}_viz.png" : "K-001900_viz.png"

        save_path = save_dir / f"{img_path.stem}_viz.png"

        # plt.savefig:
        #  - 현재 figure를 지정한 경로에 이미지 파일로 저장
        #  - bbox_inches="tight" : 여백을 최소화해서 타이트하게 자름
        plt.savefig(save_path, bbox_inches="tight")

        # 메모리 관리를 위해 figure를 닫음
        plt.close(fig)

        print(f"[INFO] 저장됨: {save_path}")


# 이 파일이 '직접 실행'될 때만 아래 코드 실행
# 다른 모듈에서 import 할 때는 실행되지 않음
if __name__ == "__main__":
    visualize_coco(
        img_dir=Config.DATA_DIR / "clean_data/images",          # 예: data/raw/train_images
        ann_root=Config.DATA_DIR / "clean_data/annotations",    # 예: data/raw/train_annotations
        save_dir=Config.BASE_DIR / "outputs/visualization_processed",    # 예: 프로젝트 루트/outputs/visualization
    )
