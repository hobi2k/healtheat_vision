import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# 프로젝트 루트 경로 설정
FILE_PATH = Path(__file__).resolve()
ROOT = FILE_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import paths

def get_plotly_fig(img_path, lbl_path):
    """한 장의 이미지와 YOLO 박스를 Plotly Figure로 반환"""
    img = cv2.imread(str(img_path))
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # 1. 이미지 추가
    fig = px.imshow(img)

    # 2. YOLO 박스 추가
    if lbl_path.exists():
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cls_id = int(parts[0])
                x_c, y_c, bw, bh = map(float, parts[1:])
                
                # 좌표 복원
                x1 = (x_c - bw/2) * w
                y1 = (y_c - bh/2) * h
                x2 = (x_c + bw/2) * w
                y2 = (y_c + bh/2) * h

                # 박스 레이어 추가
                fig.add_shape(
                    type="rect", x0=x1, y0=y1, x1=x2, y1=y2,
                    line=dict(color="Lime", width=3),
                )
                # 클래스 텍스트 추가
                fig.add_annotation(
                    x=x1, y=y1, text=f"ID:{cls_id}",
                    showarrow=False, bgcolor="Lime", font=dict(size=12, color="black"),
                    xshift=15, yshift=10
                )
    
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig

def main_plotly():
    # 경로 설정 (필요시 수정)
    img_dir = paths.YOLO_IMAGES_DIR / "train"
    lbl_dir = paths.YOLO_LABELS_DIR / "train"
    
    # 테스트용: 폴더 내 이미지 8장만 가져오기
    all_images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))[:8]
    
    if not all_images:
        print("이미지를 찾을 수 없습니다.")
        return

    cols = 2
    rows = (len(all_images) + cols - 1) // cols
    
    # 전체를 아우르는 큰 도화지(Subplots) 생성
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[p.name for p in all_images])

    for i, img_path in enumerate(all_images):
        r, c = (i // cols) + 1, (i % cols) + 1
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        
        sub_fig = get_plotly_fig(img_path, lbl_path)
        
        if sub_fig:
            # 1. 이미지 데이터 추가
            for trace in sub_fig.data:
                fig.add_trace(trace, row=r, col=c)
            
            # 2. 박스(shapes) 추가 및 좌표축 연결
            for shape in sub_fig.layout.shapes:
                shape.update(xref=f"x{i+1}", yref=f"y{i+1}")
                fig.add_shape(shape, row=r, col=c)

            # 3. 핵심: ID 텍스트(annotations) 추가 및 좌표축 연결
            for anno in sub_fig.layout.annotations:
                anno.update(xref=f"x{i+1}", yref=f"y{i+1}")
                fig.add_annotation(anno)

    # 전체 레이아웃 설정 (스크롤을 위해 높이를 크게 잡음)
    fig.update_layout(
        height=500 * rows, 
        width=1200, 
        title_text="HealthEat YOLO Viewer (Scroll Enabled)",
        showlegend=False,
        margin=dict(l=50, r=50, b=50, t=100)
    )
    # 이미지 렌더링 최적화
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, autorange=True)
    
    fig.show()

if __name__ == "__main__":
    main_plotly()