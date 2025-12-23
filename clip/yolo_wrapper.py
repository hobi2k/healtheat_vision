from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weight_path: str):
        self.model = YOLO(weight_path)

    def detect(self, image_path: str, conf: float = 0.25):
        results = self.model(image_path, conf=conf, verbose=False)[0]
        # 결과에서 bbox, score만 뽑고 class_id는 일단 무시
        boxes = results.boxes.xyxy.cpu().tolist()
        scores = results.boxes.conf.cpu().tolist()
        # class_id = results.boxes.cls.cpu().tolist()  # 나중에 쓸 수는 있음
        return boxes, scores
