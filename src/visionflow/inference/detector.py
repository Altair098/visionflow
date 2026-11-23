from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from visionflow.config import ModelConfig


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


class Detector:
    """YOLO detector with built-in ByteTrack tracking. Accepts .pt, .onnx, or .engine weights."""

    def __init__(self, cfg: ModelConfig, tracker: str = "bytetrack.yaml"):
        self.cfg = cfg
        self.tracker = tracker
        self.model = YOLO(cfg.weights, task="detect")

    def track(self, image: np.ndarray) -> list[Track]:
        result = self.model.track(
            image,
            persist=True,
            tracker=self.tracker,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            classes=self.cfg.classes,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        return [
            Track(int(tid), tuple(map(float, xyxy)), float(conf), int(cls))
            for tid, xyxy, conf, cls in zip(
                boxes.id.tolist(), boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            )
        ]
