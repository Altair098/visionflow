from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    weights: str = "yolo11n.pt"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: int | str = 0
    classes: list[int] = Field(default_factory=lambda: [0])


class ZoneConfig(BaseModel):
    name: str
    type: Literal["queue", "occupancy"]
    polygon: list[tuple[int, int]]


class CameraConfig(BaseModel):
    id: str
    source: str
    zones: list[ZoneConfig] = Field(default_factory=list)


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    tracker: str = "bytetrack.yaml"
    api: ApiConfig = Field(default_factory=ApiConfig)
    cameras: list[CameraConfig]


def load_config(path: str | Path) -> AppConfig:
    with open(path) as f:
        return AppConfig(**yaml.safe_load(f))
