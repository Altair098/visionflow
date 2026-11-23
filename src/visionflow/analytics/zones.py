from dataclasses import dataclass, field

import cv2
import numpy as np

from visionflow.config import ZoneConfig
from visionflow.inference.detector import Track


@dataclass
class ZoneState:
    name: str
    type: str
    count: int = 0
    avg_dwell_s: float = 0.0
    max_dwell_s: float = 0.0
    total_visits: int = 0


@dataclass
class Zone:
    cfg: ZoneConfig
    polygon: np.ndarray = field(init=False)
    entered_at: dict[int, float] = field(default_factory=dict)
    completed_dwells: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.polygon = np.array(self.cfg.polygon, dtype=np.int32)

    def contains(self, point: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def update(self, tracks: list[Track], now: float) -> ZoneState:
        inside = {t.track_id for t in tracks if self.contains(t.foot_point)}

        for tid in inside - self.entered_at.keys():
            self.entered_at[tid] = now
        for tid in self.entered_at.keys() - inside:
            self.completed_dwells.append(now - self.entered_at.pop(tid))
        self.completed_dwells = self.completed_dwells[-500:]

        current = [now - t0 for t0 in self.entered_at.values()]
        dwells = self.completed_dwells + current
        return ZoneState(
            name=self.cfg.name,
            type=self.cfg.type,
            count=len(inside),
            avg_dwell_s=round(float(np.mean(dwells)), 2) if dwells else 0.0,
            max_dwell_s=round(max(current), 2) if current else 0.0,
            total_visits=len(self.completed_dwells) + len(current),
        )
