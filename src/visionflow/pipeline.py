import logging
import threading
import time
from dataclasses import asdict, dataclass, field

from visionflow.analytics.zones import Zone, ZoneState
from visionflow.api.metrics import FPS, INFERENCE_LATENCY, ZONE_COUNT, ZONE_DWELL
from visionflow.config import AppConfig, CameraConfig
from visionflow.inference.detector import Detector
from visionflow.ingest.stream import StreamReader

log = logging.getLogger(__name__)


@dataclass
class CameraState:
    camera_id: str
    online: bool = False
    fps: float = 0.0
    people: int = 0
    zones: list[ZoneState] = field(default_factory=list)
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class CameraPipeline:
    def __init__(self, cam: CameraConfig, app: AppConfig):
        self.cam = cam
        self.reader = StreamReader(cam.id, cam.source)
        self.detector = Detector(app.model, app.tracker)
        self.zones = [Zone(z) for z in cam.zones]
        self.state = CameraState(cam.id)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"pipeline-{cam.id}")

    def start(self) -> None:
        self.reader.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.reader.stop()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            frame = self.reader.read()
            if frame is None:
                time.sleep(0.005)
                continue

            t0 = time.perf_counter()
            tracks = self.detector.track(frame.image)
            INFERENCE_LATENCY.labels(self.cam.id).observe(time.perf_counter() - t0)

            zone_states = [z.update(tracks, frame.timestamp) for z in self.zones]
            for zs in zone_states:
                ZONE_COUNT.labels(self.cam.id, zs.name).set(zs.count)
                ZONE_DWELL.labels(self.cam.id, zs.name).set(zs.avg_dwell_s)

            now = time.perf_counter()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            FPS.labels(self.cam.id).set(fps)

            self.state = CameraState(
                camera_id=self.cam.id,
                online=True,
                fps=round(fps, 1),
                people=len(tracks),
                zones=zone_states,
                updated_at=frame.timestamp,
            )


class PipelineManager:
    def __init__(self, cfg: AppConfig):
        self.pipelines = {c.id: CameraPipeline(c, cfg) for c in cfg.cameras}

    def start(self) -> None:
        for p in self.pipelines.values():
            p.start()
            log.info("Started camera %s", p.cam.id)

    def stop(self) -> None:
        for p in self.pipelines.values():
            p.stop()

    def states(self) -> dict[str, dict]:
        return {cid: p.state.to_dict() for cid, p in self.pipelines.items()}
