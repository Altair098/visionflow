import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    camera_id: str
    index: int
    timestamp: float
    image: np.ndarray


class StreamReader:
    """Reads a video source on a background thread and keeps only the latest frame."""

    def __init__(self, camera_id: str, source: str, reconnect_delay: float = 2.0):
        self.camera_id = camera_id
        self.source = source
        self.reconnect_delay = reconnect_delay
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._index = 0
        self.is_file = not source.startswith(("rtsp://", "http://", "https://"))

    def start(self) -> "StreamReader":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def read(self) -> Frame | None:
        with self._lock:
            frame, self._latest = self._latest, None
        return frame

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        cap = self._open()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        while not self._stop.is_set():
            ok, image = cap.read()
            if not ok:
                cap.release()
                if self.is_file:
                    cap = self._open()
                    continue
                time.sleep(self.reconnect_delay)
                cap = self._open()
                continue
            with self._lock:
                self._latest = Frame(self.camera_id, self._index, time.time(), image)
            self._index += 1
            if self.is_file:
                time.sleep(1.0 / fps)
        cap.release()
