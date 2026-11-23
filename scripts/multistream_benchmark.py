import argparse
import json
import threading
import time

import cv2
from ultralytics import YOLO


def worker(weights: str, source: str, frames: int, imgsz: int, device: str, results: list, idx: int, barrier):
    model = YOLO(weights, task="detect")
    cap = cv2.VideoCapture(source)
    images = []
    while len(images) < frames:
        ok, img = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        images.append(img)
    cap.release()

    for img in images[:10]:
        model.track(img, persist=True, imgsz=imgsz, device=device, classes=[0], verbose=False)

    barrier.wait()
    t0 = time.perf_counter()
    for img in images:
        model.track(img, persist=True, imgsz=imgsz, device=device, classes=[0], verbose=False)
    results[idx] = len(images) / (time.perf_counter() - t0)


def run(weights: str, source: str, streams: int, frames: int, imgsz: int, device: str) -> dict:
    results: list = [None] * streams
    barrier = threading.Barrier(streams)
    threads = [
        threading.Thread(target=worker, args=(weights, source, frames, imgsz, device, results, i, barrier))
        for i in range(streams)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(r is None for r in results):
        raise RuntimeError("A stream worker failed; see the traceback above")
    return {
        "streams": streams,
        "per_stream_fps": round(min(results), 1),
        "total_fps": round(sum(results), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure detection + tracking throughput across concurrent streams")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--target-fps", type=float, default=15.0)
    args = parser.parse_args()

    report = [run(args.weights, args.source, n, args.frames, args.imgsz, args.device) for n in args.streams]
    for r in report:
        r["realtime"] = r["per_stream_fps"] >= args.target_fps
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
