import argparse
import json
import threading
import time

import cv2
from ultralytics import YOLO


def load_frames(source: str, n: int) -> list:
    cap = cv2.VideoCapture(source)
    frames = []
    while len(frames) < n:
        ok, img = cap.read()
        if not ok:
            if not frames:
                raise RuntimeError(f"No frames read from {source}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(img)
    cap.release()
    return frames


def run(weights: str, frames: list, streams: int, imgsz: int, device: str) -> dict:
    kwargs = dict(persist=True, tracker="bytetrack.yaml", imgsz=imgsz, device=device, classes=[0], verbose=False)

    models = []
    for _ in range(streams):
        model = YOLO(weights, task="detect")
        for img in frames[:10]:
            model.track(img, **kwargs)
        models.append(model)

    results: list = [None] * streams
    errors: list = []
    barrier = threading.Barrier(streams)

    def worker(idx: int) -> None:
        try:
            barrier.wait(timeout=120)
            t0 = time.perf_counter()
            for img in frames:
                models[idx].track(img, **kwargs)
            results[idx] = len(frames) / (time.perf_counter() - t0)
        except Exception as e:
            errors.append(e)
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"{streams} streams failed: {errors[0]!r}")

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
    parser.add_argument("--output", default="multistream.json")
    args = parser.parse_args()

    frames = load_frames(args.source, args.frames)
    report = []
    for n in args.streams:
        try:
            r = run(args.weights, frames, n, args.imgsz, args.device)
        except Exception as e:
            print(f"Stopping at {n} streams: {e}")
            break
        r["realtime"] = r["per_stream_fps"] >= args.target_fps
        report.append(r)
        print(json.dumps(r), flush=True)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
