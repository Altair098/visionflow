import argparse
import json
import statistics
import time

import cv2
from ultralytics import YOLO


def load_frames(source: str, n: int) -> list:
    cap = cv2.VideoCapture(source)
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {source}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark detector throughput and latency")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--data", help="Dataset YAML for mAP evaluation")
    args = parser.parse_args()

    model = YOLO(args.weights, task="detect")
    frames = load_frames(args.source, args.frames)

    for f in frames[: args.warmup]:
        model.predict(f, imgsz=args.imgsz, device=args.device, verbose=False)

    latencies = []
    for f in frames:
        t0 = time.perf_counter()
        model.predict(f, imgsz=args.imgsz, device=args.device, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    report = {
        "weights": args.weights,
        "frames": len(latencies),
        "fps": round(1000 / statistics.mean(latencies), 1),
        "latency_ms_mean": round(statistics.mean(latencies), 2),
        "latency_ms_p50": round(latencies[len(latencies) // 2], 2),
        "latency_ms_p95": round(latencies[int(len(latencies) * 0.95) - 1], 2),
    }

    if args.data:
        metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
        report["map50"] = round(float(metrics.box.map50), 4)
        report["map50_95"] = round(float(metrics.box.map), 4)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
