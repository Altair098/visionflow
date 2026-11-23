# VisionFlow

Real-time multi-stream video analytics. VisionFlow ingests RTSP or file video streams, detects and tracks people, and turns raw footage into operational metrics: queue length, zone occupancy, dwell time, and compliance events.

## Features

- **Multi-stream ingestion**: concurrent RTSP / video file readers with automatic reconnect and frame dropping under load
- **Detection and tracking**: Ultralytics YOLO with ByteTrack for persistent person IDs across frames
- **Zone analytics**: polygon zones per camera for queue length, occupancy, and dwell time
- **Compliance checks**: classifier for uniform and hygiene compliance on tracked people
- **Optimized inference**: export to ONNX and TensorRT (FP16 / INT8) with a benchmarking harness
- **Serving and monitoring**: FastAPI microservice with REST endpoints and Prometheus metrics, packaged in Docker

## Architecture

```
RTSP / video ──> StreamReader (thread per camera) ──> Detector (YOLO / ONNX / TensorRT)
                                                          │
                                                          ▼
                                                   Tracker (ByteTrack)
                                                          │
                                                          ▼
                                    ZoneAnalytics (queue, occupancy, dwell time)
                                                          │
                                                          ▼
                                     FastAPI  /cameras  /metrics (Prometheus)
```

## Project Structure

```
visionflow/
├── configs/              # camera and zone definitions
├── src/visionflow/
│   ├── ingest/           # multi-stream video readers
│   ├── inference/        # detector backends (PyTorch, ONNX, TensorRT)
│   ├── tracking/         # tracker wrapper
│   ├── analytics/        # zone logic, queue length, dwell time
│   ├── api/              # FastAPI service and Prometheus metrics
│   └── pipeline.py       # per-camera processing loop
├── scripts/              # export, benchmark, run
├── tests/
└── docker/
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run.py --config configs/cameras.yaml
```

The API runs at `http://localhost:8000`:

| Endpoint | Description |
|---|---|
| `GET /health` | Service status |
| `GET /cameras` | Live metrics for every camera |
| `GET /cameras/{id}` | Metrics for one camera |
| `GET /metrics` | Prometheus metrics |

### Docker

```bash
docker build -f docker/Dockerfile -t visionflow .
docker run --gpus all -p 8000:8000 -v $(pwd)/configs:/app/configs visionflow
```

## Model Optimization

```bash
python scripts/export.py --weights yolo11n.pt --format onnx
python scripts/export.py --weights yolo11n.pt --format engine --half
python scripts/benchmark.py --weights yolo11n.engine --source data/samples/test.mp4 --data coco_val.yaml
python scripts/multistream_benchmark.py --weights yolo11n.engine --source data/samples/test.mp4 --streams 1 2 4 8
```

## Benchmarks

Reproduce with [`notebooks/benchmark.ipynb`](notebooks/benchmark.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Altair098/visionflow/blob/main/notebooks/benchmark.ipynb). Accuracy is measured on COCO val2017.

Measured on a single NVIDIA T4 (Kaggle), 640x640 input, batch size 1. mAP on COCO val2017 (5,000 images).

| Model | Backend | Precision | FPS | Latency p50 (ms) | Latency p95 (ms) | mAP50-95 | Speedup |
|---|---|---|---|---|---|---|---|
| YOLO11n | PyTorch | FP32 | 91.0 | 10.83 | 12.19 | 0.3889 | 1.0x |
| YOLO11n | ONNX Runtime | FP32 | 110.0 | 9.01 | 9.67 | 0.3878 | 1.2x |
| YOLO11n | TensorRT | FP16 | 193.5 | 4.98 | 6.54 | 0.3878 | 2.1x |
| YOLO11n | TensorRT | INT8 | 196.2 | 4.87 | 6.84 | 0.3705 | 2.2x |

### Multi-stream detection + tracking (TensorRT FP16, ByteTrack)

| Streams | Per-stream FPS | Total FPS | Real-time (>=15 FPS) |
|---|---|---|---|
| 1 | 102.3 | 102.3 | yes |
| 2 | 67.0 | 134.1 | yes |
| 4 | 25.4 | 101.9 | yes |
| 8 | 10.9 | 87.5 | no |
| 12 | 7.2 | 86.4 | no |
| 16 | 5.5 | 87.7 | no |

### Findings

- **TensorRT FP16 is the production choice**: 2.1x faster than PyTorch with a 0.3% relative mAP drop (0.3889 to 0.3878).
- **INT8 is not worth it for YOLO11n**: only 1% faster than FP16 but loses 4.5% relative mAP. The nano model is too small for INT8 to pay off; it should matter more for larger models.
- **4 real-time streams per T4**: throughput peaks at 2 streams (134 FPS total) and degrades beyond 4, where per-stream Python threads compete for the GIL and CPU-side preprocessing and tracking become the bottleneck rather than the GPU. Batched multi-stream inference and DeepStream are the next steps to scale past this.

## Roadmap

- [x] Multi-stream ingestion, detection, tracking, zone analytics, API
- [x] ONNX / TensorRT export and benchmarking (2.1x speedup, 4 real-time streams per T4)
- [ ] Fine-tuned person detector on retail footage
- [ ] Uniform and hygiene compliance classifier
- [ ] Batched multi-stream inference to scale past 4 streams per GPU
- [ ] NVIDIA DeepStream pipeline and Jetson deployment
- [ ] Drift monitoring and automated retraining

## License

MIT
