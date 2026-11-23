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
python scripts/benchmark.py --weights models/yolo11n.engine --source data/samples/store.mp4
```

## Benchmarks

| Model | Backend | Precision | FPS | Latency (ms) | mAP50-95 |
|---|---|---|---|---|---|
| YOLO11n | PyTorch | FP32 | – | – | – |
| YOLO11n | ONNX Runtime | FP32 | – | – | – |
| YOLO11n | TensorRT | FP16 | – | – | – |
| YOLO11n | TensorRT | INT8 | – | – | – |

## Roadmap

- [x] Multi-stream ingestion, detection, tracking, zone analytics, API
- [x] ONNX / TensorRT export and benchmarking scripts
- [ ] Fine-tuned person detector on retail footage
- [ ] Uniform and hygiene compliance classifier
- [ ] NVIDIA DeepStream pipeline and Jetson deployment
- [ ] Drift monitoring and automated retraining

## License

MIT
