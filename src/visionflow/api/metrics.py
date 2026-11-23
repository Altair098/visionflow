from prometheus_client import Gauge, Histogram

FPS = Gauge("visionflow_fps", "Processed frames per second", ["camera"])
INFERENCE_LATENCY = Histogram(
    "visionflow_inference_seconds",
    "Detection and tracking latency per frame",
    ["camera"],
    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5),
)
ZONE_COUNT = Gauge("visionflow_zone_people", "People currently inside a zone", ["camera", "zone"])
ZONE_DWELL = Gauge("visionflow_zone_avg_dwell_seconds", "Average dwell time in a zone", ["camera", "zone"])
