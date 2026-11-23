import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX or TensorRT")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--format", choices=["onnx", "engine"], default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 precision")
    parser.add_argument("--int8", action="store_true", help="INT8 precision (TensorRT)")
    parser.add_argument("--data", default="coco8.yaml", help="Calibration data for INT8")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights)
    path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        data=args.data if args.int8 else None,
        dynamic=args.dynamic,
        simplify=True,
    )
    print(f"Exported: {path}")


if __name__ == "__main__":
    main()
