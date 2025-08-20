import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    print("Loading model weights...")
    weights_file = Path('runs/detect/train24/weights/best.pt')  # Using best weights for evaluation

    if weights_file.exists():
        print(f"Loading weights from {weights_file}")
        model = YOLO(str(weights_file))
        
        # Run validation
        print("Running validation...")
        metrics = model.val(
            data="C:/Users/kunal/Downloads/building-detection.v1i.yolov12/building-detection-1/data.yaml",
            imgsz=640,
            batch=1,  # Single batch size
            workers=0,  # No additional workers
            plots=True  # Save validation plots
        )
        
        # Print metrics
        print("\nValidation Results:")
        print(f"mAP50: {metrics.box.map50:.3f}")  # mAP at IoU 0.50
        print(f"mAP50-95: {metrics.box.map:.3f}")  # mAP at IoU 0.50-0.95
        print(f"Precision: {metrics.box.mp:.3f}")
        print(f"Recall: {metrics.box.mr:.3f}")
    else:
        print("Weights file not found at:", weights_file)

if __name__ == '__main__':
    main() 