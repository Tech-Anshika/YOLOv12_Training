import sys
import subprocess
import pkg_resources
import torch
from multiprocessing import freeze_support
from pathlib import Path

def main():
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
    else:
        print("No GPU found, using CPU")
        device = 'cpu'

    print("Importing YOLO...")
    from ultralytics import YOLO

    # Load the last trained model
    weights_file = Path('runs/detect/train24/weights/best.pt')
    if weights_file.exists():
        print(f"Loading weights from {weights_file}")
        model = YOLO(str(weights_file))
        
        print(f"Starting new training from previous weights on {device}...")
        try:
            results = model.train(
                data="C:/Users/kunal/Downloads/building-detection.v1i.yolov12/building-detection-1/data.yaml",
                epochs=20,  # Train for 20 more epochs
                imgsz=640,
                verbose=True,
                device=0,
                batch=8,
                workers=2,
                val=False,
                save_period=5,
                patience=0,
                cache=False,
                amp=True,
                overlap_mask=False,
                plots=False,
                resume=False,  # Don't resume, start new training
                model=str(weights_file)  # Use previous weights as starting point
            )
            print("Training completed successfully!")
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
    else:
        print(f"Could not find weights file at {weights_file}")

if __name__ == '__main__':
    freeze_support()
    main() 