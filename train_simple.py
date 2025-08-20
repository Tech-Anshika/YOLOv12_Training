import sys
import subprocess
import pkg_resources
import torch
from multiprocessing import freeze_support

required = {'ultralytics', 'torch', 'numpy'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

def main():
    if missing:
        print("Installing missing packages...")
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

    print("Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Clear CUDA cache
        torch.cuda.empty_cache()
    else:
        print("No GPU found, using CPU")
        device = 'cpu'

    print("Importing YOLO...")
    from ultralytics import YOLO

    print("Loading model...")
    model = YOLO("yolov8s.pt")

    print(f"Starting training on {device}...")
    try:
        results = model.train(
            data="C:/Users/kunal/Downloads/building-detection.v1i.yolov12/building-detection-1/data.yaml",
            epochs=20,
            imgsz=640,
            verbose=True,
            device=0,  # Use GPU
            batch=8,  # Reduced batch size for lower memory usage
            workers=2,  # Reduced workers
            val=False,  # Disable validation during training
            save_period=5,  # Save model every 5 epochs
            patience=0,  # Disable early stopping
            cache=False,  # Disable image caching
            amp=True,  # Enable automatic mixed precision
            overlap_mask=False,  # Reduce memory usage
            plots=False,  # Disable plotting to save memory
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        print("Python version:", sys.version)
        print("Packages:")
        for pkg in pkg_resources.working_set:
            print(f"{pkg.key}=={pkg.version}")

if __name__ == '__main__':
    freeze_support()
    main() 