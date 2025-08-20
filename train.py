try:
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics import YOLO
    
    print("Starting training...")
    # Load a model
    model = YOLO('yolov8s.pt')  # load a pretrained model
    
    # Train the model
    model.train(
        data="C:/Users/kunal/Downloads/building-detection.v1i.yolov12/building-detection-1/data.yaml",
        epochs=50,
        imgsz=640
    )
except Exception as e:
    print(f"Error occurred: {e}") 