# 🏙️ Building Detection using YOLOv12
# ------------------------------------

# 📌 Project Overview
# This project aims to detect buildings in images using YOLOv12 object detection.
# We experimented with three models:
#   1. Roboflow v1 Object Detection (Fast)
#   2. Roboflow YOLOv12 (Fast)
#   3. Locally trained YOLOv12 (Ultralytics)

# 📂 Dataset
# - Source: Custom annotated dataset from Roboflow
# - Classes: Building
# - Total Images: 2349
# - Total Instances: 31,256
# - Format: YOLOv12 (.yaml, .txt annotations)
# - Splits: train, test, valid

# 🤖 Models & Performance
# ------------------------
# 🔹 Model 1 (Roboflow v1)
# - Framework: Roboflow 3.0 Object Detection (Fast)
# - mAP@50: 77.5%
# - Precision: 78.2%
# - Recall: 69.8%
# - Date Trained: 16 Aug 2025
#
# 🔹 Model 2 (Roboflow YOLOv12)
# - Framework: YOLOv12 Object Detection (Fast)
# - mAP@50: 79.1%
# - Precision: 79.1%
# - Recall: 71.7%
# - Date Trained: 18 Aug 2025
#
# 🔹 Model 3 (Local Training YOLOv12)
# - Framework: YOLOv12 (Ultralytics, PyTorch)
# - Dataset: Same Roboflow dataset, trained locally
# - Images: 2349
# - Instances: 31,256
# - Precision: 77.9%
# - Recall: 70.2%
# - mAP@50: 77.1%
# - mAP@50-95: 49.9%
# - Results: runs/train/exp/

# 📊 Model Comparison Table
# --------------------------
# | Model                  | mAP@50 | Precision | Recall | mAP@50-95 |
# |-------------------------|--------|-----------|--------|-----------|
# | Model 1 (Roboflow v1)   | 77.5%  | 78.2%     | 69.8%  | -         |
# | Model 2 (Roboflow YOLOv12) | 79.1% | 79.1% | 71.7%  | -         |
# | Model 3 (Local YOLOv12) | 77.1%  | 77.9%     | 70.2%  | 49.9%     |
#
# ✅ Best Model: Model 2 (Roboflow YOLOv12)
# 📌 Note: Model 3 includes mAP@50-95, useful for strict IoU benchmarks.

# ================================
# 1. Clone & Install
# ================================
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -r requirements.txt

# ================================
# 2. Train YOLOv12 (Local)
# ================================
yolo detect train data=building-detection.yaml model=yolov12n.pt epochs=50 imgsz=640

# ================================
# 3. Run Inference (Testing)
# ================================
# On a single image
yolo detect predict model=runs/train/exp/weights/best.pt source=sample.jpg

# On a video
yolo detect predict model=runs/train/exp/weights/best.pt source=video.mp4

# Results will be saved in runs/detect/predict/

# ================================
# 4. Deployment
# ================================
# --- Roboflow API Example ---
python - <<EOF
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("building-detection")
model = project.version(1).model
prediction = model.predict("your_image.jpg").json()
print(prediction)
EOF

# --- Local FastAPI/Streamlit deployment can also be done ---

# ================================
# 📜 License
# ================================
# This project is licensed under AGPL-3.0.

# ================================
# 👩‍💻 Contributor
# ================================
# Anshika Tyagi
