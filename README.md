# 🏙️ Building Detection using YOLOv12

## 📌 Project Overview
This project aims to detect **buildings** in images using **YOLOv12** object detection.  
The objective is to create accurate and robust models for building recognition in aerial and urban imagery.  

We experimented with **three models**:
1. Roboflow 3.0 Object Detection (Fast)  
2. Roboflow YOLOv12 (Fast)  
3. Locally trained YOLOv12 (Ultralytics)  

---

## 📂 Dataset
- **Source**: Custom annotated dataset from **Roboflow**.  
- **Classes**: `Building`  
- **Total Images**: 2,349  
- **Total Instances**: 31,256  
- **Format**: YOLOv12 (`.yaml`, `.txt` annotations)  
- **Splits**: `train`, `test`, `valid`  

---

## 🤖 Models & Performance

### 🔹 Model 1 (Roboflow - v1)
- **Framework**: Roboflow 3.0 Object Detection (Fast)  
- **mAP@50**: `77.5%`  
- **Precision**: `78.2%`  
- **Recall**: `69.8%`  
- **Date Trained**: 16 Aug 2025  

---

### 🔹 Model 2 (Roboflow - YOLOv12)
- **Framework**: YOLOv12 Object Detection (Fast)  
- **mAP@50**: **79.1%**  
- **Precision**: **79.1%**  
- **Recall**: **71.7%**  
- **Date Trained**: 18 Aug 2025  

---

### 🔹 Model 3 (Local Training - YOLOv12)
- **Framework**: YOLOv12 (Ultralytics, PyTorch)  
- **Dataset**: Same Roboflow dataset, trained locally  
- **Images**: `2349`  
- **Instances**: `31,256`  
- **Precision**: `77.9%`  
- **Recall**: `70.2%`  
- **mAP@50**: `77.1%`  
- **mAP@50-95**: `49.9%`  
- **Results Directory**: `runs/train/exp/`  

---

## 📊 Model Comparison

| Model | mAP@50 | Precision | Recall | mAP@50-95 |
|-------|--------|-----------|--------|-----------|
| Model 1 (Roboflow v1) | 77.5% | 78.2% | 69.8% | – |
| Model 2 (Roboflow YOLOv12) | **79.1%** | **79.1%** | **71.7%** | – |
| Model 3 (Local YOLOv12) | 77.1% | 77.9% | 70.2% | 49.9% |

✅ **Best Model**: **Model 2 (Roboflow YOLOv12)** due to highest mAP@50, precision, and recall.  
📌 However, **Model 3** includes **mAP@50-95**, making it useful for strict IoU benchmarks.  

---

## ⚙️ Training (Local)

```bash
# Clone YOLOv12
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics

# Install dependencies
pip install -r requirements.txt

# Train YOLOv12
yolo detect train data=building-detection.yaml model=yolov12n.pt epochs=50 imgsz=640

##  🧪 Testing & Inference

# Run inference on an image
yolo detect predict model=runs/train/exp/weights/best.pt source=sample.jpg

# Run inference on a video
yolo detect predict model=runs/train/exp/weights/best.pt source=video.mp4

