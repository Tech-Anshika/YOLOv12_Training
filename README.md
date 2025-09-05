# 🏙️ Building Detection using 

## 📌 Project Overview
This project detects **buildings** in images using **YOLOv12**.  
We experimented with three models:
1. **Roboflow v1 Object Detection (Fast)**
2. **Roboflow YOLOv12 (Fast)**
3. **Locally trained YOLOv12 (Ultralytics)**

---

## 📂 Dataset
- **Source:** Custom annotated dataset from Roboflow  
- **Classes:** Building  
- **Total Images:** 2349  
- **Total Instances:** 31,256  
- **Format:** YOLOv12 (`.yaml`, `.txt` annotations)  
- **Splits:** Train, Test, Valid  

---

## 🤖 Models & Performance

### 🔹 Model 1 (Roboflow v1)
- Framework: Roboflow 3.0 Object Detection (Fast)  
- mAP@50: **77.5%**  
- Precision: **78.2%**  
- Recall: **69.8%**  
- Date Trained: *16 Aug 2025*
- **Roboflow v1:** [Click here](https://app.roboflow.com/image-detection-f0jrc/building-detection-qkcv9/models)
 


### 🔹 Model 2 (Roboflow YOLOv12)
- Framework: YOLOv12 Object Detection (Fast)  
- mAP@50: **79.1%**  
- Precision: **79.1%**  
- Recall: **71.7%**  
- Date Trained: *18 Aug 2025*
- **Roboflow YOLOv12:** [Click here](https://app.roboflow.com/my-projects-jk4cn/building-detection-sr9ws/models)


### 🔹 Model 3 (Local Training YOLOv12)
- Framework: YOLOv12 (Ultralytics, PyTorch)  
- Dataset: Same Roboflow dataset, trained locally  
- Images: 2349  
- Instances: 31,256  
- Precision: **77.9%**  
- Recall: **70.2%**  
- mAP@50: **77.1%**  
- mAP@50-95: **49.9%**  
- Results: Stored in `runs/train/exp/`

---

## 📊 Model Comparison Table

| Model                     | mAP@50 | Precision | Recall | mAP@50-95 |
|----------------------------|--------|-----------|--------|-----------|
| Roboflow v1                | 77.5%  | 78.2%     | 69.8%  | -         |
| Roboflow YOLOv12 (Best ) | 79.1%  | 79.1%     | 71.7%  | -         |
| Local YOLOv12              | 77.1%  | 77.9%     | 70.2%  | 49.9%     |

✅ **Best Model: Roboflow YOLOv12**  
📌 *Local model also provides mAP@50-95, useful for stricter IoU benchmarks.*

---

## 🖼️ Sample Predictions

### Roboflow YOLOv12 Predictions
![prediction_output](https://github.com/user-attachments/assets/56e9137b-0af2-402e-9fda-b96d2756d126)  
![prediction_output 2](https://github.com/user-attachments/assets/170681fd-f063-4da1-971f-e83db073e48e)  
  

---
 
## ⚙️ Installation & Training

```bash
# Clone Ultralytics repo
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -r requirements.txt

# Train YOLOv12 locally
yolo detect train data=building-detection.yaml model=yolov12n.pt epochs=50 imgsz=640
```
---
## 🧪 Testing & Inference

```bash
### Run inference on an image
yolo detect predict model=runs/train/exp/weights/best.pt source=sample.jpg

### Run inference on a video
yolo detect predict model=runs/train/exp/weights/best.pt source=video.mp4
```
---
## 📜 License
This project is licensed under AGPL-3.0.

---
## 👩‍💻 Contributor

- **[Anshika Tyagi](https://github.com/Tech-Anshika)**
  


