# 🧠 Real-Time Object Detection
Detect objects in real-time using YOLOv8 and your webcam!

---

## 🚀 Quick Start

1. Clone the Repository
```bash
git clone https://github.com/jatinn2512/Object-detector.git
cd Object-detector
```

## Install Dependencies

Option A: Using requirements.txt
```bash
pip install -r requirements.txt
```

Option B: Manually Install Each
```bash
pip install opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

## Run the App

```bash
python app.py
```
- A webcam window will open  
- Objects will be detected in real-time with bounding boxes and labels  
- Press **q** to quit the app  


## 🗂️ Project Structure
```bash
Object-detector/
├── app.py
├── requirements.txt
├── README.md
└── yolov8n.pt         # YOLOv8 model file (downloaded automatically if missing)
```

## 🧠 How It Works
- Loads the lightweight **YOLOv8n model** (`yolov8n.pt`)  
- Captures webcam frames using **OpenCV**  
- Detects & classifies objects using **YOLOv8**  
- Draws bounding boxes with:  
  - Object label (e.g., "person")  
  - Confidence score (e.g., 0.91)  
- Provides real-time visual feedback  

## 💡 Sample Output

```bash
Detected: person 0.91
Detected: cell phone 0.85
```
```bash
*(The number shown in `person 0.91` means YOLO is ~91% confident it’s a person.)*  
*(The number shown in `cell phone 0.85` means YOLO is ~85% confident it’s a cell phone.)*
```