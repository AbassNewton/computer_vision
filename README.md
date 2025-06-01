# Real-time Object Detection & Tracking

This project detects and tracks specific objects (e.g., persons, vehicles) in real-time video using YOLOv8 and SORT, and provides a web-based interface via Streamlit.

## Features
- YOLOv8 detection (fine-tuned)
- SORT tracking with unique object IDs
- Streamlit UI for uploading and visualizing results
- Tracking logs export to CSV

## How to Use

### 1. Fine-tune YOLOv8
Use `ultralytics` to train on your annotated dataset.

```bash
yolo detect train model=yolov8n.pt data=data/data.yaml epochs=50 imgsz=640
