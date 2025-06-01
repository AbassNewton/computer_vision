# app.py
import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
import numpy as np
from ultralytics import YOLO
from sort import Sort
import matplotlib.pyplot as plt
import seaborn as sns

# Liste des classes COCO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

def detect_and_track(video_path, model_path, output_csv, target_class_name=None):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    tracker = Sort()
    log = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = video_path.replace(".mp4", "_out.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    class_id_target = COCO_CLASSES.index(target_class_name) if target_class_name else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if class_id_target is not None and cls_id != class_id_target:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            if conf > 0.5:
                detections.append([x1, y1, x2, y2, conf])

        if detections:
            tracked = tracker.update(np.array(detections)[:, :4])
            for *xyxy, obj_id in tracked:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f'ID {int(obj_id)}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                log.append([cap.get(cv2.CAP_PROP_POS_MSEC), int(obj_id), x1, y1, x2, y2])

        out.write(frame)

    cap.release()
    out.release()
    df = pd.DataFrame(log, columns=["timestamp_ms", "id", "x1", "y1", "x2", "y2"])
    df.to_csv(output_csv, index=False)
    return out_path, df

# --- Interface Streamlit ---
st.title("🎯 Détection et Suivi en Temps Réel")

video_file = st.file_uploader("📥 Uploader une vidéo MP4", type=["mp4"])
model_file = st.file_uploader("📦 Uploader un modèle YOLOv8 (.pt) (optionnel)", type=["pt"])

# 🔁 Si aucun modèle fourni, utiliser le modèle par défaut
default_model_path = os.path.join(os.path.dirname(__file__), "best.pt")

selected_class = st.selectbox("🧠 Choisir une classe à détecter :", COCO_CLASSES)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
        temp_vid.write(video_file.read())
        video_path = temp_vid.name

    if model_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model:
            temp_model.write(model_file.read())
            model_path = temp_model.name
    else:
        model_path = default_model_path

    if st.button("🚀 Lancer la détection et le tracking"):
        output_csv = "tracking_log.csv"
        output_vid, df = detect_and_track(video_path, model_path, output_csv, selected_class)

        st.success("✅ Traitement terminé !")
        st.video(output_vid)

        with open(output_vid, "rb") as f:
            st.download_button("📥 Télécharger la vidéo annotée", f.read(), file_name=os.path.basename(output_vid), mime="video/mp4")

        st.subheader("📊 Histogramme des objets trackés")
        fig1, ax1 = plt.subplots()
        df['id'].value_counts().sort_index().plot(kind='bar', ax=ax1)
        ax1.set_xlabel("ID")
        ax1.set_ylabel("Nombre de frames")
        ax1.set_title("Occurrences par ID")
        st.pyplot(fig1)

        fig1.savefig("tracking_histogram.png")
        with open("tracking_histogram.png", "rb") as f:
            st.download_button("📈 Télécharger l'histogramme", f.read(), file_name="tracking_histogram.png", mime="image/png")

        st.subheader("🔥 Carte thermique des positions")
        heatmap_data = df[['x1', 'y1']].copy()
        heatmap_data['x1'] = heatmap_data['x1'] // 50
        heatmap_data['y1'] = heatmap_data['y1'] // 50
        heatmap_pivot = heatmap_data.pivot_table(index='y1', columns='x1', aggfunc='size', fill_value=0)
        fig2, ax2 = plt.subplots()
        sns.heatmap(heatmap_pivot, cmap='Reds', ax=ax2)
        st.pyplot(fig2)

        st.subheader("🗂️ Télécharger les logs CSV")
        st.download_button("📄 Télécharger CSV", df.to_csv(index=False), file_name="tracking_log.csv", mime="text/csv")
