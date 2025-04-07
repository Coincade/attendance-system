import streamlit as st
import pandas as pd
import numpy as np
import cv2
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from Home import face_rec

st.set_page_config(page_title="Real-Time Prediction", layout="wide")
st.title("🧠 Real-Time Attendance Logging")

# Load registered data
reference_df = face_rec.retrieve_data("academy:register")
st.success("Employee embeddings loaded.")

# Load model
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_models', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640))

# Start/Stop camera state
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# Track last seen time to avoid duplicates
if "last_logged" not in st.session_state:
    st.session_state.last_logged = {}

# UI Buttons
col1, col2 = st.columns(2)
if col1.button("▶️ Start Camera"):
    st.session_state.camera_active = True
if col2.button("⏹️ Stop Camera"):
    st.session_state.camera_active = False

# Logging to Redis
def log_to_redis(emp_id, name, timestamp):
    if emp_id and name and name != "Unknown":
        log_string = f"{name}@{emp_id}@{timestamp}"
        face_rec.r.lpush("attendance:logs", log_string)
        print(f"[LOGGED] {log_string}")

# Identify person using face embedding
def identify_person(embedding, threshold=0.5):
    embeddings = np.array(reference_df["facial_features"].tolist())
    sims = cosine_similarity(embeddings, embedding.reshape(1, -1)).flatten()
    idx = np.argmax(sims)
    if sims[idx] >= threshold:
        row = reference_df.iloc[idx]
        return row["Name"], row["Employee_id"]
    return "Unknown", "Unknown"

# Camera logic
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    st.info("Camera active. Detecting faces...")

    while st.session_state.camera_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceapp.get(rgb_frame)
        now = datetime.now()

        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embedding = res['embedding']
            name, emp_id = identify_person(embedding)

            if name != "Unknown":
                last_seen = st.session_state.last_logged.get(emp_id)
                if not last_seen or (now - last_seen).total_seconds() > 60:
                    log_to_redis(emp_id, name, now.strftime("%Y-%m-%d %H:%M:%S"))
                    st.session_state.last_logged[emp_id] = now

            # Draw box + name
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    st.success("Camera stopped.")
