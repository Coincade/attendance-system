import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from Home import face_rec
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Pantry Logger", layout="wide")
st.title("🍽️ Pantry Seen Logger (Every Detection Counts)")

# Load face embeddings
reference_df = face_rec.retrieve_data("academy:register")

# Load model
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_models', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640))

# Start/stop camera
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

col1, col2 = st.columns(2)
if col1.button("▶️ Start Camera"):
    st.session_state.camera_active = True
if col2.button("⏹️ Stop Camera"):
    st.session_state.camera_active = False

# Log function (log every detection as 'seen')
def log_to_redis(emp_id, name, timestamp):
    log_string = f"{emp_id}@{name}@{timestamp}@seen"
    face_rec.r.lpush("pantry:logs", log_string)

# Identify person
def identify_person(embedding, threshold=0.5):
    embeddings = np.array(reference_df["facial_features"].tolist())
    sims = cosine_similarity(embeddings, embedding.reshape(1, -1)).flatten()
    idx = np.argmax(sims)
    if sims[idx] >= threshold:
        row = reference_df.iloc[idx]
        return row["Name"], row["Employee_id"]
    return "Unknown", "Unknown"

# Run camera
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    st.info("Camera active: logging every face detection.")

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
                log_to_redis(emp_id, name, now)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    st.success("Camera stopped.")
