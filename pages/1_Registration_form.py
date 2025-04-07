import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Registration Form", layout="centered")
st.subheader("📝 Registration Form")

# Initialize registration object
register_form = face_rec.RegisterationForm()

# Store webcam state
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# Input fields
person_name = st.text_input("👤 Name", placeholder="First Name and Last Name")
employee_id = st.text_input("🆔 Employee ID", placeholder="Enter your unique ID")

# Start/Stop camera buttons
col1, col2 = st.columns(2)
if col1.button("▶️ Start Camera"):
    st.session_state.camera_active = True
if col2.button("⏹️ Stop Camera"):
    st.session_state.camera_active = False

# Display message
if st.session_state.camera_active:
    st.info("Camera active. Look at the camera to collect face samples...")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while st.session_state.camera_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        reg_img, embedding = register_form.get_embedding(frame_rgb)

        if embedding is not None:
            # Save embeddings to file
            with open('face_embedding.txt', mode='ab') as f:
                np.savetxt(f, embedding)

        stframe.image(reg_img, channels="BGR")

    cap.release()
    st.success("Camera stopped.")

# Save to Redis
if st.button("✅ Submit Registration"):
    return_value = register_form.save_data_in_redis_db(employee_id, person_name)

    if return_value == True:
        st.success(f"{person_name} registered successfully!")
    elif return_value == "name_false":
        st.error("Name cannot be empty.")
    elif return_value == "employee_id_false":
        st.error("Employee ID cannot be empty.")
    elif return_value == "file_false":
        st.error("No face embeddings collected. Please start the camera first.")
    else:
        st.error("Something went wrong.")
