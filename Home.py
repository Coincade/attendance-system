import streamlit as st


st.set_page_config(page_title="Attendance System",layout="wide")

st.header("Attendance System using Face Recognition")

with st.spinner("Loading Models and Connection to DB.."):
    import face_rec
st.success("Models and Connection to DB loaded successfully")
