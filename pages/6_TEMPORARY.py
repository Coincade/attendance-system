import streamlit as st
from Home import face_rec

st.set_page_config(page_title="DEBUG Pantry Logs")

st.title("🛠️ Pantry Logs Debug Viewer")

logs = face_rec.r.lrange("pantry:logs", 0, -1)
logs = [log.decode() for log in logs]

if not logs:
    st.warning("No logs found in pantry:logs")
else:
    for log in logs[:20]:
        st.text(log)
