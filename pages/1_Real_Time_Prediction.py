import streamlit as st
from Home import face_rec

from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title="Predictions",layout="centered")
st.subheader("Predictions")

# Retreive the data from Redis DB
with st.spinner("Retrieving Data from Redis DB.."):
    redis_face_db = face_rec.retrieve_data('academy:register')
    st.dataframe(redis_face_db)
st.success("Data retrieved successfully")

# time
waitTime = 30 # time in seconds
setTime = time.time()

realtime_pred = face_rec.RealTimePred() # real time prediction class


# Real Time Prediction
# use streamlit webrtc to connect the camera

def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24") # 3 dimension numpy array
    # flipped = img[:,::-1,:] # flip the image

    #operation that you can perform on the array
    pred_img = realtime_pred.face_prediction(
        img,redis_face_db,
        feature_column='facial_features',
        name_role=['Name','Employee_id'],
        thresh=0.5
        )
    
    timenow = time.time()
    diff = timenow - setTime

    if(diff >= waitTime):
        realtime_pred.saveLogs_redis()
        setTime = time.time() # reset time

        print(f"Logs saved at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(key="realtimePrediction",video_frame_callback=video_frame_callback)






