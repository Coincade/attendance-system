import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title="Registration Form",layout="centered")
st.subheader("Registration Form")

# Initialize the registration form
register_form = face_rec.RegisterationForm()

# 1. Collect Person name and role
# form
person_name = st.text_input(label="Name", placeholder="First Name and Last Name")
# role = st.selectbox(label="Select Your Role", options=('Employee', 'Admin'))
role = 'Employee'

# 2. Collect facial embedding of person
def video_callback_function(frame):
    img = frame.to_ndarray(format="bgr24")
    reg_img, embedding = register_form.get_embedding(img)
    # two step process
    # 1st step: save data to local computer .txt
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)
    # 2nd step: save data to redis db
    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")

webrtc_streamer(key="registeration", video_frame_callback=video_callback_function)

# 3. Save the data in Redis DB

if st.button('Submit'):
   return_value = register_form.save_data_in_redis_db(person_name, role)
   if return_value == True:
       st.success(f'{person_name} Registered Successfully')
   elif return_value == 'name_false':
        st.error('Name cannot be empty')
   elif return_value == 'file_false':
        st.error('Please collect the facial embedding first')
   else:
        st.error('Something went wrong')
