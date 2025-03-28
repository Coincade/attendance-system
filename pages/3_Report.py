import streamlit as st
from Home import face_rec

st.set_page_config(page_title="Report",layout="wide")
st.subheader("Reporting")

# Retrieve the logs from the database
#extract data from redis list
list_name = 'attendance:logs'

def load_logs(name, end):
    logs_list = face_rec.r.lrange(name,start=0,end=end) # extract all data from redis DB, set end value to 100 if you want only first 100 records
    return logs_list

# tabs to show the info
tab1, tab2 = st.tabs(['Registered Employees', 'Attendance Logs'])

with tab1:
    if st.button('Refresh Employee Data'):
        # Retreive the data from Redis DB
        with st.spinner("Retrieving Data from Redis DB.."):
            redis_face_db = face_rec.retrieve_data('academy:register')
            st.dataframe(redis_face_db[['Name', 'Role']])

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(list_name,-1))







