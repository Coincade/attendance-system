import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from Home import face_rec

st.set_page_config(page_title="Report", layout="wide")
st.subheader("Reporting")

# Redis list key for attendance logs
list_name = 'attendance:logs'

# Load raw logs from Redis
def load_logs(name, end=-1):
    logs_list = face_rec.r.lrange(name, start=0, end=end)
    decoded_logs = [log.decode() for log in logs_list]
    return decoded_logs

# Convert raw logs to structured DataFrame
def logs_to_dataframe(logs):
    data = []
    for log in logs:
        parts = log.split("@")
        if len(parts) == 3:  # Expected format: employee_id@name@role@timestamp
            name, employee_id, timestamp = parts
            data.append([name, employee_id, timestamp])
    df = pd.DataFrame(data, columns=['Name', 'Employee_id', 'Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values(by=['Employee_id', 'Timestamp'], inplace=True)
    print (df)
    return df

# Calculate total working hours from logs
def calculate_total_working_hours(df_logs):
    df_logs['Date'] = df_logs['Timestamp'].dt.date
    summary = []

    grouped = df_logs.groupby(['Employee_id', 'Date'])

    for (employee_id, date), group in grouped:
        group = group.sort_values(by='Timestamp').reset_index(drop=True)
        name = group.loc[0, 'Name']
        in_time = group.iloc[0]['Timestamp']
        out_time = group.iloc[-1]['Timestamp']
        duration = out_time - in_time

        summary.append({
            'Employee_id': employee_id,
            'Name': name,
            'Date': date,
            'First Log (IN)': in_time,
            'Last Log (OUT)': out_time,
            'Total Hours': round(duration.total_seconds() / 3600, 2)
        })

    return pd.DataFrame(summary)



    work_summary.append({
            'Employee_id': employee_id,
            'Name': group.loc[0, 'Name'],
            'Date': date,
            'Total Hours': round(total_time.total_seconds() / 3600, 2)
        })

    return pd.DataFrame(work_summary)


# Streamlit layout
tab1, tab2, tab3 = st.tabs(['Registered Employees', 'Attendance Logs', 'Working Hours Summary'])

# Tab 1: Registered employees
with tab1:
    with st.spinner("Retrieving Data from Redis DB..."):
        redis_face_db = face_rec.retrieve_data('academy:register')
        st.dataframe(redis_face_db[['Employee_id', 'Name']])


# Tab 2: Attendance logs
with tab2:
    selected_date = st.date_input("Select Date for Logs", datetime.now().date())

    logs = load_logs(list_name)
    df_logs = logs_to_dataframe(logs)

    # Get unique IDs for dropdown
    all_emp_ids = df_logs['Employee_id'].unique().tolist()
    search_emp_code = st.selectbox("Search by Employee ID (optional)", options=["All"] + sorted(all_emp_ids))

    # Filter logs
    df_logs = df_logs[df_logs['Timestamp'].dt.date == selected_date]

    if search_emp_code != "All":
        df_logs = df_logs[df_logs['Employee_id'] == search_emp_code]

    st.dataframe(df_logs)




# Tab 3: Working hours summary
with tab3:
    selected_date_summary = st.date_input("Select Date for Summary", datetime.now().date(), key="summary_date")

    # Load logs and convert
    logs = load_logs(list_name)
    df_logs = logs_to_dataframe(logs)

    # Filter logs for selected date
    df_logs = df_logs[df_logs['Timestamp'].dt.date == selected_date_summary]

    # Create dropdown of unique employee IDs for that day
    all_emp_ids = sorted(df_logs['Employee_id'].unique().tolist())
    selected_emp_code = st.selectbox("Select Employee ID (or leave empty to show all)", options=["All"] + all_emp_ids, key="emp_select_summary")

    # Filter logs based on employee selection
    if selected_emp_code != "All":
        df_logs = df_logs[df_logs['Employee_id'] == selected_emp_code]

    # Now calculate total working hours
    df_summary = calculate_total_working_hours(df_logs)
    st.dataframe(df_summary)

    # Optional download
    st.download_button("Download Summary as CSV", df_summary.to_csv(index=False), file_name=f"summary_{selected_date_summary}.csv")


      
