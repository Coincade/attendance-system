import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from Home import face_rec

st.set_page_config(page_title="Smart Pantry Report", layout="wide")
st.title("🍽️ Smart Pantry Time Summary")

# Load logs from Redis
def load_logs():
    logs = face_rec.r.lrange("pantry:logs", 0, -1)
    return [log.decode() for log in logs]

# Convert logs to DataFrame
def parse_logs(logs):
    data = []
    for log in logs:
        parts = log.split("@")
        if len(parts) == 4:
            emp_id, name, timestamp, action = parts
            data.append([emp_id, name, pd.to_datetime(timestamp), action])
    df = pd.DataFrame(data, columns=["Employee_id", "Name", "Timestamp", "Action"])
    df["Date"] = df["Timestamp"].dt.date
    return df

# Function to calculate sessions (reusable)
def generate_sessions(df_logs, selected_date, min_duration_minutes=None):
    df_logs = df_logs[df_logs["Date"] == selected_date]
    summary = []

    min_gap = timedelta(minutes=5)
    max_invisible = timedelta(hours=1)

    for emp_id, group in df_logs.groupby("Employee_id"):
        group = group.sort_values("Timestamp").reset_index(drop=True)
        name = group.iloc[0]["Name"]

        sessions = []
        session_start = group.loc[0, "Timestamp"]

        for i in range(1, len(group)):
            current_time = group.loc[i, "Timestamp"]
            prev_time = group.loc[i - 1, "Timestamp"]
            gap = current_time - prev_time

            if gap > max_invisible:
                sessions.append((session_start, prev_time))
                session_start = current_time
            elif gap > min_gap:
                sessions.append((session_start, current_time))
                session_start = None
            else:
                if session_start is None:
                    session_start = prev_time

        if session_start is not None:
            sessions.append((session_start, group.iloc[-1]["Timestamp"]))

        for start, end in sessions:
            if start and end:
                duration = (end - start).total_seconds() / 60
                if min_duration_minutes is None or duration > min_duration_minutes:
                    summary.append({
                        "Employee_id": emp_id,
                        "Name": name,
                        "Date": selected_date,
                        "Entry Time": start.strftime("%H:%M:%S"),
                        "Exit Time": end.strftime("%H:%M:%S"),
                        "Duration (min)": round(duration, 2)
                    })

    return pd.DataFrame(summary)

# UI
selected_date = st.date_input("📅 Select Date", datetime.now().date())

# Load + Process Logs
logs = load_logs()
df_logs = parse_logs(logs)

# Section 1: Only >5 Min
st.subheader("🧾 Valid Pantry Visits (>5 min)")
summary_df = generate_sessions(df_logs, selected_date, min_duration_minutes=5)

if summary_df.empty:
    st.warning("No pantry sessions longer than 5 minutes were found for this date.")
else:
    st.dataframe(summary_df)

    st.subheader("⏳ Total Pantry Time per Employee (Filtered)")
    total_df = summary_df.groupby(['Employee_id', 'Name'])['Duration (min)'].sum().reset_index()
    total_df.rename(columns={"Duration (min)": "Total Duration (min)"}, inplace=True)
    st.dataframe(total_df)

    st.download_button("📥 Download >5 Min Visits", summary_df.to_csv(index=False), file_name=f"pantry_visits_{selected_date}.csv")
    st.download_button("📥 Download Total Durations", total_df.to_csv(index=False), file_name=f"pantry_totals_{selected_date}.csv")

# Section 2: All Sessions
st.divider()
st.subheader("🔎 All Pantry Sessions (Internal Reference)")
all_sessions_df = generate_sessions(df_logs, selected_date, min_duration_minutes=None)

if all_sessions_df.empty:
    st.warning("No pantry sessions found for this date.")
else:
    st.dataframe(all_sessions_df)

    st.download_button("📥 Download All Sessions", all_sessions_df.to_csv(index=False), file_name=f"all_pantry_sessions_{selected_date}.csv")
