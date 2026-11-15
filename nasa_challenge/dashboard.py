# dashboard.py
"""
AI Space Guardian — Streamlit dashboard
Talks to backend.py (FastAPI) at localhost:8000
ALL DATA IS SYNTHETIC.
"""
import streamlit as st
import pandas as pd
import requests

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Space Guardian", page_icon="🛰️", layout="wide")

st.title("🛰️ AI Space Guardian")
st.caption("⚠️ All data is synthetic — not real NASA data, not medically validated.")

# ---------- Top: crew status cards ----------
st.subheader("Crew Status")

try:
    astronauts = requests.get(f"{API}/astronauts", timeout=3).json()
except Exception as e:
    st.error(f"Cannot reach backend at {API}. Is `uvicorn backend:app` running? ({e})")
    st.stop()

cols = st.columns(4)
for col, ast in zip(cols, astronauts):
    aid = ast["astronaut_id"]
    try:
        latest = requests.get(f"{API}/astronauts/{aid}/latest", timeout=3).json()
    except Exception:
        latest = {}

    with col:
        st.markdown(f"### {ast['astronaut_name']}")
        st.caption(f"`{aid}` · {ast['mission_role']}")
        if latest:
            hr   = latest.get("heart_rate_bpm")
            spo2 = latest.get("spo2_percent")
            sbp  = latest.get("systolic_bp_mmhg")
            dbp  = latest.get("diastolic_bp_mmhg")
            temp = latest.get("body_temperature_c")
            rad  = latest.get("radiation_dose_rate_usv_h")
            status = latest.get("sensor_status", "ok")

            icon = {"ok": "✅", "disconnected": "🔌", "malfunction": "⚠️",
                    "corrupted": "🟥", "missing": "❓"}.get(status, "❔")
            st.markdown(f"**Status:** {icon} `{status}`")
            st.metric("Heart rate", f"{hr} BPM" if hr is not None else "—")
            st.metric("SpO₂", f"{spo2} %" if spo2 is not None else "—")
            st.metric("Blood pressure",
                      f"{sbp}/{dbp} mmHg" if sbp and dbp else "—")
            st.metric("Body temp", f"{temp} °C" if temp is not None else "—")
            st.metric("Radiation", f"{rad} μSv/h" if rad is not None else "—")

# ---------- Middle: crew comparison chart ----------
st.subheader("Crew Comparison")

sensor_label = st.selectbox(
    "Sensor",
    options=[
        ("Heart rate",                "heart_rate_bpm"),
        ("Blood oxygen (SpO₂)",       "spo2_percent"),
        ("Radiation dose rate",       "radiation_dose_rate_usv_h"),
        ("Systolic blood pressure",   "systolic_bp_mmhg"),
        ("Body temperature",          "body_temperature_c"),
        ("Activity intensity",        "activity_intensity"),
    ],
    format_func=lambda x: x[0],
)
sensor_key = sensor_label[1]

try:
    cmp = requests.get(
        f"{API}/astronauts/compare",
        params={"sensor": sensor_key, "last_n": 500},
        timeout=5,
    ).json()
except Exception as e:
    st.error(f"Backend error: {e}")
    st.stop()

# Build a wide frame: one row per timestamp, one column per astronaut
frames = []
for aid, points in cmp.items():
    if not points:
        continue
    df = pd.DataFrame(points)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").rename(columns={"value": aid})
    frames.append(df)

if frames:
    merged = pd.concat(frames, axis=1).sort_index()
    st.line_chart(merged, height=320)
else:
    st.info("No data for this sensor.")

# ---------- Bottom: alerts ----------
st.subheader("Recent Simulated Alerts")
try:
    alerts = requests.get(f"{API}/alerts", timeout=3).json()
except Exception:
    alerts = []

if alerts:
    adf = pd.DataFrame(alerts)[[
        "timestamp", "astronaut_id", "sensor_name",
        "anomaly_type", "simulated_severity"
    ]]
    st.dataframe(adf, use_container_width=True, hide_index=True)
else:
    st.info("No alerts in dataset.")

# ---------- Sidebar: disclaimer ----------
with st.sidebar:
    st.markdown("### ⚠️ Disclaimer")
    st.write(
        "This dashboard displays **synthetic simulator output** "
        "for development and prototype testing. "
        "It is **not** real astronaut data and **not** medically validated."
    )
    st.markdown("---")
    st.markdown("### API")
    st.code(API)
    st.markdown("### Dataset")
    try:
        s = requests.get(f"{API}/summary", timeout=3).json()
        st.write(f"Astronaut records: **{s['total_astronaut_records']:,}**")
        st.write(f"Environment records: **{s['total_environment_records']:,}**")
        st.write(f"Alerts: **{s['total_alerts']}**")
    except Exception:
        st.write("Backend not reachable")