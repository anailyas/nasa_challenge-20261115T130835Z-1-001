# generate_astronaut_dataset.py
"""
AI Space Guardian - Virtual Sensor Data Generation System
=========================================================
Simulates 4 astronauts wearing virtual smart T-shirts, smartwatches,
radiation detectors, and movement sensors, plus spacecraft environment sensors.

ALL DATA IS SYNTHETIC. Not real NASA data. Not medically validated.
"""

import json
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SEED = 42
MISSION_ID = "ARTEMIS-MARS-2031"
N_DAYS = 1
SAMPLE_INTERVAL_SEC = 5          # one reading every 5 seconds
T0 = datetime(2031, 3, 1, 0, 0, 0)

# Derived
N_STEPS = int(N_DAYS * 24 * 60 * 60 / SAMPLE_INTERVAL_SEC)  # 17,280 per astronaut
OUT = Path("AI_SPACE_GUARDIAN_DATA")
(OUT / "visualizations").mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. ASTRONAUT PROFILES
# ---------------------------------------------------------------------------
ASTRONAUTS = [
    {
        "astronaut_id": "AST-001", "astronaut_name": "Cmdr. Elena Vasquez",
        "mission_role": "Mission Commander", "mission_id": MISSION_ID,
        "mission_day": 1, "age": 42,
        "device_ids": "DEV-001-TShirt;DEV-001-Watch;DEV-001-Radiation;DEV-001-Movement",
        "profile_status": "active",
        "registration_date": "2031-02-20",
    },
    {
        "astronaut_id": "AST-002", "astronaut_name": "Dr. Marcus Chen",
        "mission_role": "Flight Engineer / Medical Officer", "mission_id": MISSION_ID,
        "mission_day": 1, "age": 38,
        "device_ids": "DEV-002-TShirt;DEV-002-Watch;DEV-002-Radiation;DEV-002-Movement",
        "profile_status": "active",
        "registration_date": "2031-02-20",
    },
    {
        "astronaut_id": "AST-003", "astronaut_name": "Lt. Aisha Okafor",
        "mission_role": "Payload Specialist", "mission_id": MISSION_ID,
        "mission_day": 1, "age": 35,
        "device_ids": "DEV-003-TShirt;DEV-003-Watch;DEV-003-Radiation;DEV-003-Movement",
        "profile_status": "active",
        "registration_date": "2031-02-20",
    },
    {
        "astronaut_id": "AST-004", "astronaut_name": "Dr. Yuki Tanaka",
        "mission_role": "Science Officer", "mission_id": MISSION_ID,
        "mission_day": 1, "age": 40,
        "device_ids": "DEV-004-TShirt;DEV-004-Watch;DEV-004-Radiation;DEV-004-Movement",
        "profile_status": "active",
        "registration_date": "2031-02-20",
    },
]

# ---------------------------------------------------------------------------
# 2. DEVICES
# ---------------------------------------------------------------------------
DEVICE_TYPES = ["TShirt", "Watch", "Radiation", "Movement"]

def build_devices():
    rows = []
    for a in ASTRONAUTS:
        aid = a["astronaut_id"]
        num = aid.split("-")[1]
        for dt in DEVICE_TYPES:
            rows.append({
                "device_id": f"DEV-{num}-{dt}",
                "astronaut_id": aid,
                "device_type": dt,
                "device_name": f"{a['astronaut_name']} {dt} Wearable (Virtual)",
                "connection_status": "connected",
                "installation_date": "2031-02-25",
                "data_source": "virtual_simulator",
            })
    # spacecraft environmental sensors (not per-astronaut)
    for env in [
        ("ENV-O2", "Cabin Oxygen Sensor"),
        ("ENV-TEMP", "Cabin Temperature Sensor"),
        ("ENV-PRESS", "Cabin Pressure Sensor"),
        ("ENV-HUM", "Cabin Humidity Sensor"),
        ("ENV-AIR", "Air Quality Sensor"),
    ]:
        rows.append({
            "device_id": env[0],
            "astronaut_id": "",   # spacecraft-level
            "device_type": "Environment",
            "device_name": env[1],
            "connection_status": "connected",
            "installation_date": "2031-02-25",
            "data_source": "virtual_simulator",
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 3. PHYSIOLOGICAL SIGNAL GENERATORS
# ---------------------------------------------------------------------------

def circadian(hour_of_day):
    """Circadian weight, peaks mid-day, dips at night (returns 0..1)."""
    return 0.5 * (1 + np.cos((hour_of_day - 14) * 2 * np.pi / 24))

def build_activity_curve(n, interval_s, seed_offset):
    """Return per-step activity level category and intensity (0-100)."""
    r = np.random.default_rng(SEED + seed_offset)
    t_hours = (np.arange(n) * interval_s) / 3600.0
    hod = (t_hours % 24)

    sleep = ((hod >= 22.5) | (hod < 6.5))
    exercise = ((hod >= 7) & (hod < 8)) | ((hod >= 17) & (hod < 18.5))
    walking = ((hod >= 6.5) & (hod < 7)) | ((hod >= 8) & (hod < 9)) | \
              ((hod >= 12.5) & (hod < 13)) | ((hod >= 18.5) & (hod < 19))
    working = (~sleep) & (~exercise) & (~walking) & (hod >= 9) & (hod < 17)
    resting = ~(sleep | exercise | walking | working)

    intensity = np.zeros(n, dtype=float)
    labels = np.empty(n, dtype=object)

    intensity[sleep]    = r.normal(4, 1.5, sleep.sum()).clip(0, 12)
    intensity[resting]  = r.normal(12, 4, resting.sum()).clip(5, 25)
    intensity[working]  = r.normal(35, 8, working.sum()).clip(15, 55)
    intensity[walking]  = r.normal(55, 10, walking.sum()).clip(35, 75)
    intensity[exercise] = r.normal(82, 10, exercise.sum()).clip(55, 100)

    labels[sleep]    = "sleeping"
    labels[resting]  = "resting"
    labels[working]  = "working_inside_spacecraft"
    labels[walking]  = "walking"
    labels[exercise] = "exercising"
    return intensity, labels, t_hours

def smooth(arr, window):
    """Simple moving average to give continuity."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")

def build_astronaut_signals(profile, idx):
    """Generate all per-astronaut continuous signals."""
    r = np.random.default_rng(SEED + 100 + idx)
    n = N_STEPS
    interval_s = SAMPLE_INTERVAL_SEC

    activity, act_label, t_hours = build_activity_curve(n, interval_s, seed_offset=10 * (idx + 1))

    base_hr    = [62, 70, 66, 74][idx]
    base_sbp   = [118, 124, 116, 121][idx]
    base_dbp   = [76, 80, 74, 78][idx]
    base_temp  = [36.6, 36.7, 36.5, 36.8][idx]
    base_br    = [14, 15, 13, 16][idx]
    base_spo2  = [98.0, 97.5, 98.2, 97.8][idx]

    hr = base_hr + 0.45 * activity + 3 * circadian(t_hours) \
         + r.normal(0, 2.0, n)
    hr = smooth(hr, 6).clip(45, 165)

    br = base_br + 0.10 * activity + r.normal(0, 0.8, n)
    br = smooth(br, 8).clip(9, 34)

    spo2 = base_spo2 - 0.008 * activity + r.normal(0, 0.25, n)
    spo2 = smooth(spo2, 10).clip(90, 100)

    sbp = base_sbp + 0.20 * activity + r.normal(0, 2.5, n)
    dbp = base_dbp + 0.10 * activity + r.normal(0, 1.8, n)
    sbp = smooth(sbp, 10).clip(95, 160)
    dbp = smooth(dbp, 10).clip(55, 105)

    temp = base_temp + 0.003 * activity + 0.15 * circadian(t_hours) \
           + r.normal(0, 0.08, n)
    temp = smooth(temp, 20).clip(35.5, 39.0)

    sleep_status = np.where(act_label == "sleeping", "asleep", "awake")

    sleep_hours = np.zeros(n)
    for i in range(n):
        lo = max(0, i - int(24 * 3600 / interval_s) + 1)
        window = sleep_status[lo:i+1]
        sleep_hours[i] = np.sum(window == "asleep") * interval_s / 3600.0

    hod = (t_hours % 24)
    wake_mask = (hod >= 6.5) & (hod < 7.0)
    last_q = [72, 65, 80, 58][idx] + r.normal(0, 3)
    nightly_quality = np.full(n, np.nan)
    for i in range(n):
        if wake_mask[i]:
            last_q = float(np.clip([72, 65, 80, 58][idx] + r.normal(0, 4), 30, 95))
        nightly_quality[i] = last_q
    sleep_quality = nightly_quality

    stress = 20 + 0.35 * activity + 6 * circadian(t_hours) + r.normal(0, 4, n)
    n_spikes = int(n / 4000)
    spike_idx = r.integers(0, n, n_spikes)
    stress[spike_idx] += r.uniform(15, 30, n_spikes)
    stress = smooth(stress, 6).clip(0, 100)

    movement = activity * 0.9 + r.normal(0, 3, n)
    movement = smooth(movement, 3).clip(0, 120)

    trend_slope = [-0.002, 0.001, -0.001, -0.003][idx]
    muscle = 80 + trend_slope * np.arange(n) + r.normal(0, 1.2, n)
    muscle = smooth(muscle, 25).clip(40, 100)

    base_dose = [0.30, 0.32, 0.28, 0.35][idx]
    dose_rate = np.abs(r.normal(base_dose, 0.03, n))
    if idx == 2:
        ev_start = int(n * 0.45)
        ev_len = int(n * 0.02)
        dose_rate[ev_start:ev_start+ev_len] += r.uniform(2.5, 5.0)
    dose_rate = smooth(dose_rate, 12).clip(0.05, 20.0)

    dt_h = interval_s / 3600.0
    cumulative = np.cumsum(dose_rate * dt_h)

    scenario = act_label.copy()

    return {
        "t_hours": t_hours,
        "activity": activity,
        "activity_label": act_label,
        "heart_rate": hr,
        "breathing_rate": br,
        "spo2": spo2,
        "systolic": sbp,
        "diastolic": dbp,
        "temperature": temp,
        "sleep_status": sleep_status,
        "sleep_duration": sleep_hours,
        "sleep_quality": sleep_quality,
        "stress": stress,
        "movement": movement,
        "muscle": muscle,
        "dose_rate": dose_rate,
        "cumulative_dose": cumulative,
        "scenario": scenario,
    }

# ---------------------------------------------------------------------------
# 4. ANOMALY / MISSING-DATA / SENSOR-FAULT INJECTION
# ---------------------------------------------------------------------------
def inject_anomalies(sig, idx):
    """Inject synthetic labelled anomalies and return alert rows."""
    r = np.random.default_rng(SEED + 2000 + idx)
    n = N_STEPS
    alerts = []
    ast_id = ASTRONAUTS[idx]["astronaut_id"]
    dev_prefix = ast_id.split("-")[1]

    sensor_status = np.full(n, "ok", dtype=object)

    def add_alert(i, device, sensor, observed, expected, atype, sev):
        alerts.append({
            "alert_id": f"ALT-{uuid.uuid4().hex[:10]}",
            "timestamp": (T0 + timedelta(seconds=int(i * SAMPLE_INTERVAL_SEC))).isoformat(),
            "astronaut_id": ast_id,
            "device_id": f"DEV-{dev_prefix}-{device}",
            "sensor_name": sensor,
            "observed_value": round(float(observed), 3) if observed == observed else "",
            "expected_reference": expected,
            "anomaly_type": atype,
            "scenario_label": "simulated_anomaly",
            "simulated_severity": sev,
            "is_simulated": True,
        })

    # --- 4a. T-shirt sensor disconnection (2 short windows) ---
    for _ in range(2):
        start = r.integers(500, n - 500)
        length = r.integers(20, 60)
        sl = slice(int(start), int(start + length))
        for key in ["heart_rate", "breathing_rate", "spo2", "systolic",
                    "diastolic", "temperature"]:
            sig[key][sl] = np.nan
        sensor_status[sl] = "disconnected"
        sig["scenario"][sl] = "sensor_disconnection"
        add_alert(start, "TShirt", "heart_rate", np.nan, "60-100 bpm",
                  "sensor_disconnection", "medium")

    # --- 4b. Radiation detector equipment malfunction ---
    start = r.integers(500, n - 400)
    length = r.integers(30, 80)
    sl = slice(int(start), int(start + length))
    sig["dose_rate"][sl] = np.nan
    sensor_status[sl] = "malfunction"
    sig["scenario"][sl] = "equipment_malfunction"
    add_alert(start, "Radiation", "radiation_dose_rate", np.nan, "0.05-1.0 uSv/h",
              "equipment_malfunction", "high")

    # --- 4c. Corrupted readings ---
    start = int(r.integers(500, n - 200))
    sig["heart_rate"][start:start+5] = 999.0
    sig["scenario"][start:start+5] = "corrupted_reading"
    sensor_status[start:start+5] = "corrupted"
    add_alert(start, "TShirt", "heart_rate", 999.0, "60-100 bpm",
              "corrupted_reading", "medium")

    # --- 4d. Unusual SpO2 drop ---
    start = int(r.integers(500, n - 300))
    sig["spo2"][start:start+40] = r.normal(88, 1.2, 40)
    sig["scenario"][start:start+40] = "unusual_sensor_reading"
    add_alert(start, "TShirt", "spo2_percent",
              float(np.nanmean(sig["spo2"][start:start+40])),
              "95-100 %", "unusual_sensor_reading", "medium")

    # --- 4e. Movement sensor packet loss ---
    start = int(r.integers(500, n - 200))
    sig["movement"][start:start+30] = np.nan
    sig["scenario"][start:start+30] = "missing_measurement"
    sensor_status[start:start+30] = "missing"
    add_alert(start, "Movement", "movement_indicator", np.nan, "0-120 sensor units",
              "missing_measurement", "low")

    # --- 4f. Watch battery / connection loss ---
    start = int(r.integers(500, n - 300))
    length = int(r.integers(40, 100))
    sl = slice(start, start + length)
    sig["sleep_quality"][sl] = np.nan
    sig["sleep_duration"][sl] = np.nan
    sensor_status[sl] = "disconnected"
    add_alert(start, "Watch", "sleep_quality_score", np.nan, "0-100",
              "sensor_disconnection", "low")

    return sig, sensor_status, alerts

# ---------------------------------------------------------------------------
# 5. BUILD ASTRONAUT SENSOR DATAFRAME
# ---------------------------------------------------------------------------
def build_astronaut_dataframe(all_alerts):
    rows = []
    for idx, a in enumerate(ASTRONAUTS):
        aid = a["astronaut_id"]
        num = aid.split("-")[1]
        sig = build_astronaut_signals(a, idx)
        sig, sensor_status, alerts = inject_anomalies(sig, idx)
        all_alerts.extend(alerts)

        n = N_STEPS
        timestamps = [T0 + timedelta(seconds=i * SAMPLE_INTERVAL_SEC) for i in range(n)]

        df = pd.DataFrame({
            "record_id": [str(uuid.uuid4()) for _ in range(n)],
            "timestamp": timestamps,
            "astronaut_id": aid,
            "device_id": f"DEV-{num}-TShirt;DEV-{num}-Watch;DEV-{num}-Radiation;DEV-{num}-Movement",
            "sensor_id": f"SENSOR-SET-{num}",
            "spo2_percent": sig["spo2"],
            "heart_rate_bpm": sig["heart_rate"],
            "radiation_dose_rate_usv_h": sig["dose_rate"],
            "cumulative_radiation_dose_usv": sig["cumulative_dose"],
            "muscle_performance_score": sig["muscle"],
            "systolic_bp_mmhg": sig["systolic"],
            "diastolic_bp_mmhg": sig["diastolic"],
            "body_temperature_c": sig["temperature"],
            "breathing_rate_bpm": sig["breathing_rate"],
            "sleep_duration_hours": sig["sleep_duration"],
            "sleep_quality_score": sig["sleep_quality"],
            "sleep_status": sig["sleep_status"],
            "activity_level": sig["activity_label"],
            "activity_intensity": sig["activity"],
            "movement_indicator": sig["movement"],
            "stress_indicator": sig["stress"],
            "scenario_label": sig["scenario"],
            "sensor_status": sensor_status,
            "data_source": "virtual_simulator",
            "is_simulated": True,
        })
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

# ---------------------------------------------------------------------------
# 6. SPACECRAFT ENVIRONMENT DATA
# ---------------------------------------------------------------------------
def build_environment_dataframe():
    r = np.random.default_rng(SEED + 9999)
    n = N_STEPS
    timestamps = [T0 + timedelta(seconds=i * SAMPLE_INTERVAL_SEC) for i in range(n)]

    o2   = 20.9 + r.normal(0, 0.05, n)
    temp = 22.0 + 0.4 * np.sin(np.arange(n) / 1000) + r.normal(0, 0.1, n)
    press= 101.3 + r.normal(0, 0.05, n)
    hum  = 45.0 + 2.0 * np.sin(np.arange(n) / 800) + r.normal(0, 0.4, n)
    air  = 15.0 + r.normal(0, 2.0, n)

    start = int(n * 0.6)
    temp[start:start+50] = r.normal(27, 0.5, 50)
    air[start:start+50]  = r.normal(55, 5, 50)

    scenario = np.full(n, "normal", dtype=object)
    scenario[start:start+50] = "equipment_anomaly"

    return pd.DataFrame({
        "record_id": [str(uuid.uuid4()) for _ in range(n)],
        "timestamp": timestamps,
        "environment_id": "ENV-CABIN-01",
        "cabin_oxygen_percent": o2.clip(19.0, 23.0),
        "cabin_temperature_c": temp.clip(18, 30),
        "cabin_pressure_kpa": press.clip(95, 105),
        "humidity_percent": hum.clip(20, 70),
        "air_quality_indicator": air.clip(0, 100),
        "scenario_label": scenario,
        "sensor_status": "ok",
        "is_simulated": True,
    })

# ---------------------------------------------------------------------------
# 7. SAVE FILES
# ---------------------------------------------------------------------------
def _sanitize(obj):
    """Recursively convert numpy / pandas / datetime objects to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize(v) for v in obj]
    if obj is None:
        return None
    try:
        if obj is pd.NaT:
            return None
    except Exception:
        pass
    if isinstance(obj, float) and (obj != obj):          # NaN
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    return obj

def save_all(astronauts_df, devices_df, astro_df, env_df, alerts_df):
    astronauts_df.to_csv(OUT / "astronauts.csv", index=False, encoding="utf-8")
    devices_df.to_csv(OUT / "devices.csv", index=False, encoding="utf-8")
    astro_df.to_csv(OUT / "astronaut_sensor_data.csv", index=False,
                    na_rep="", encoding="utf-8")
    env_df.to_csv(OUT / "spacecraft_environment_data.csv", index=False,
                  na_rep="", encoding="utf-8")
    alerts_df.to_csv(OUT / "astronaut_alert_scenarios.csv", index=False,
                     encoding="utf-8")

    # ---- JSON export (timestamps -> ISO strings, NaN -> null) ----
    json_df = astro_df.copy()
    json_df["timestamp"] = json_df["timestamp"].apply(
        lambda t: t.isoformat() if pd.notna(t) else None
    )
    records = json_df.to_dict(orient="records")
    records = _sanitize(records)
    with open(OUT / "astronaut_sensor_data.json", "w", encoding="utf-8") as f:
        json.dump(records, f)

# ---------------------------------------------------------------------------
# 8. VISUALIZATIONS
# ---------------------------------------------------------------------------
def make_visualizations(astro_df, env_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vdir = OUT / "visualizations"
    colors = {"AST-001": "#1f77b4", "AST-002": "#ff7f0e",
              "AST-003": "#2ca02c", "AST-004": "#d62728"}
    ids = sorted(astro_df["astronaut_id"].unique())

    def ds(df, cols):
        sub = df[["timestamp", "astronaut_id"] + cols].dropna().copy()
        return sub.iloc[::60]

    def line_plot(col, title, ylabel, fname):
        d = ds(astro_df, [col])
        plt.figure(figsize=(11, 4.5))
        for aid in ids:
            s = d[d.astronaut_id == aid]
            plt.plot(s.timestamp, s[col], color=colors[aid], label=aid, lw=0.8)
        plt.title(title); plt.ylabel(ylabel); plt.xlabel("Time")
        plt.legend(loc="upper right", fontsize=8); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(vdir / fname, dpi=110); plt.close()

    line_plot("heart_rate_bpm", "Heart Rate — All Astronauts", "BPM", "01_heart_rate.png")
    line_plot("spo2_percent", "Blood Oxygen Saturation (SpO2)", "%", "02_spo2.png")
    line_plot("radiation_dose_rate_usv_h", "Radiation Dose Rate", "uSv/h", "03_radiation_dose_rate.png")
    line_plot("cumulative_radiation_dose_usv", "Cumulative Radiation Dose", "uSv", "04_cumulative_dose.png")
    line_plot("systolic_bp_mmhg", "Systolic Blood Pressure", "mmHg", "05_systolic_bp.png")
    line_plot("diastolic_bp_mmhg", "Diastolic Blood Pressure", "mmHg", "06_diastolic_bp.png")
    line_plot("body_temperature_c", "Body Temperature", "C", "07_body_temperature.png")
    line_plot("breathing_rate_bpm", "Breathing Rate", "breaths/min", "08_breathing_rate.png")
    line_plot("activity_intensity", "Physical Activity Intensity", "0-100", "09_activity.png")
    line_plot("muscle_performance_score", "Muscle Performance Score", "0-100", "10_muscle_perf.png")

    sleep_max = astro_df.groupby("astronaut_id")["sleep_duration_hours"].max()
    plt.figure(figsize=(7, 4))
    sleep_max.plot(kind="bar", color=[colors[i] for i in sleep_max.index])
    plt.title("Max Sleep Duration (24h window)"); plt.ylabel("hours")
    plt.tight_layout(); plt.savefig(vdir / "11_sleep_duration.png", dpi=110); plt.close()

    d = env_df[["timestamp", "cabin_oxygen_percent", "cabin_temperature_c",
                "cabin_pressure_kpa", "humidity_percent"]].iloc[::60]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, col, ylab in zip(axes,
        ["cabin_oxygen_percent", "cabin_temperature_c", "cabin_pressure_kpa", "humidity_percent"],
        ["O2 %", "Temp C", "Pressure kPa", "Humidity %"]):
        ax.plot(d.timestamp, d[col], lw=0.8)
        ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Spacecraft Cabin Environmental Conditions")
    plt.tight_layout(); plt.savefig(vdir / "12_environment.png", dpi=110); plt.close()

# ---------------------------------------------------------------------------
# 9. DATA DICTIONARY
# ---------------------------------------------------------------------------
def write_data_dictionary():
    text = """# AI Space Guardian — Data Dictionary

> **ALL DATA IS SYNTHETIC / SIMULATED.** It is not real NASA astronaut
> data and is **not medically validated**. Generated by `generate_astronaut_dataset.py`
> for development and AI-model prototyping only.

## Identifiers
| Field | Meaning |
|---|---|
| `astronaut_id` | Primary key for an astronaut (`AST-001` ... `AST-004`). |
| `device_id` | Primary key for a virtual device (`DEV-NNN-Type`). |
| `record_id` | UUID primary key for one consolidated sensor row. |
| `sensor_id` | Identifier of the sensor set carried by one astronaut. |
| `environment_id` | Spacecraft-cabin environment recording group. |
| `alert_id` | UUID for a labelled simulated anomaly event. |

## Files
| File | Purpose |
|---|---|
| `astronauts.csv` | Astronaut profiles (1 row per astronaut). |
| `devices.csv` | All virtual wearable + environmental sensors. |
| `astronaut_sensor_data.csv` / `.json` | Per-astronaut continuous sensor rows. |
| `spacecraft_environment_data.csv` | Cabin environment readings. |
| `astronaut_alert_scenarios.csv` | Labelled synthetic anomalies and faults. |

## Units & Ranges (`astronaut_sensor_data`)
| Field | Unit | Typical range | Notes |
|---|---|---|---|
| `spo2_percent` | % | 88-100 | Simulated SpO2 |
| `heart_rate_bpm` | BPM | 45-165 | Activity-linked |
| `radiation_dose_rate_usv_h` | uSv/h | 0.05-20 | Solar event spike for AST-003 |
| `cumulative_radiation_dose_usv` | uSv | integral of dose_rate dt | Mathematically integrated |
| `muscle_performance_score` | 0-100 | 40-100 | Simulated physical-performance indicator (**not** muscle mass) |
| `systolic_bp_mmhg` / `diastolic_bp_mmhg` | mmHg | 95-160 / 55-105 | Activity-linked |
| `body_temperature_c` | C | 35.5-39.0 | Circadian + activity |
| `breathing_rate_bpm` | breaths/min | 9-34 | Activity-linked |
| `sleep_duration_hours` | hours | 0-10 | Trailing-24h sleep hours |
| `sleep_quality_score` | 0-100 | 30-95 | Last-night score |
| `sleep_status` | text | `asleep`/`awake` | Coherent with activity |
| `activity_level` | text | see below | Category label |
| `activity_intensity` | 0-100 | 0-100 | Continuous intensity |
| `movement_indicator` | sensor units | 0-120 | Arbitrary IMU-style units |
| `stress_indicator` | 0-100 | 0-100 | Simulated, not diagnostic |

### `activity_level` values
`sleeping`, `resting`, `walking`, `exercising`, `working_inside_spacecraft`

## Spacecraft environment (`spacecraft_environment_data.csv`)
| Field | Unit | Range |
|---|---|---|
| `cabin_oxygen_percent` | % | ~19-23 (cabin oxygen concentration, distinct from astronaut SpO2) |
| `cabin_temperature_c` | C | 18-30 |
| `cabin_pressure_kpa` | kPa | 95-105 |
| `humidity_percent` | % | 20-70 |
| `air_quality_indicator` | 0-100 | 0 = best |

## Flags & Labels
| Field | Meaning |
|---|---|
| `scenario_label` | normal/activity label or injected anomaly label |
| `sensor_status` | `ok`, `disconnected`, `malfunction`, `corrupted`, `missing` |
| `data_source` | `virtual_simulator` |
| `is_simulated` | Always `True` |
| `simulated_severity` | `low`/`medium`/`high` (alert file only) |

## Missing data
Missing readings are stored as empty CSV cells / JSON `null`.
They are **never** replaced with fabricated values.

## Isolation rules
* Each device belongs to exactly one astronaut.
* Each record links to exactly one astronaut.
* Environment rows are spacecraft-level (no astronaut_id).
* Cumulative radiation per astronaut is independent.

## Simulation assumptions
* One reading per 5 s, 24 h -> 17,280 rows per astronaut, 69,120 total.
* Circadian cycle drives sleep (22:30-06:30) and body temperature.
* Exercise windows 07:00-08:00 and 17:00-18:30.
* Radiation events injected only for AST-003.
* Anomalies injected per astronaut with unique random seeds.
"""
    (OUT / "data_dictionary.md").write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# 10. SUMMARY REPORT
# ---------------------------------------------------------------------------
def write_summary(astro_df, env_df, alerts_df, devices_df):
    lines = []
    lines.append("# AI Space Guardian — Dataset Summary\n")
    lines.append("> All data is **synthetic**. Not real NASA data. Not medically validated.\n")
    lines.append(f"**Generated:** {datetime.utcnow().isoformat()} UTC  ")
    lines.append(f"**Simulation:** {N_DAYS} day(s), interval {SAMPLE_INTERVAL_SEC}s  ")
    lines.append(f"**Random seed:** {SEED}\n")

    lines.append("## Record counts")
    lines.append(f"- Astronaut sensor records: **{len(astro_df):,}**")
    lines.append(f"- Spacecraft environment records: **{len(env_df):,}**")
    lines.append(f"- Labelled alerts / anomalies: **{len(alerts_df):,}**")
    lines.append(f"- Device rows: **{len(devices_df)}**\n")

    lines.append("## Records per astronaut")
    for aid, cnt in astro_df.groupby("astronaut_id").size().items():
        lines.append(f"- `{aid}`: {cnt:,}")
    lines.append("")

    lines.append("## Missing readings per sensor")
    for col in ["heart_rate_bpm", "spo2_percent", "radiation_dose_rate_usv_h",
                "sleep_quality_score", "movement_indicator"]:
        n_missing = int(astro_df[col].isna().sum())
        lines.append(f"- `{col}`: {n_missing:,} missing")
    lines.append("")

    lines.append("## Sensor-status distribution")
    for status, cnt in astro_df["sensor_status"].value_counts().items():
        lines.append(f"- `{status}`: {cnt:,}")
    lines.append("")

    lines.append("## Scenario-label distribution")
    for lbl, cnt in astro_df["scenario_label"].value_counts().items():
        lines.append(f"- `{lbl}`: {cnt:,}")
    lines.append("")

    lines.append("## Alerts per astronaut")
    if len(alerts_df):
        for aid in sorted(alerts_df.astronaut_id.unique()):
            lines.append(f"- `{aid}`: {len(alerts_df[alerts_df.astronaut_id==aid])} labelled events")
    lines.append("")

    lines.append("## Time coverage")
    lines.append(f"- Start: {astro_df.timestamp.min()}")
    lines.append(f"- End:   {astro_df.timestamp.max()}")
    lines.append(f"- Duration: {astro_df.timestamp.max() - astro_df.timestamp.min()}")
    lines.append("")

    lines.append("## Files")
    for f in sorted(OUT.iterdir()):
        if f.is_file():
            lines.append(f"- `{f.name}` ({f.stat().st_size:,} bytes)")
        elif f.is_dir():
            lines.append(f"- `{f.name}/` ({len(list(f.iterdir()))} files)")

    (OUT / "dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------------------
# 11. MAIN
# ---------------------------------------------------------------------------
def main():
    print("AI Space Guardian — virtual sensor data generation")
    print(f"  astronauts={len(ASTRONAUTS)}  days={N_DAYS}  interval={SAMPLE_INTERVAL_SEC}s")
    print(f"  timestamps per astronaut = {N_STEPS:,}")
    print(f"  output -> {OUT.resolve()}")

    astronauts_df = pd.DataFrame(ASTRONAUTS)
    devices_df = build_devices()

    print("  generating astronaut signals...")
    all_alerts = []
    astro_df = build_astronaut_dataframe(all_alerts)
    alerts_df = pd.DataFrame(all_alerts)

    print("  generating environment data...")
    env_df = build_environment_dataframe()

    print("  saving files...")
    save_all(astronauts_df, devices_df, astro_df, env_df, alerts_df)

    print("  writing data dictionary...")
    write_data_dictionary()

    print("  generating visualizations...")
    make_visualizations(astro_df, env_df)

    print("  writing summary report...")
    write_summary(astro_df, env_df, alerts_df, devices_df)

    print("DONE.")
    print(f"  Astronaut records: {len(astro_df):,}")
    print(f"  Environment records: {len(env_df):,}")
    print(f"  Alerts: {len(alerts_df)}")

if __name__ == "__main__":
    main()