# backend.py
"""
AI Space Guardian — FastAPI backend
Uses predictor.py for the collapse-risk model.
ALL DATA IS SYNTHETIC. Predictions are NOT a medical device.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import timedelta
import math
import pandas as pd

from predictor import predict as ai_predict, comm_delay_seconds

# ---------------------------------------------------------------------------
# Load data ONCE at startup
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "AI_SPACE_GUARDIAN_DATA"

if not DATA_DIR.exists():
    raise SystemExit(
        f"ERROR: {DATA_DIR} not found. Run generate_astronaut_dataset.py first."
    )

print("Loading dataset into memory...")
ASTRO    = pd.read_csv(DATA_DIR / "astronaut_sensor_data.csv", parse_dates=["timestamp"])
ENV      = pd.read_csv(DATA_DIR / "spacecraft_environment_data.csv", parse_dates=["timestamp"])
ALERTS   = pd.read_csv(DATA_DIR / "astronaut_alert_scenarios.csv")
PROFILES = pd.read_csv(DATA_DIR / "astronauts.csv")
DEVICES  = pd.read_csv(DATA_DIR / "devices.csv")
print(f"  loaded {len(ASTRO):,} astronaut rows, {len(ENV):,} env rows")

# ---------------------------------------------------------------------------
# Per-astronaut cursor — advances on every /latest call
#
# Each astronaut starts at a DIFFERENT phase of the 24h day so the
# collapse-risk board shows different scores immediately.
# Row = seconds_of_day / 5.  N_STEPS = 17,280 for 24h at 5s.
#   AST-001 -> 00:00  (sleeping)
#   AST-002 -> 07:12  (morning exercise window)
#   AST-003 -> 10:48  (working, solar event approaching)
#   AST-004 -> 17:30  (evening exercise window)
# ---------------------------------------------------------------------------
_START_OFFSETS = {
    "AST-001": 0,
    "AST-002": 5184,     # 07:12
    "AST-003": 7776,     # 10:48
    "AST-004": 12600,    # 17:30
}

_latest_cursor = {}


def _next_index(astronaut_id: str) -> int:
    if astronaut_id not in _latest_cursor:
        _latest_cursor[astronaut_id] = _START_OFFSETS.get(astronaut_id, 0)
    df = ASTRO[ASTRO.astronaut_id == astronaut_id]
    if len(df) == 0:
        return 0
    idx = _latest_cursor[astronaut_id]
    _latest_cursor[astronaut_id] = (idx + 1) % len(df)
    return idx


# ---------------------------------------------------------------------------
# JSON-safe helpers
# ---------------------------------------------------------------------------
def _safe_scalar(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def to_records(df: pd.DataFrame):
    return [
        {k: _safe_scalar(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _f(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return v


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Space Guardian API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "AI Space Guardian",
        "status": "ok",
        "disclaimer": "All data is synthetic. Not real NASA data.",
        "endpoints": [
            "/astronauts",
            "/astronauts/{id}",
            "/astronauts/{id}/sensors",
            "/astronauts/{id}/latest",
            "/astronauts/{id}/predict",
            "/astronauts/predict/all",
            "/astronauts/compare",
            "/alerts",
            "/environment",
            "/summary",
        ],
    }


@app.get("/astronauts")
def list_astronauts():
    return to_records(PROFILES)


@app.get("/astronauts/compare")
def compare(sensor: str = "heart_rate_bpm", last_n: int = 300):
    valid = ["heart_rate_bpm", "spo2_percent", "radiation_dose_rate_usv_h",
             "systolic_bp_mmhg", "body_temperature_c", "activity_intensity"]
    if sensor not in valid:
        raise HTTPException(400, f"sensor must be one of {valid}")
    out = {}
    for aid in sorted(ASTRO.astronaut_id.unique()):
        df = ASTRO[ASTRO.astronaut_id == aid][["timestamp", sensor]].tail(last_n)
        df = df.dropna().rename(columns={sensor: "value"})
        out[aid] = to_records(df)
    return out


# NOTE: literal route MUST come before /astronauts/{astronaut_id}
@app.get("/astronauts/predict/all")
def predict_all(phase: str = "mars_mid", window: int = 12):
    """
    Return collapse-risk prediction for ALL astronauts in one call.
    Each astronaut is scored independently using their own recent history.
    """
    results = []
    for aid in sorted(ASTRO.astronaut_id.unique()):
        try:
            results.append(_compute_prediction(aid, phase, window))
        except HTTPException:
            continue
    return results


@app.get("/astronauts/{astronaut_id}")
def get_astronaut(astronaut_id: str):
    row = PROFILES[PROFILES.astronaut_id == astronaut_id]
    if row.empty:
        raise HTTPException(404, f"Astronaut {astronaut_id} not found")
    devices = to_records(DEVICES[DEVICES.astronaut_id == astronaut_id])
    profile = to_records(row)[0]
    profile["devices"] = devices
    return profile


@app.get("/astronauts/{astronaut_id}/sensors")
def get_sensors(astronaut_id: str, sensor: str = "heart_rate_bpm",
                last_n: int = 300, downsample: int = 1):
    valid = ["heart_rate_bpm", "spo2_percent", "radiation_dose_rate_usv_h",
             "cumulative_radiation_dose_usv", "systolic_bp_mmhg",
             "diastolic_bp_mmhg", "body_temperature_c", "breathing_rate_bpm",
             "activity_intensity", "muscle_performance_score",
             "stress_indicator", "movement_indicator"]
    if sensor not in valid:
        raise HTTPException(400, f"sensor must be one of {valid}")
    df = ASTRO[ASTRO.astronaut_id == astronaut_id][["timestamp", sensor]]
    if df.empty:
        raise HTTPException(404, f"No data for {astronaut_id}")
    df = df.tail(last_n).iloc[::downsample]
    df = df.dropna().rename(columns={sensor: "value"})
    return to_records(df)


@app.get("/astronauts/{astronaut_id}/latest")
def get_latest(astronaut_id: str):
    df = ASTRO[ASTRO.astronaut_id == astronaut_id].reset_index(drop=True)
    if df.empty:
        raise HTTPException(404, f"No data for {astronaut_id}")

    idx = _next_index(astronaut_id)
    last = df.iloc[idx]

    return {
        "astronaut_id": astronaut_id,
        "timestamp": last["timestamp"].isoformat(),
        "heart_rate_bpm":                _f(last["heart_rate_bpm"]),
        "spo2_percent":                  _f(last["spo2_percent"]),
        "systolic_bp_mmhg":              _f(last["systolic_bp_mmhg"]),
        "diastolic_bp_mmhg":             _f(last["diastolic_bp_mmhg"]),
        "body_temperature_c":            _f(last["body_temperature_c"]),
        "breathing_rate_bpm":            _f(last["breathing_rate_bpm"]),
        "radiation_dose_rate_usv_h":     _f(last["radiation_dose_rate_usv_h"]),
        "cumulative_radiation_dose_usv": _f(last["cumulative_radiation_dose_usv"]),
        "muscle_performance_score":      _f(last["muscle_performance_score"]),
        "stress_indicator":              _f(last["stress_indicator"]),
        "activity_level":                _f(last["activity_level"]),
        "sleep_status":                  _f(last["sleep_status"]),
        "scenario_label":                _f(last["scenario_label"]),
        "sensor_status":                 _f(last["sensor_status"]),
    }


@app.get("/astronauts/{astronaut_id}/predict")
def get_prediction(astronaut_id: str, phase: str = "mars_mid", window: int = 12):
    return _compute_prediction(astronaut_id, phase, window)


def _compute_prediction(astronaut_id: str, phase: str, window: int):
    """Shared helper used by both /predict and /predict/all."""
    df = ASTRO[ASTRO.astronaut_id == astronaut_id].reset_index(drop=True)
    if df.empty:
        raise HTTPException(404, f"No data for {astronaut_id}")

    # Read the SAME row /latest just returned, using modulo so that
    # cursor-1 at cursor=0 wraps to the correct starting offset.
    if astronaut_id in _latest_cursor:
        idx = (_latest_cursor[astronaut_id] - 1) % len(df)
    else:
        idx = _START_OFFSETS.get(astronaut_id, 0) % len(df)

    current_row = df.iloc[idx]

    current = {
        "heart_rate_bpm":     _f(current_row["heart_rate_bpm"]),
        "spo2_percent":       _f(current_row["spo2_percent"]),
        "systolic_bp_mmhg":   _f(current_row["systolic_bp_mmhg"]),
        "diastolic_bp_mmhg":  _f(current_row["diastolic_bp_mmhg"]),
        "body_temperature_c": _f(current_row["body_temperature_c"]),
        "breathing_rate_bpm": _f(current_row["breathing_rate_bpm"]),
        "stress_indicator":   _f(current_row["stress_indicator"]),
        "sensor_status":      _f(current_row["sensor_status"]),
    }

    lo = max(0, idx - window + 1)
    hist_df = df.iloc[lo:idx + 1]
    history = to_records(hist_df[["timestamp", "heart_rate_bpm", "spo2_percent",
                                  "systolic_bp_mmhg", "body_temperature_c",
                                  "breathing_rate_bpm", "stress_indicator"]])

    result = ai_predict(astronaut_id, current, history, mission_phase=phase)

    now = current_row["timestamp"]
    result["timestamp"] = now.isoformat()
    result["earth_receives_at"] = (
        now + timedelta(seconds=result["comm_delay_seconds"])
    ).isoformat()
    result["mission_phase"] = phase
    return result


@app.get("/alerts")
def get_alerts(astronaut_id: str = None, severity: str = None):
    df = ALERTS.copy()
    if astronaut_id:
        df = df[df.astronaut_id == astronaut_id]
    if severity:
        df = df[df.simulated_severity == severity]
    return to_records(df)


@app.get("/environment")
def get_environment(last_n: int = 300):
    df = ENV.tail(last_n).drop(columns=["record_id"])
    return to_records(df)


@app.get("/summary")
def summary():
    return {
        "total_astronaut_records": len(ASTRO),
        "total_environment_records": len(ENV),
        "total_alerts": len(ALERTS),
        "astronauts": sorted(ASTRO.astronaut_id.unique().tolist()),
        "time_range": {
            "start": ASTRO.timestamp.min().isoformat(),
            "end":   ASTRO.timestamp.max().isoformat(),
        },
        "is_simulated": True,
        "disclaimer": "All data is synthetic. Not real NASA data. Not medically validated.",
    }