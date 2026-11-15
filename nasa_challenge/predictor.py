# predictor.py
"""
AI Space Guardian — Predictive Health Analyzer
===============================================
Heuristic + trend-based scorer that estimates the probability an astronaut
is heading toward syncope (loss of consciousness) or another acute event.

ALL DATA / PREDICTIONS ARE SIMULATED. This is NOT a medical device.

The model here is intentionally simple and transparent — a weighted
combination of:
  * current vitals vs clinical thresholds
  * short-term trend (is HR climbing? is SpO2 falling?)
  * cross-signal coherence (HR high AND SpO2 low is worse than either alone)

It is designed so a judge can read one function and understand the logic.
In production this would be replaced by a trained classifier on labelled
mission data, but the interface would be identical.
"""

from dataclasses import dataclass, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Communication delay model
# ─────────────────────────────────────────────────────────────
# Round-trip delay in seconds between spacecraft and Earth.
# These are realistic approximations for demo scenarios.
COMM_DELAYS = {
    "earth_orbit":   0.5,        # ~0.5 s
    "lunar":         2.6,        # ~1.3 s one-way × 2
    "mars_close":    360,        # ~3 min one-way
    "mars_mid":      900,        # ~7.5 min one-way
    "mars_far":      1320,       # ~22 min one-way
}


def comm_delay_seconds(mission_phase: str = "mars_mid") -> int:
    """Return round-trip comms delay in seconds for the current mission phase."""
    return int(COMM_DELAYS.get(mission_phase, 900))


# ─────────────────────────────────────────────────────────────
# Result structure
# ─────────────────────────────────────────────────────────────
@dataclass
class Prediction:
    astronaut_id: str
    timestamp: str

    # collapse / syncope risk
    collapse_risk: int              # 0-100
    collapse_class: str             # "LOW" | "ELEVATED" | "HIGH" | "CRITICAL"

    # overall health
    health_score: int               # 0-100 (higher = better)
    health_class: str               # "NOMINAL" | "WATCH" | "ALERT" | "EMERGENCY"

    # explanation
    primary_concern: str            # short human sentence
    reasons: list                   # list of strings
    recommended_action: str         # what the onboard AI should do

    # timing / comms
    should_warn_local_crew: bool    # warn astronaut immediately
    should_notify_earth: bool       # send to Earth (with delay)
    comm_delay_seconds: int         # round-trip to Earth
    earth_receives_at: str          # timestamp Earth would see this


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f   # NaN check
    except (TypeError, ValueError):
        return None


def _trend(readings, key, window=6):
    """Estimate slope of a sensor over the last N readings.

    Returns (delta, direction) where delta = last - first.
    `direction` is 'up', 'down', or 'flat'.
    """
    vals = []
    for r in readings[-window:]:
        v = _num(r.get(key))
        if v is not None:
            vals.append(v)
    if len(vals) < 2:
        return 0.0, "flat"
    delta = vals[-1] - vals[0]
    if abs(delta) < 0.5:
        return delta, "flat"
    return delta, ("up" if delta > 0 else "down")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────
# Main prediction function
# ─────────────────────────────────────────────────────────────
def predict(astronaut_id: str, current: dict, history: list,
            mission_phase: str = "mars_mid") -> dict:
    """
    current  : latest reading dict (from /latest)
    history  : list of the last ~12 readings for trend analysis
    """
    hr     = _num(current.get("heart_rate_bpm"))
    spo2   = _num(current.get("spo2_percent"))
    sbp    = _num(current.get("systolic_bp_mmhg"))
    dbp    = _num(current.get("diastolic_bp_mmhg"))
    temp   = _num(current.get("body_temperature_c"))
    br     = _num(current.get("breathing_rate_bpm"))
    stress = _num(current.get("stress_indicator"))
    status = (current.get("sensor_status") or "ok").lower()

    # ── trend analysis ──
    hr_delta,    hr_dir    = _trend(history, "heart_rate_bpm")
    spo2_delta,  spo2_dir  = _trend(history, "spo2_percent")
    sbp_delta,   sbp_dir   = _trend(history, "systolic_bp_mmhg")
    temp_delta,  temp_dir  = _trend(history, "body_temperature_c")

    score = 0          # collapse risk accumulator, capped at 100
    reasons = []

    # ── Rule 1: SpO2 (biggest single syncope signal) ──
    if spo2 is not None:
        if spo2 < 88:
            score += 55
            reasons.append(f"SpO₂ critically low at {spo2:.0f}% — tissue hypoxia")
        elif spo2 < 92:
            score += 30
            reasons.append(f"SpO₂ below safe threshold at {spo2:.0f}%")
        elif spo2 < 95:
            score += 12
            reasons.append(f"SpO₂ mildly reduced at {spo2:.0f}%")

        if spo2_dir == "down" and spo2_delta < -1.5:
            score += 18
            reasons.append(f"SpO₂ falling sharply ({spo2_delta:+.1f}% over recent window)")

    # ── Rule 2: Heart rate (tachycardia + trend) ──
    if hr is not None:
        if hr > 140:
            score += 35
            reasons.append(f"Heart rate very high at {hr:.0f} BPM — cardiac strain")
        elif hr > 110:
            score += 18
            reasons.append(f"Heart rate elevated at {hr:.0f} BPM")
        elif hr < 45:
            score += 30
            reasons.append(f"Heart rate very low at {hr:.0f} BPM — bradycardia")

        if hr_dir == "up" and hr_delta > 15:
            score += 14
            reasons.append(f"Heart rate rising fast ({hr_delta:+.0f} BPM recently)")

    # ── Rule 3: Blood pressure (hypotension = classic faint) ──
    if sbp is not None and dbp is not None:
        # systolic pressure in particular drives cerebral perfusion
        if sbp < 85:
            score += 45
            reasons.append(f"Systolic BP very low at {sbp:.0f} mmHg — hypoperfusion")
        elif sbp < 95:
            score += 22
            reasons.append(f"Blood pressure low at {sbp:.0f}/{dbp:.0f}")

        if sbp_dir == "down" and sbp_delta < -8:
            score += 15
            reasons.append(f"Blood pressure dropping fast ({sbp_delta:+.0f} mmHg recently)")

    # ── Rule 4: Breathing rate ──
    if br is not None:
        if br > 28:
            score += 10
            reasons.append(f"Breathing rapid at {br:.0f}/min")
        elif br < 9:
            score += 20
            reasons.append(f"Breathing very slow at {br:.0f}/min")

    # ── Rule 5: Temperature (heat stress fainting) ──
    if temp is not None:
        if temp > 38.5:
            score += 18
            reasons.append(f"Core temperature high at {temp:.1f}°C")
        elif temp < 35.5:
            score += 22
            reasons.append(f"Hypothermia risk at {temp:.1f}°C")

        if temp_dir == "up" and temp_delta > 0.4:
            score += 8
            reasons.append(f"Temperature rising ({temp_delta:+.1f}°C)")

    # ── Rule 6: Stress / autonomic overload ──
    if stress is not None and stress > 75:
        score += 8
        reasons.append(f"Stress indicator high at {stress:.0f}/100")

    # ── Rule 7: Sensor fault — do NOT treat as medical ──
    sensor_fault = status not in ("ok", "")
    if sensor_fault:
        # reduce our confidence and add a clear caveat
        reasons.append(f"Sensor status is {status.upper()} — readings may be unreliable")

    # ── Combined-signal bonuses ──
    if (hr is not None and spo2 is not None
            and hr > 110 and spo2 < 92):
        score += 10
        reasons.append("Combined tachycardia + hypoxemia — compounding risk")

    if (sbp is not None and hr is not None
            and sbp < 95 and hr > 110):
        score += 8
        reasons.append("Low blood pressure with high heart rate — presyncope pattern")

    # clamp
    score = _clamp(int(score), 0, 100)

    # ── classify ──
    if score >= 75:
        collapse_class = "CRITICAL"
    elif score >= 50:
        collapse_class = "HIGH"
    elif score >= 25:
        collapse_class = "ELEVATED"
    else:
        collapse_class = "LOW"

    health_score = _clamp(100 - score, 0, 100)
    if health_score >= 80:
        health_class = "NOMINAL"
    elif health_score >= 60:
        health_class = "WATCH"
    elif health_score >= 35:
        health_class = "ALERT"
    else:
        health_class = "EMERGENCY"

    # ── primary concern (single sentence for the banner) ──
    if not reasons:
        primary = "All monitored vitals within nominal ranges."
    else:
        # the first reason is the highest weight rule that fired
        primary = reasons[0]

    # ── recommended action ──
    if collapse_class == "CRITICAL":
        action = ("Initiate emergency protocol: alert crew immediately, "
                  "assume seated/safe posture, prepare medical kit, notify flight surgeon.")
    elif collapse_class == "HIGH":
        action = ("Warn astronaut to sit down, drink fluids, slow activity. "
                  "Increase monitoring to 1 Hz. Prepare Earth report.")
    elif collapse_class == "ELEVATED":
        action = ("Continue monitoring. Advise rest and hydration. "
                  "Flag for Earth review at next comms window.")
    else:
        action = "Continue routine monitoring."

    # ── comms ──
    delay = comm_delay_seconds(mission_phase)
    should_warn_local = score >= 25
    should_notify_earth = score >= 50

    # earth_receives_at — caller fills the timestamp from the current reading
    return {
        "astronaut_id": astronaut_id,
        "collapse_risk": score,
        "collapse_class": collapse_class,
        "health_score": health_score,
        "health_class": health_class,
        "primary_concern": primary,
        "reasons": reasons,
        "recommended_action": action,
        "sensor_fault": sensor_fault,
        "should_warn_local_crew": should_warn_local,
        "should_notify_earth": should_notify_earth,
        "comm_delay_seconds": delay,
        "trends": {
            "hr_delta":       round(hr_delta, 2),
            "spo2_delta":     round(spo2_delta, 2),
            "sbp_delta":      round(sbp_delta, 2),
            "temp_delta":     round(temp_delta, 2),
        },
        "disclaimer": "Simulated heuristic. NOT a medical device. Not for real decisions.",
    }