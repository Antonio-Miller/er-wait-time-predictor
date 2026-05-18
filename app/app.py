"""
app.py  –  ER Wait Time Estimator
Streamlit UI: clean, calm, fast to operate under pressure.
Run:  streamlit run app/app.py
"""

import os, json, sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st  # type: ignore

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ER Wait Time",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #f4f6f8; max-width: 780px; margin: 0 auto; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Header ── */
.er-header {
    display: flex; align-items: center; gap: 14px;
    padding: 20px 24px; background: #0d7377;
    border-radius: 14px; margin-bottom: 28px;
}
.er-header-title {
    font-size: 1.35rem; font-weight: 600; color: #ffffff;
    letter-spacing: -0.01em; margin: 0;
}
.er-header-sub {
    font-size: 0.8rem; color: rgba(255,255,255,0.65);
    margin: 2px 0 0; font-weight: 400;
}

/* ── Cards ── */
.card {
    background: #ffffff; border-radius: 12px;
    padding: 20px 22px 22px; margin-bottom: 16px;
    border: 1px solid #e2e8f0;
}
.card-title {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748b;
    margin-bottom: 16px; padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
}

/* ── Labels ── */
label, .stSelectbox label, .stNumberInput label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #1e293b !important;
}

/* ── Number inputs ── */
.stNumberInput input {
    font-size: 1.05rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    text-align: center !important;
    border-radius: 8px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #f8fafc !important;
}
.stNumberInput input:focus {
    border-color: #0d7377 !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(13,115,119,0.12) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    border-radius: 8px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #f8fafc !important;
    font-size: 0.88rem !important;
    color: #0f172a !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #0d7377 !important;
    box-shadow: 0 0 0 3px rgba(13,115,119,0.12) !important;
}

/* ── Pills ── */
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 500; margin-top: 8px;
}
.pill-info    { background: #e0f2f1; color: #0f5c5e; border: 1px solid #b2dfdb; }
.pill-warn    { background: #fff3e0; color: #7c4400; border: 1px solid #ffe0b2; }
.pill-neutral { background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0; }

/* ── Button ── */
.stButton > button {
    background: #0d7377 !important; color: #ffffff !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 600 !important;
    width: 100% !important; letter-spacing: 0.01em !important;
    transition: background 0.18s ease !important; margin-top: 8px !important;
}
.stButton > button:hover { background: #0a5c60 !important; }

/* ── Result ── */
.result-box {
    border-radius: 14px; padding: 28px 24px;
    text-align: center; margin-bottom: 16px;
}
.result-low    { background: #e6f7f7; border: 2px solid #0d7377; }
.result-medium { background: #fff8e6; border: 2px solid #d4a017; }
.result-high   { background: #fff0f0; border: 2px solid #c0392b; }

.result-mins {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 5rem; font-weight: 600; line-height: 1; color: #0f172a;
}
.result-unit  { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
.result-status {
    display: inline-block; margin-top: 12px; padding: 4px 14px;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.status-low    { background: #0d7377; color: #fff; }
.status-medium { background: #d4a017; color: #fff; }
.status-high   { background: #c0392b; color: #fff; }
.advice        { font-size: 0.82rem; color: #475569; margin-top: 10px; }

/* ── Summary ── */
.summary-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 7px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.83rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-key   { color: #94a3b8; font-weight: 400; }
.summary-value { color: #0f172a; font-weight: 500; text-align: right; }

/* ── Badges ── */
.badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.badge {
    background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 999px; padding: 3px 10px;
    font-size: 0.74rem; color: #334155;
}
.badge strong { color: #0d7377; }

/* ── Perf table ── */
.perf-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid #f8fafc; font-size: 0.82rem;
}
.perf-row:last-child { border-bottom: none; }
.best-tag {
    background: #0d7377; color: #fff; border-radius: 999px;
    font-size: 0.65rem; font-weight: 600; padding: 1px 7px;
    margin-left: 6px; letter-spacing: 0.06em; text-transform: uppercase;
}

/* ── Idle ── */
.idle-box { text-align: center; padding: 40px 24px; }
.idle-icon  { font-size: 2.8rem; margin-bottom: 10px; }
.idle-title { font-size: 1.05rem; font-weight: 600; color: #334155; margin-bottom: 6px; }
.idle-sub   { font-size: 0.83rem; color: #94a3b8; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Load model & metadata ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "..", "model", "metadata.json")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(META_PATH) as f:
        return json.load(f)

model    = load_model()
metadata = load_metadata()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="er-header">
  <div style="font-size:2rem;line-height:1">🏥</div>
  <div>
    <p class="er-header-title">ER Wait Time Estimator</p>
    <p class="er-header-sub">Emergency Department · Predictive Triage Support</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. PATIENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="card"><div class="card-title">Patient</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    patient_age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=45, step=1
    )
with col2:
    triage_level = st.selectbox(
        "Triage Level",
        options=[1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: {
            1: "1 · Immediate",
            2: "2 · Emergent",
            3: "3 · Urgent",
            4: "4 · Less Urgent",
            5: "5 · Non-Urgent",
        }[x],
    )

chief_complaint_category = st.selectbox(
    "Chief Complaint",
    options=["Trauma", "Cardiac", "Respiratory", "Abdominal",
             "Neurological", "Musculoskeletal", "Infection", "Other"],
    index=7,
)

if patient_age <= 5:
    st.markdown('<div class="pill pill-info">🧒 Paediatric fast-track</div>', unsafe_allow_html=True)
elif patient_age <= 12:
    st.markdown('<div class="pill pill-info">🧒 Child priority</div>', unsafe_allow_html=True)
elif patient_age >= 75:
    st.markdown('<div class="pill pill-info">👴 Elderly priority</div>', unsafe_allow_html=True)
elif patient_age >= 65:
    st.markdown('<div class="pill pill-info">👴 Elderly priority</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. ARRIVAL TIME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="card"><div class="card-title">Arrival Time</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3, gap="medium")
with col3:
    hour_of_day = st.number_input(
        "Hour (0–23)", min_value=0, max_value=23, value=14, step=1, format="%d"
    )
with col4:
    day_of_week = st.selectbox(
        "Day",
        options=list(range(7)),
        index=0,
        format_func=lambda x: ["Monday","Tuesday","Wednesday",
                                "Thursday","Friday","Saturday","Sunday"][x],
    )
with col5:
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=0,
        format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"][x-1],
    )

is_weekend = 1 if day_of_week >= 5 else 0
season_map = {12:"Winter",1:"Winter",2:"Winter",
              3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",
              9:"Autumn",10:"Autumn",11:"Autumn"}
season = season_map[month]

col_a, col_b = st.columns(2)
with col_a:
    wknd_cls = "pill-warn" if is_weekend else "pill-neutral"
    st.markdown(
        f'<div class="pill {wknd_cls}">📅 {"Weekend" if is_weekend else "Weekday"}</div>',
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(f'<div class="pill pill-neutral">🌤 {season}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. DEPARTMENT STATUS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="card"><div class="card-title">Department Status</div>', unsafe_allow_html=True)

col6, col7, col8 = st.columns(3, gap="medium")
with col6:
    patients_in_waiting_room = st.number_input(
        "Patients waiting", min_value=0, max_value=100, value=15, step=1
    )
with col7:
    doctors_on_duty = st.number_input(
        "Doctors on duty", min_value=1, max_value=10, value=3, step=1
    )
with col8:
    nurses_on_duty = st.number_input(
        "Nurses on duty", min_value=1, max_value=20, value=6, step=1
    )

ratio = (doctors_on_duty + nurses_on_duty) / (patients_in_waiting_room + 1)
ratio_cls = "pill-info" if ratio >= 0.5 else "pill-warn"
st.markdown(
    f'<div class="pill {ratio_cls}">Staff-to-patient ratio: <strong>{ratio:.2f}</strong></div>',
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
predict_clicked = st.button("⏱  Estimate Wait Time")
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RESULT
# ══════════════════════════════════════════════════════════════════════════════
if predict_clicked:
    input_data = pd.DataFrame([{
        "hour_of_day":              int(hour_of_day),
        "day_of_week":              day_of_week,
        "is_weekend":               is_weekend,
        "month":                    month,
        "patient_age":              int(patient_age),
        "triage_level":             triage_level,
        "chief_complaint_category": chief_complaint_category,
        "patients_in_waiting_room": int(patients_in_waiting_room),
        "doctors_on_duty":          int(doctors_on_duty),
        "nurses_on_duty":           int(nurses_on_duty),
    }])
    AGE_GROUPS = ["0-4", "5-17", "18-59", "60-70", "71+"]

    input_data["staff_to_patient_ratio"] = (
        (input_data["doctors_on_duty"] + input_data["nurses_on_duty"])
        / (input_data["patients_in_waiting_room"] + 1)
    )
    input_data["age_group"] = pd.cut(
        input_data["patient_age"],
        bins=[-1, 4, 17, 59, 70, 120],
        labels=AGE_GROUPS,
    )
    age = input_data["patient_age"].astype(float)
    input_data["age_priority_score"] = np.clip(np.abs(age - 38.0) / 38.0, 0.0, 1.0)
    input_data["hour_sin"]  = np.sin(2 * np.pi * input_data["hour_of_day"] / 24)
    input_data["hour_cos"]  = np.cos(2 * np.pi * input_data["hour_of_day"] / 24)
    input_data["month_sin"] = np.sin(2 * np.pi * input_data["month"] / 12)
    input_data["month_cos"] = np.cos(2 * np.pi * input_data["month"] / 12)

    prediction = max(1.0, round(float(model.predict(input_data)[0]), 1))

    if prediction < 30:
        box_cls, status_cls, status_lbl = "result-low",    "status-low",    "Short Wait"
        advice = "Patient can be seen shortly."
    elif prediction < 60:
        box_cls, status_cls, status_lbl = "result-medium", "status-medium", "Moderate Wait"
        advice = "Let the patient know the expected wait time."
    else:
        box_cls, status_cls, status_lbl = "result-high",   "status-high",   "Long Wait"
        advice = "High demand — review resources and communicate delays."

    st.markdown(f"""
    <div class="result-box {box_cls}">
      <div class="result-mins">{int(prediction)}</div>
      <div class="result-unit">estimated minutes</div>
      <span class="result-status {status_cls}">{status_lbl}</span>
      <div class="advice">{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    res  = metadata["results"]
    best = metadata["best_model_name"]
    mae  = res[best]["MAE"]
    r2   = res[best]["R2"]
    st.markdown(f"""
    <div class="badge-row">
      <div class="badge">Model <strong>{best}</strong></div>
      <div class="badge">Avg error <strong>±{mae} min</strong></div>
      <div class="badge">R² <strong>{r2}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    triage_labels = {1:"Immediate",2:"Emergent",3:"Urgent",4:"Less Urgent",5:"Non-Urgent"}
    day_labels    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    summary = {
        "Age":          f"{int(patient_age)} yrs",
        "Triage":       f"Level {triage_level} — {triage_labels[triage_level]}",
        "Complaint":    chief_complaint_category,
        "Arrival":      f"{int(hour_of_day):02d}:00 · {day_labels[day_of_week]}",
        "Season":       season,
        "Waiting room": f"{int(patients_in_waiting_room)} patients",
        "Staff":        f"{int(doctors_on_duty)} doctors · {int(nurses_on_duty)} nurses",
    }
    rows_html = "".join(
        f'<div class="summary-row">'
        f'<span class="summary-key">{k}</span>'
        f'<span class="summary-value">{v}</span>'
        f'</div>'
        for k, v in summary.items()
    )
    st.markdown(
        f'<div class="card"><div class="card-title">Summary</div>{rows_html}</div>',
        unsafe_allow_html=True,
    )

else:
    st.markdown("""
    <div class="card">
      <div class="idle-box">
        <div class="idle-icon">⏱</div>
        <div class="idle-title">Ready to estimate</div>
        <div class="idle-sub">Fill in the fields above and tap <strong>Estimate Wait Time</strong>.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    res  = metadata["results"]
    best = metadata["best_model_name"]
    rows = ""
    for name, m in res.items():
        tag = '<span class="best-tag">Best</span>' if name == best else ""
        rows += (
            f'<div class="perf-row">'
            f'<span style="color:#0f172a;font-weight:{"600" if name==best else "400"}">{name}{tag}</span>'
            f'<span style="color:#64748b">MAE {m["MAE"]} min &nbsp;·&nbsp; R² {m["R2"]}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="card"><div class="card-title">Model Performance</div>{rows}</div>',
        unsafe_allow_html=True,
    )

    plot_path = os.path.join(BASE_DIR, "..", "model", "plots", "feature_importance.png")
    if os.path.exists(plot_path):
        with st.expander("📊 Feature Importance"):
            st.image(plot_path, width="stretch")