"""
app.py
======
Streamlit web application for ER Wait Time Prediction.

Loads the trained Random Forest model and lets hospital staff
input patient/operational details to get a wait time estimate.

Usage:
    streamlit run app/app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ER Wait Time Estimator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Hide default streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* App background */
  .stApp {
    background: #f7f8fc;
  }

  /* Top header bar */
  .header-bar {
    background: #0a2540;
    color: white;
    padding: 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .header-bar h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    margin: 0;
    color: white;
  }
  .header-bar p {
    margin: 0;
    font-size: 0.85rem;
    color: #94a3b8;
  }

  /* Section cards */
  .section-card {
    background: white;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border: 1px solid #e8ecf4;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #f1f5f9;
  }

  /* Result box */
  .result-low {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1.5px solid #6ee7b7;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
  }
  .result-medium {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1.5px solid #fcd34d;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
  }
  .result-high {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 1.5px solid #fca5a5;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
  }
  .result-number {
    font-family: 'DM Serif Display', serif;
    font-size: 4.5rem;
    line-height: 1;
    color: #0a2540;
  }
  .result-unit {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.3rem;
  }
  .result-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #0a2540;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.8rem;
  }

  /* Metric pills */
  .metric-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 1rem;
  }
  .metric-pill {
    background: #f1f5f9;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.75rem;
    color: #475569;
    border: 1px solid #e2e8f0;
  }
  .metric-pill span {
    font-weight: 600;
    color: #0a2540;
  }

  /* Button */
  .stButton > button {
    background: #0a2540 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
  }
  .stButton > button:hover {
    background: #1e3a5f !important;
  }

  /* Sliders and selects */
  .stSlider > div > div { color: #0a2540; }
  label { font-size: 0.85rem !important; font-weight: 500 !important; color: #374151 !important; }

  /* Info box */
  .info-box {
    background: #eff6ff;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #1e40af;
    margin-top: 1rem;
  }
            
  .streamlit-expanderHeader {
    background-color: white !important;
    border-radius: 12px !important;
    border: 1px solid #e8ecf4 !important;
    color: #0a2540 !important; /* Header text color */
  }

  .streamlit-expanderContent {
    background-color: white !important;
    border: 1px solid #e8ecf4 !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
  }
  
  /* Make the "View Model Logic" text bold and dark */
  .stExpander p {
    color: #0a2540 !important;
    font-weight: 500;
  }
            
</style>
""", unsafe_allow_html=True)

# ── Load model and metadata ───────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "..", "model", "metadata.json")

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
<div class="header-bar">
  <div>🏥</div>
  <div>
    <h1>ER Wait Time Estimator</h1>
    <p>Emergency Department · Predictive Triage Support Tool</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout: left inputs | right result ───────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:

    # ── Section 1: Patient Information ───────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.slider("Patient Age", min_value=1, max_value=100, value=45)
    with col2:
        triage_level = st.selectbox(
            "Triage Level",
            options=[1, 2, 3, 4, 5],
            index=2,
            format_func=lambda x: {
                1: "1 — Immediate (Critical)",
                2: "2 — Emergent",
                3: "3 — Urgent",
                4: "4 — Less Urgent",
                5: "5 — Non-Urgent",
            }[x]
        )

    chief_complaint_category = st.selectbox(
        "Chief Complaint",
        options=["Trauma", "Cardiac", "Respiratory", "Abdominal",
                 "Neurological", "Musculoskeletal", "Infection", "Other"],
        index=7,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Arrival Time ───────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Arrival Time</div>', unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        hour_of_day = st.slider("Hour of Day", 0, 23, 14,
                                format="%d:00")
    with col4:
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            index=0,
            format_func=lambda x: ["Monday","Tuesday","Wednesday",
                                    "Thursday","Friday","Saturday","Sunday"][x]
        )
    with col5:
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=0,
            format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                    "Jul","Aug","Sep","Oct","Nov","Dec"][x-1]
        )

    # Derive is_weekend and season automatically
    is_weekend = 1 if day_of_week >= 5 else 0
    season_map = {12:"Winter",1:"Winter",2:"Winter",
                  3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",
                  9:"Autumn",10:"Autumn",11:"Autumn"}
    season = season_map[month]

    weekend_text = "🔴 Weekend" if is_weekend else "🟢 Weekday"
    st.markdown(f'<div class="info-box">Auto-detected: <strong>{weekend_text}</strong> &nbsp;·&nbsp; Season: <strong>{season}</strong></div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Department Status ─────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Current Department Status</div>', unsafe_allow_html=True)

    col6, col7, col8 = st.columns(3)
    with col6:
        patients_in_waiting_room = st.slider("Patients Waiting", 0, 100, 15)
    with col7:
        doctors_on_duty = st.slider("Doctors on Duty", 1, 8, 3)
    with col8:
        nurses_on_duty = st.slider("Nurses on Duty", 2, 16, 6)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────────────────────
    predict_clicked = st.button("⏱  Estimate Wait Time")


# ── Right column: Result ──────────────────────────────────────────────────────
with right:

    if predict_clicked:
        # Build input dataframe (12 features exactly matching train_models.py)
        input_data = pd.DataFrame([{
            "hour_of_day":              hour_of_day,
            "day_of_week":              day_of_week,
            "is_weekend":               is_weekend,
            "month":                    month,
            "season":                   season,
            "patient_age":              patient_age,
            "triage_level":             triage_level,
            "chief_complaint_category": chief_complaint_category,
            "patients_in_waiting_room": patients_in_waiting_room,
            "doctors_on_duty":          doctors_on_duty,
            "nurses_on_duty":           nurses_on_duty,
        }])

        input_data['staff_to_patient_ratio'] = (input_data['doctors_on_duty'] + input_data['nurses_on_duty']) / (input_data['patients_in_waiting_room'] + 1)

        prediction = float(model.predict(input_data)[0])
        prediction = max(1.0, round(prediction, 1))

        # Colour tier
        if prediction < 30:
            css_class = "result-low"
            status    = "🟢 Short Wait"
            advice    = "Patient can be seen relatively quickly."
        elif prediction < 60:
            css_class = "result-medium"
            status    = "🟡 Moderate Wait"
            advice    = "Consider informing the patient of expected wait."
        else:
            css_class = "result-high"
            status    = "🔴 Long Wait"
            advice    = "High demand — consider patient communication & resource review."

        st.markdown(f"""
        <div class="{css_class}">
          <div class="result-number">{int(prediction)}</div>
          <div class="result-unit">estimated minutes</div>
          <div class="result-label">{status}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="info-box" style="margin-top:0.8rem">{advice}</div>',
                    unsafe_allow_html=True)

        # Model info pills
        res = metadata["results"]
        best = metadata["best_model_name"]
        mae  = res[best]["MAE"]
        r2   = res[best]["R2"]

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-pill">Model: <span>{best}</span></div>
          <div class="metric-pill">Avg error: <span>±{mae} min</span></div>
          <div class="metric-pill">R²: <span>{r2}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Summary of inputs used
        st.markdown('<div class="section-card" style="margin-top:1rem">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)

        triage_labels = {1:"Immediate",2:"Emergent",3:"Urgent",4:"Less Urgent",5:"Non-Urgent"}
        day_labels    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        summary_data = {
            "Age":           patient_age,
            "Triage":        f"Level {triage_level} — {triage_labels[triage_level]}",
            "Complaint":     chief_complaint_category,
            "Arrival":       f"{hour_of_day:02d}:00, {day_labels[day_of_week]}",
            "Season":        season,
            "Status":        f"{patients_in_waiting_room} waiting", # <--- FIXED HERE
            "Staff":         f"{doctors_on_duty} doctors / {nurses_on_duty} nurses",
        }

        for k, v in summary_data.items():
            col_a, col_b = st.columns([1, 2])
            col_a.markdown(f"<span style='font-size:0.78rem;color:#94a3b8'>{k}</span>", unsafe_allow_html=True)
            col_b.markdown(f"<span style='font-size:0.82rem;font-weight:500;color:#0a2540'>{v}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Placeholder before first prediction
        st.markdown("""
        <div class="section-card" style="text-align:center; padding: 3rem 1.5rem;">
          <div style="font-size:3rem; margin-bottom:1rem">⏱</div>
          <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#0a2540; margin-bottom:0.5rem">
            Ready to Estimate
          </div>
          <div style="font-size:0.85rem; color:#94a3b8; line-height:1.6">
            Fill in the patient and department details<br>on the left, then click <strong>Estimate Wait Time</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Model performance summary
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

        res  = metadata["results"]
        best = metadata["best_model_name"]
        for name, metrics in res.items():
            is_best = name == best
            prefix  = "🏆 " if is_best else "　 "
            weight  = "font-weight:600; color:#1e40af" if is_best else "color:#64748b"
            st.markdown(
                f'<div style="font-size:0.82rem;{weight};margin-bottom:0.4rem">'
                f'{prefix}{name} &nbsp;·&nbsp; MAE {metrics["MAE"]} min &nbsp;·&nbsp; R2 {metrics["R2"]}'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
       # Explainable AI Consulting Feature
        with st.expander("📊 View Model Logic (Feature Importance)"):
            st.write("This chart shows which factors our Random Forest heavily relies on to calculate the wait time.")
            
            plot_path = os.path.join(BASE_DIR, "..", "model", "plots", "feature_importance.png")
            if os.path.exists(plot_path):
                st.image(plot_path, width="stretch")
            else:
                st.info("Feature importance plot not found.")