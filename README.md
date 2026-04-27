# 🏥 ER Wait Time Predictor — AI Consulting POC

> **Course:** Inteligência Artificial (L.EIC029) — FEUP 2025/2026  
> **Assignment:** Assignment 2 — ML Consulting POC  
> **Client:** Hospital (TBD — confirm contact person)  
> **Problem:** Predicting Emergency Room wait times to improve patient experience and resource allocation  

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [What We've Already Defined](#what-weve-already-defined)
3. [What Still Needs to Be Decided](#what-still-needs-to-be-decided)
4. [Project Roadmap](#project-roadmap)
5. [Repository Structure](#repository-structure)
6. [Step-by-Step Development Guide](#step-by-step-development-guide)
7. [Deliverables Checklist](#deliverables-checklist)
8. [PowerPoint Slide Checklist](#powerpoint-slide-checklist)

---

## Project Overview

### The Problem
Emergency rooms are chaotic. Patients arrive without appointments, staff numbers fluctuate, and no one knows how long the wait will be. This leads to:
- Patients leaving without being seen (AMA — "Against Medical Advice")
- Inefficient staff allocation
- Poor patient satisfaction scores

### Our Solution
A machine learning model that predicts **how many minutes a patient will wait** before being seen by a doctor, based on information available at the moment they check in.

### Proof of Concept
We are **not** building a full hospital system. We are building:
- A trained regression model
- A simple web app where a receptionist can enter patient info and get a wait time estimate

This proves the concept works. The hospital would then invest in integrating it with their real data.

---

## What we have defined

### The Client
- **Type:** Hospital / Emergency Department
- **Contact Person:** TBD — needs to be someone you know connected to a hospital (receptionist, nurse, administrator, medical student, etc.)
- **Their Pain:** Patients complain about not knowing wait times; staff struggle to manage expectations

### The Problem Type
- **ML Task:** Regression (predicting a continuous number — minutes)
- **Target Variable:** `wait_time_minutes` — the number of minutes from patient check-in until first seen by a doctor

### The General Feature Set (Attributes)
These are the inputs the model will use to make its prediction:

| Feature | Description | Type |
|---|---|---|
| `hour_of_day` | Hour patient arrived (0–23) | Numerical |
| `day_of_week` | Monday=0, Sunday=6 | Numerical |
| `is_weekend` | 1 if Saturday or Sunday | Boolean |
| `month` | Month of the year (1–12) | Numerical |
| `patient_age` | Age of the patient | Numerical |
| `triage_level` | Urgency level 1 (critical) to 5 (minor) | Numerical |
| `chief_complaint_category` | Type of complaint (trauma, cardiac, respiratory, etc.) | Categorical |
| `patients_in_waiting_room` | How many people are currently waiting | Numerical |
| `doctors_on_duty` | Number of doctors currently working | Numerical |
| `nurses_on_duty` | Number of nurses currently working | Numerical |
| `avg_wait_last_hour` | Rolling average wait time in the last 60 min | Numerical |
| `season` | Spring/Summer/Autumn/Winter | Categorical |

### The Three Algorithms to Compare
1. **Linear Regression** — simple baseline, fast, interpretable
2. **Random Forest Regressor** — handles non-linear relationships, robust
3. **Gradient Boosting Regressor (XGBoost)** — typically best performance

### Evaluation Metrics
- **MAE** (Mean Absolute Error) — average minutes off in predictions
- **RMSE** (Root Mean Squared Error) — penalizes large errors more
- **R²** (R-squared) — how much variance is explained by the model

---

## What Still Needs to Be Decided

### ❓ Contact Person
- [ ] Confirm who your hospital contact is and their role
- [ ] Get a short quote or description of the problem from their perspective (for the presentation)

### ❓ Hospital Name / Type
- [ ] Public hospital? Private clinic? University hospital?
- [ ] This affects the stakeholders slide and the organizational challenges

### ❓ Web App Framework
- **Recommended: Streamlit** — easiest for ML demos, pure Python, runs in browser
- Alternative: Flask + HTML frontend (more work, more impressive)
- [ ] Decide as a team

### ❓ Synthetic Data Size
- Recommended: generate **10,000 rows** of fake patient arrivals
- More data = more convincing results
- [ ] Confirm with team

---

## Project Roadmap

```
WEEK 1 — Foundation
├── Confirm client contact person
├── Define features and target variable (done above)
├── Write synthetic data generation script
└── Checkpoint: show teacher the data + problem definition

WEEK 2 — Modelling
├── Train the 3 models on synthetic data
├── Evaluate with MAE, RMSE, R²
├── Plot results (feature importance, actual vs predicted)
└── Checkpoint: show teacher model comparison results

WEEK 3 — Web App
├── Build Streamlit (or Flask) web app
├── Load trained model in the app
├── User inputs patient info → app shows predicted wait time
└── Checkpoint: live demo of the app

WEEK 4 — Presentation
├── Build PowerPoint from template
├── Prepare 10-minute presentation
└── Final submission + presentation in lab
```

---

## Repository Structure

```
er-wait-predictor/
│
├── README.md                   ← This file
│
├── data/
│   └── generate_data.py        ← Script to create synthetic dataset
│   └── synthetic_er_data.csv   ← Generated fake dataset (git-ignored or committed)
│
├── model/
│   └── train_models.py         ← Train and compare the 3 algorithms
│   └── evaluate_models.py      ← Metrics, plots, feature importance
│   └── best_model.pkl          ← Saved trained model (pickle file)
│
├── app/
│   └── app.py                  ← Streamlit web application
│   └── requirements.txt        ← Python dependencies for the app
│
├── notebooks/
│   └── exploration.ipynb       ← Optional: Jupyter notebook for exploration
│
├── presentation/
│   └── assignment2_slides.pptx ← Final PowerPoint (from template)
│
└── requirements.txt            ← All Python dependencies
```

---

## Step-by-Step Development Guide

### Step 1 — Set Up the Repository
```bash
# Clone your existing git repo
git clone <your-repo-url>
cd er-wait-predictor

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install core dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib
pip freeze > requirements.txt
```

---

### Step 2 — Generate Synthetic Data (`data/generate_data.py`)

This is the most important creative step. You need to bake **realistic patterns** into your fake data so the model has something to learn. Key patterns to include:

- **Rush hours:** Wait times are longer at 8–10am and 6–9pm
- **Weekends:** More patients, longer waits
- **Triage level:** Level 1 (critical) gets seen immediately; level 5 waits longest
- **Crowding:** More patients in waiting room = longer wait
- **Staffing:** More doctors = shorter wait

The script should produce a CSV with ~10,000 rows.

---

### Step 3 — Train and Compare Models (`model/train_models.py`)

```
1. Load the CSV
2. Encode categorical variables (one-hot encoding for chief_complaint_category, season)
3. Split into train (80%) and test (20%)
4. Train: Linear Regression, Random Forest, XGBoost
5. Evaluate each on test set: MAE, RMSE, R²
6. Save the best model as best_model.pkl using joblib
7. Generate plots:
   - Bar chart comparing MAE/RMSE across models
   - Actual vs Predicted scatter plot (best model)
   - Feature importance chart (Random Forest or XGBoost)
```

---

### Step 4 — Build the Web App (`app/app.py`)

The app should:
1. Show a title: *"ER Wait Time Estimator"*
2. Have input fields for each feature (sliders, dropdowns, number inputs)
3. Have a **"Predict Wait Time"** button
4. Load `best_model.pkl` and run prediction
5. Display the result clearly: *"Estimated wait time: **42 minutes**"*
6. Optional: Show a color indicator (green < 30min, yellow 30–60min, red > 60min)

```bash
# To run the app locally:
streamlit run app/app.py
```

---

### Step 5 — Build the PowerPoint

Use the template provided on Moodle. See the [PowerPoint Slide Checklist](#powerpoint-slide-checklist) below.

---

## Deliverables Checklist

### Code
- [ ] `data/generate_data.py` — synthetic data generation
- [ ] `data/synthetic_er_data.csv` — the generated dataset
- [ ] `model/train_models.py` — training and evaluation script
- [ ] `model/best_model.pkl` — saved best model
- [ ] `app/app.py` — working Streamlit web app
- [ ] `requirements.txt` — all dependencies listed

### Presentation
- [ ] PowerPoint follows template structure
- [ ] All slides completed (see checklist below)
- [ ] Results slide includes actual numbers (MAE, RMSE, R²) for all 3 models

### Process
- [ ] All group members present at every checkpoint
- [ ] Weekly checkpoint discussed with teacher in lab session
- [ ] Final presentation prepared and rehearsed

---

## PowerPoint Slide Checklist

| Slide | Content | Status |
|---|---|---|
| **Customer** | Hospital name, contact person name & role | ⬜ |
| **Problem** | ER wait time unpredictability; current approach (manual, no prediction) | ⬜ |
| **Available Data** | Patient arrivals log, triage records, staffing schedules | ⬜ |
| **Solution Overview** | Web app + ML model + possible future integrations (SMS alerts, dashboard) | ⬜ |
| **Predictive Model** | Target: `wait_time_minutes`; list all features/attributes | ⬜ |
| **Empirical Study: Setup** | Synthetic data description, 80/20 split, MAE/RMSE/R² metrics | ⬜ |
| **Empirical Study: Results** | Table comparing 3 algorithms; best model highlighted | ⬜ |
| **Stakeholders** | ER doctors, nurses, receptionists, hospital IT, hospital management | ⬜ |
| **Expected Challenges** | Technical: data quality, real-time data feeds. Organizational: staff adoption, trust in AI | ⬜ |
| **Unavailable Data** | Real historical wait times, patient medical history, lab result delays | ⬜ |
| **Ethical Risks** | Bias by age/demographics, over-reliance on model, privacy of patient data | ⬜ |

---

## Key Ethical Risks to Address (Slide Content)

- **Demographic bias:** If training data reflects past inequities (e.g. certain groups historically waited longer), the model may perpetuate this
- **Over-reliance:** Staff might stop using clinical judgment if they trust the model too much
- **Privacy:** Patient data (even for a POC) must be handled under GDPR / hospital data protection rules
- **Transparency:** Can the model explain *why* it predicted a long wait? (Explainability)

---

## Tech Stack Summary

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Generation | `pandas`, `numpy` |
| ML Models | `scikit-learn`, `xgboost` |
| Visualizations | `matplotlib`, `seaborn` |
| Model Saving | `joblib` |
| Web App | `streamlit` |
| Version Control | Git |

---

*README last updated: April 2026*  
*Course: L.EIC029 — Inteligência Artificial, FEUP*