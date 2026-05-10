# ER Wait Time Predictor — AI Consulting POC

> **Course:** Inteligência Artificial (L.EIC029) — FEUP 2025/2026  
> **Assignment:** Assignment 2 — ML Consulting POC  
> **Client:** Hospital (TBD — confirm contact person)  
> **Problem:** Predicting Emergency Room wait times to improve patient experience and resource allocation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What's Already Defined](#whats-already-defined)
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
- Patients leaving without being seen
- Inefficient staff allocation
- Poor patient satisfaction scores

### Our Solution
A machine learning model that predicts **how many minutes a patient will wait** before being seen by a doctor, based on information available at the moment they check in.

### Why This Is a POC (Not a Full System)
We are **not** building a full hospital system. We are building:
- A trained regression model
- A simple web app where a receptionist can enter patient info and get a wait time estimate

---

## What's Already Defined

### The Client
- **Type:** Hospital / Emergency Department
- **Contact Person:** TBD
- **Their Pain:** Patients complain about not knowing wait times; staff struggle to manage expectations

### The Problem Type
- **ML Task:** Regression (predicting a continuous number — minutes)
- **Target Variable:** `wait_time_minutes`

### The Feature Set (11 Features)

| Feature | Description | Type |
|---|---|---|
| `hour_of_day` | Hour patient arrived (0–23) | Numerical |
| `day_of_week` | Monday=0, Sunday=6 | Numerical |
| `is_weekend` | 1 if Saturday or Sunday | Boolean |
| `month` | Month of the year (1–12) | Numerical |
| `season` | Spring/Summer/Autumn/Winter | Categorical |
| `patient_age` | Age of the patient | Numerical |
| `triage_level` | Urgency level 1 (critical) to 5 (minor) | Numerical |
| `chief_complaint_category` | Type of complaint (trauma, cardiac, etc.) | Categorical |
| `patients_in_waiting_room` | How many people are currently waiting | Numerical |
| `doctors_on_duty` | Number of doctors currently working | Numerical |
| `nurses_on_duty` | Number of nurses currently working | Numerical |

### The Three Algorithms

| Algorithm | Role | Why |
|---|---|---|
| **Linear Regression** | Baseline | Simple, fast, fully interpretable |
| **Decision Tree Regressor** | Core model | Taught in class — must understand and defend this |
| **Random Forest Regressor** | Best performer | Natural extension: "what if we combine many trees?" |

> **Note:** The professor specifically suggested Decision Trees over XGBoost so the group can properly explain and defend the algorithm during the oral discussion.

### Results (on synthetic data)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~ 31.18 min | ~ 49.14 min | ~ 0.6264 |
| Decision Tree | ~ 20.77 min | ~ 37.10 min | ~ 0.7870 |
| **Random Forest** | **~ 15.51 min** | **~ 29.14 min** | **~ 0.8686** |

---

## What Still Needs to Be Decided

- [ ] **Contact Person** — confirm who your hospital contact is and their role
- [ ] **Hospital Name / Type** — public? private? affects the stakeholders slide

---

## Project Roadmap

```text
WEEK 1 — Foundation
├── Confirm client contact person
├── Define features and target variable ✅
├── Write synthetic data generation script ✅
└── Checkpoint: show teacher data + problem definition

WEEK 2 — Modelling
├── Train the 3 models on synthetic data ✅
├── Evaluate with MAE, RMSE, R² ✅
├── Generate plots ✅
└── Checkpoint: show teacher model comparison results

WEEK 3 — Web App
├── Build Streamlit web app ✅
├── Load trained model in the app ✅
├── User inputs patient info → app shows predicted wait time ✅
└── Checkpoint: live demo of the app

WEEK 4 — Presentation
├── Build PowerPoint from template
├── Prepare 10-minute presentation
└── Final submission + presentation in lab
```

---

## Repository Structure

```
assignment2/
│
├── README.md
│
├── data/
│   ├── generate_data.py          ← Script to create synthetic dataset
│   └── synthetic_er_data.csv     ← Generated dataset (10,000 rows)
│
├── model/
│   ├── train_models.py           ← Train and compare the 3 algorithms
│   ├── best_model.pkl            ← Saved best model (Random Forest)
│   ├── metadata.json             ← Feature names + results summary
│   └── plots/
│       ├── model_comparison.png
│       ├── actual_vs_predicted.png
│       ├── feature_importance.png
│       └── decision_tree.png     ← Visual of shallow tree (for slides)
│
├── app/
│   └── app.py                    ← Streamlit web application
│
├── presentation/
│   └── assignment2_slides.pptx
│
└── requirements.txt
```

---

## Step-by-Step Development Guide

### Step 1 — Set Up the Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Generate Synthetic Data

```bash
python3 data/generate_data.py
```

### Step 3 — Train and Evaluate Models

```bash
python3 model/train_models.py
```

### Step 4 — Run the Web App

```bash
streamlit run app/app.py
```

---

## Deliverables Checklist

### Code
- [x] `data/generate_data.py`
- [x] `data/synthetic_er_data.csv`
- [x] `train_models.py`
- [x] `model/best_model.pkl`
- [x] `model/plots/` — all 4 plots
- [x] `app/app.py`
- [x] `requirements.txt`

### Presentation
- [ ] PowerPoint follows template structure
- [ ] All slides completed
- [ ] Results slide includes numbers from the table above

---

## PowerPoint Slide Checklist

| Slide | Content | Status |
|---|---|---|
| **Customer** | Hospital name, contact person name & role | ⬜ |
| **Problem** | ER wait time unpredictability; current approach | ⬜ |
| **Available Data** | Patient arrivals log, triage records, staffing schedules | ⬜ |
| **Solution Overview** | Web app + ML model + future integrations | ⬜ |
| **Predictive Model** | Target: `wait_time_minutes`; list all features | ⬜ |
| **Empirical Study: Setup** | Synthetic data (10k rows), 80/20 split, metrics | ⬜ |
| **Empirical Study: Results** | Table comparing 3 algorithms + plots | ⬜ |
| **Stakeholders** | ER doctors, nurses, receptionists, IT, management | ⬜ |
| **Expected Challenges** | Technical: data quality. Organisational: staff adoption | ⬜ |
| **Unavailable Data** | Real historical wait times, lab result delays | ⬜ |
| **Ethical Risks** | Demographic bias, over-reliance, GDPR | ⬜ |

---

## Understanding Decision Trees (for oral defence)

The professor will ask about this. Key concepts to know:

- **How it works:** At each node, the tree splits the data on the feature that most reduces prediction error. It keeps splitting until it reaches `max_depth`.
- **Why depth matters:** A deeper tree memorises the training data (overfitting). We use `max_depth=8` as a balance.
- **Why Random Forest beats a single tree:** It trains 200 trees on random subsets of data and averages their predictions — this reduces variance and overfitting.
- **The `decision_tree.png` plot** shows a depth=3 version — use this in slides to explain the concept visually.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Data Generation | `pandas`, `numpy` |
| ML Models | `scikit-learn` |
| Visualisations | `matplotlib`, `seaborn` |
| Model Saving | `joblib` |
| Web App | `streamlit` |
| Version Control | Git |

---

*README last updated: May 2026 — Course: L.EIC029, FEUP*