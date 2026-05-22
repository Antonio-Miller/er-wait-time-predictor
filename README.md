# ER Wait Time Predictor — AI Consulting POC

> **Course:** Inteligência Artificial (L.EIC029) — FEUP 2025/2026
> **Assignment:** Assignment 2 — ML Consulting POC
> **Client:** Centro Hospitalar Universitário de São João (CHUSJ), Porto
> **Contact:** Dr. Ana Ferreira — Head of Emergency Department Operations
> **Problem:** Predicting Emergency Room wait times to improve patient experience and resource allocation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Feature Set](#feature-set)
3. [Results](#results)
4. [Repository Structure](#repository-structure)
5. [Step-by-Step Development Guide](#step-by-step-development-guide)
6. [Deliverables Checklist](#deliverables-checklist)
7. [Understanding Decision Trees (for oral defence)](#understanding-decision-trees-for-oral-defence)
8. [Tech Stack](#tech-stack)

---

## Project Overview

### The Problem
Emergency rooms are chaotic. Patients arrive without appointments, staff numbers fluctuate, and no one knows how long the wait will be. This leads to:
- Patients leaving without being seen
- Inefficient staff allocation
- Poor patient satisfaction scores

### Our Solution
A machine learning model that predicts **how many minutes a patient will wait** before being seen by a doctor, based on information available at the moment they check in.

A **Streamlit web app** allows receptionists to input patient data and get an instant prediction, colour-coded by urgency:
- 🟢 Green — under 30 minutes
- 🟠 Orange — 30–60 minutes
- 🔴 Red — over 60 minutes

### Why This Is a POC (Not a Full System)
We are **not** building a full hospital system. We are building:
- A trained regression model
- A simple web app for demo and validation purposes

---

## Feature Set

### Raw Input Features

| Feature | Description | Type |
|---|---|---|
| `hour_of_day` | Hour patient arrived (0–23) | Numerical |
| `day_of_week` | Monday=0, Sunday=6 | Numerical |
| `is_weekend` | 1 if Saturday or Sunday | Boolean |
| `month` | Month of the year (1–12) | Numerical |
| `patient_age` | Age of the patient (0–120) | Numerical |
| `triage_level` | Urgency level 1 (critical) to 5 (minor) | Numerical |
| `chief_complaint_category` | Type of complaint (8 categories) | Categorical |
| `patients_in_waiting_room` | How many people are currently waiting | Numerical |
| `doctors_on_duty` | Number of doctors currently working | Numerical |
| `nurses_on_duty` | Number of nurses currently working | Numerical |

### Engineered Features (derived automatically during training and prediction)

| Feature | How it's computed |
|---|---|
| `staff_to_patient_ratio` | `(doctors + nurses) / (waiting_room + 1)` |
| `age_priority_score` | Captures U-shaped priority (children and elderly prioritised) |
| `hour_sin` / `hour_cos` | Circular encoding of hour so 23:00 and 00:00 are treated as close |
| `month_sin` / `month_cos` | Circular encoding of month for seasonality |
| `age_group` | Categorical bucket: 0-4 / 5-17 / 18-59 / 60-70 / 71+ |

> **Note:** `season` is displayed in the web app UI for context but is **not** a model input — it is fully captured by `month` and its circular encodings.

### The Three Algorithms

| Algorithm | Role | Why |
|---|---|---|
| **Linear Regression** | Baseline | Simple, fast, fully interpretable |
| **Decision Tree Regressor** | Core model | Taught in class — must understand and defend this |
| **Random Forest Regressor** | Best performer | Natural extension: combines 200 trees to reduce variance |

---

## Results

### Dataset
- **10,000 synthetic rows** generated via `data/generate_data.py`
- Based on M/M/c queuing theory for realistic arrival and service rate simulation
- **80/20 train/test split** (8,000 train / 2,000 test)
- **5-fold cross-validation** used for model selection

### Overall Wait Time Distribution
| Statistic | Value |
|---|---|
| Mean | 58.9 min |
| Std | 38.4 min |
| Min | 1.0 min |
| 25th percentile | 33.6 min |
| Median | 52.4 min |
| 75th percentile | 76.7 min |
| Max | 279.0 min |

### Model Comparison

| Model | CV MAE | Test MAE | Improvement vs Baseline |
|---|---|---|---|
| Naive Baseline (mean) | — | 28.31 min | 0.0% |
| Linear Regression | 12.71 ± 0.27 | 12.26 min | 56.7% |
| Decision Tree | 10.85 ± 0.13 | 10.49 min | 63.0% |
| **Random Forest** | **9.26 ± 0.06** | **8.72 min** | **69.2%** |

The **Random Forest** is the best model and is saved as `model/best_model.pkl`.

### Synthetic Data Validation

Key patterns confirmed in the generated data:

| Check | Result |
|---|---|
| Weekend vs Weekday mean wait | 82.3 min vs 49.7 min ✅ |
| Peak hours (07–10h, 17–20h) vs quiet (03–06h) | 67.0–67.4 min vs 52.5 min ✅ |
| Winter vs other seasons | 72.1 min vs ~54–55 min ✅ |
| Trauma/Cardiac vs Musculoskeletal/Other | 47.4–47.8 min vs 67.1–70.2 min ✅ |

### Mean Wait by Triage Level

| Triage | Meaning | Mean Wait |
|---|---|---|
| 1 | Immediate | 3.1 min |
| 2 | Emergent | 27.0 min |
| 3 | Urgent | 42.8 min |
| 4 | Less Urgent | 62.8 min |
| 5 | Non-Urgent | 92.6 min |

### Age Priority Check (Triage 2–5 only)

| Age Group | Mean Wait |
|---|---|
| Under 5 | 53.6 min |
| 5–17 | 58.9 min |
| **18–59** | **71.7 min** ← longest |
| 60–70 | 59.8 min |
| 71+ | 55.8 min |

Adults (18–59) wait longest; children and elderly are prioritised, as expected.

### Age Sanity Check (model predictions, triage 3, weekend evening, January)

| Patient | Predicted Wait |
|---|---|
| 2 years old | 57.3 min |
| **35 years old** | **61.5 min** ← longest |
| 80 years old | 57.4 min |

The model correctly learns the clinical priority pattern.

---

## Repository Structure

```
assignment2/
│
├── README.md
│
├── data/
│   ├── generate_data.py          ← Synthetic dataset generation (M/M/c queuing model)
│   └── synthetic_er_data.csv     ← Generated dataset (10,000 rows × 16 columns)
│
├── model/
│   ├── train_models.py           ← Feature engineering, training, evaluation, plots
│   ├── best_model.pkl            ← Saved best model pipeline (Random Forest)
│   ├── metadata.json             ← Feature names + results summary (used by app)
│   └── plots/
│       ├── model_comparison.png      ← MAE / RMSE / R² bar charts
│       ├── actual_vs_predicted.png   ← Scatter plot for best model
│       ├── feature_importance.png    ← Top 15 features by importance
│       └── decision_tree.png         ← Depth=3 tree (for slides/oral defence)
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

Outputs `data/synthetic_er_data.csv` (10,000 rows × 16 columns) and prints a validation summary to the terminal.

### Step 3 — Train and Evaluate Models

```bash
python3 model/train_models.py
```

Outputs:
- Printed results table (CV MAE, Test MAE, R², improvement %)
- `model/best_model.pkl`
- `model/metadata.json`
- All 4 plots in `model/plots/`

### Step 4 — Run the Web App

```bash
streamlit run app/app.py
```

---

## Deliverables Checklist

### Code
- [x] `data/generate_data.py`
- [x] `data/synthetic_er_data.csv`
- [x] `model/train_models.py`
- [x] `model/best_model.pkl`
- [x] `model/metadata.json`
- [x] `model/plots/` — all 4 plots
- [x] `app/app.py`
- [x] `requirements.txt`

### Presentation
- [x] PowerPoint follows template structure
- [x] All slides completed
- [x] Results slide includes numbers from the table above

---

## Understanding Decision Trees (for oral defence)

The professor will ask about this. Key concepts to know:

- **How it works:** At each node, the tree splits the data on the feature that most reduces prediction error (measured by MSE reduction). It keeps splitting until it reaches `max_depth`.
- **Why depth matters:** A deeper tree memorises the training data (overfitting). We use `max_depth=8` as a balance between expressiveness and generalisation.
- **Why Random Forest beats a single tree:** It trains 200 trees on random subsets of data and features, then averages their predictions — this reduces variance without increasing bias.
- **The `decision_tree.png` plot** shows a depth=3 version for readability — use this in slides to explain the concept visually without overwhelming the audience.
- **CV MAE ± STD:** The small standard deviation on Random Forest (±0.06) means it is extremely stable across folds — a strong sign it will generalise well.
- **Feature importance:** Triage level, patients in waiting room, and staff-to-patient ratio are the top predictors — this matches clinical intuition and is easy to explain to a non-technical audience.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Data Generation | `pandas`, `numpy`, `scipy` |
| ML Models | `scikit-learn` |
| Visualisations | `matplotlib`, `seaborn` |
| Model Saving | `joblib` |
| Web App | `streamlit` |
| Version Control | Git |

---

*README last updated: May 2026 — Course: L.EIC029, FEUP*