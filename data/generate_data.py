"""
generate_data.py
================
Synthetic data generator for the ER Wait Time Prediction POC.

Generates realistic fake patient arrival records with wait times
that follow patterns observed in real emergency departments:
  - Rush hours (morning + evening) have longer waits
  - Weekends are busier
  - More critical triage levels (lower number) = shorter wait
  - More patients in waiting room = longer wait
  - More doctors on duty = shorter wait

Usage:
    python generate_data.py

Output:
    data/synthetic_er_data.csv  (10,000 rows)
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Config ───────────────────────────────────────────────────────────────────
N_ROWS = 10_000
OUTPUT_DIR = "dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_er_data.csv")


# ── Helper: clamp values to a range ──────────────────────────────────────────
def clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

hour_of_day = np.random.randint(0, 24, N_ROWS)          # 0–23
day_of_week = np.random.randint(0, 7, N_ROWS)           # 0 = Monday … 6 = Sunday
month       = np.random.randint(1, 13, N_ROWS)          # 1–12

is_weekend = (day_of_week >= 5).astype(int)

# Season derived from month
def month_to_season(m):
    if m in [12, 1, 2]:  return "Winter"
    if m in [3, 4, 5]:   return "Spring"
    if m in [6, 7, 8]:   return "Summer"
    return "Autumn"

season = np.array([month_to_season(m) for m in month])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PATIENT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

patient_age = np.random.randint(1, 101, N_ROWS)         # 1–100 years

# Triage level: 1 (critical) → 5 (minor)
# Weighted so minor complaints are more common
triage_level = np.random.choice(
    [1, 2, 3, 4, 5],
    size=N_ROWS,
    p=[0.05, 0.10, 0.25, 0.35, 0.25]
)

# Chief complaint category
complaint_categories = [
    "Trauma", "Cardiac", "Respiratory", "Abdominal",
    "Neurological", "Musculoskeletal", "Infection", "Other"
]
complaint_weights = [0.12, 0.10, 0.12, 0.15, 0.08, 0.18, 0.15, 0.10]
chief_complaint_category = np.random.choice(
    complaint_categories, size=N_ROWS, p=complaint_weights
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  OPERATIONAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

# Doctors on duty: fewer at night, more during day shifts
# Base: 3 doctors; +2 during day (8–20h); −1 on weekends
base_doctors = 3
day_shift_bonus   = ((hour_of_day >= 8) & (hour_of_day < 20)).astype(int) * 2
weekend_penalty   = is_weekend * (-1)
doctors_on_duty   = clamp(
    base_doctors + day_shift_bonus + weekend_penalty
    + np.random.randint(-1, 2, N_ROWS),   # small random variation
    lo=1, hi=8
)

# Nurses: roughly 2× doctors with some noise
nurses_on_duty = clamp(
    doctors_on_duty * 2 + np.random.randint(-1, 2, N_ROWS),
    lo=2, hi=16
)

# Patients in waiting room: more during rush hours and weekends
base_waiting = 10
hour_factor  = np.where(
    (hour_of_day >= 8)  & (hour_of_day < 11), 15,   # morning rush
    np.where(
        (hour_of_day >= 18) & (hour_of_day < 22), 20,   # evening rush
        np.where(hour_of_day < 6, -5, 5)                # quiet night
    )
)
weekend_waiting = is_weekend * 8

patients_in_waiting_room = clamp(
    base_waiting + hour_factor + weekend_waiting
    + np.random.randint(-5, 6, N_ROWS),
    lo=0, hi=60
)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TARGET VARIABLE: wait_time_minutes
#
#  Built from a linear combination of the features above + noise.
#  Each coefficient reflects a realistic real-world relationship.
# ─────────────────────────────────────────────────────────────────────────────

wait_time = (
    # Intercept (baseline wait)
    20.0

    # Triage: level 1 = almost no wait, level 5 = long wait
    + (triage_level - 1) * 12.0          # +0, +12, +24, +36, +48 min

    # Crowding: each extra patient adds ~1.5 min
    + patients_in_waiting_room * 1.5

    # Staffing: each extra doctor reduces wait by ~8 min
    - doctors_on_duty * 8.0

    # Rush hours add extra time
    + np.where(
        (hour_of_day >= 8)  & (hour_of_day < 11), 15.0,
        np.where(
            (hour_of_day >= 18) & (hour_of_day < 22), 20.0,
            0.0
        )
    )

    # Weekends add ~10 min on average
    + is_weekend * 10.0

    # Elderly patients (>70) and very young (<5) slightly longer to process
    + np.where(patient_age > 70, 5.0, 0.0)
    + np.where(patient_age < 5,  5.0, 0.0)

    # Cardiac / Neurological complaints tend to be seen faster
    + np.where(np.isin(chief_complaint_category, ["Cardiac", "Neurological"]), -10.0, 0.0)

    # Random noise: real ERs have lots of unmeasured variability
    + np.random.normal(0, 10, N_ROWS)
)

# Clamp to realistic range: 1 min (immediate) to 240 min (4 hours)
wait_time = clamp(np.round(wait_time, 1), lo=1.0, hi=240.0)

# Rolling average wait in the last hour (simulated as noisy version of current wait)
avg_wait_last_hour = clamp(
    np.round(wait_time + np.random.normal(0, 8, N_ROWS), 1),
    lo=1.0, hi=240.0
)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ASSEMBLE DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

df = pd.DataFrame({
    # Temporal
    "hour_of_day":              hour_of_day,
    "day_of_week":              day_of_week,
    "is_weekend":               is_weekend,
    "month":                    month,
    "season":                   season,

    # Patient
    "patient_age":              patient_age,
    "triage_level":             triage_level,
    "chief_complaint_category": chief_complaint_category,

    # Operational
    "patients_in_waiting_room": patients_in_waiting_room,
    "doctors_on_duty":          doctors_on_duty,
    "nurses_on_duty":           nurses_on_duty,
    "avg_wait_last_hour":       avg_wait_last_hour,

    # Target
    "wait_time_minutes":        wait_time,
})


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SAVE
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅  Dataset saved to: {OUTPUT_FILE}")
print(f"    Shape: {df.shape}")
print()
print("── Target variable summary ─────────────────────────────")
print(df["wait_time_minutes"].describe().round(1).to_string())
print()
print("── Feature preview (first 5 rows) ──────────────────────")
print(df.head().to_string())