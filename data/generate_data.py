"""
generate_data.py
================
Synthetic data generator for the ER Wait Time Prediction POC.
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Config ───────────────────────────────────────────────────────────────────
N_ROWS = 10_000
OUTPUT_DIR = "data" 
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_er_data.csv")

def clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
hour_of_day = np.random.randint(0, 24, N_ROWS)
day_of_week = np.random.randint(0, 7, N_ROWS)
month       = np.random.randint(1, 13, N_ROWS)
is_weekend  = (day_of_week >= 5).astype(int)

def month_to_season(m):
    if m in [12, 1, 2]:  return "Winter"
    if m in [3, 4, 5]:   return "Spring"
    if m in [6, 7, 8]:   return "Summer"
    return "Autumn"
season = np.array([month_to_season(m) for m in month])

# ─────────────────────────────────────────────────────────────────────────────
# 2. PATIENT FEATURES
# ─────────────────────────────────────────────────────────────────────────────
patient_age = np.random.randint(1, 101, N_ROWS)
triage_level = np.random.choice([1, 2, 3, 4, 5], size=N_ROWS, p=[0.05, 0.10, 0.25, 0.35, 0.25])

complaint_categories = ["Trauma", "Cardiac", "Respiratory", "Abdominal", "Neurological", "Musculoskeletal", "Infection", "Other"]
complaint_weights = [0.12, 0.10, 0.12, 0.15, 0.08, 0.18, 0.15, 0.10]
chief_complaint_category = np.random.choice(complaint_categories, size=N_ROWS, p=complaint_weights)

# ─────────────────────────────────────────────────────────────────────────────
# 3. OPERATIONAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
base_doctors = 3
day_shift_bonus = ((hour_of_day >= 8) & (hour_of_day < 20)).astype(int) * 2
weekend_penalty = is_weekend * (-1)
doctors_on_duty = clamp(base_doctors + day_shift_bonus + weekend_penalty + np.random.randint(-1, 2, N_ROWS), lo=1, hi=8)

nurses_on_duty = clamp(doctors_on_duty * 2 + np.random.randint(-1, 2, N_ROWS), lo=2, hi=16)

# INCREASED RUSH HOUR BUSYNESS
base_waiting = 8
hour_factor = np.where(
    (hour_of_day >= 9)  & (hour_of_day <= 13), 25,   # Morning rush (much busier now)
    np.where((hour_of_day >= 18) & (hour_of_day <= 22), 35, # Evening rush (much busier now)
    np.where((hour_of_day >= 14) & (hour_of_day <= 17), 15, # Afternoon steady
    np.where(hour_of_day <= 6, -8, 0)))              # Deep night
)
weekend_waiting = is_weekend * 12

# ADDED SEASONAL VOLUME (Winter flu season = more people waiting)
seasonal_volume = np.where(season == "Winter", 15, np.where(season == "Summer", 5, 0))

patients_in_waiting_room = clamp(
    base_waiting + hour_factor + weekend_waiting + seasonal_volume + np.random.randint(-5, 6, N_ROWS),
    lo=0, hi=100
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. TARGET VARIABLE: wait_time_minutes
# ─────────────────────────────────────────────────────────────────────────────

# Step A: Minimum physical wait times.
min_physical_wait = np.where(triage_level == 1, np.random.uniform(0, 2, N_ROWS),
                    np.where(triage_level == 2, np.random.uniform(2, 5, N_ROWS),
                    np.where(triage_level == 3, np.random.uniform(5, 10, N_ROWS),
                    np.where(triage_level == 4, np.random.uniform(10, 15, N_ROWS),
                                                np.random.uniform(15, 20, N_ROWS)))))

# Step B: Queue Pressure (NOW INCLUDES NURSES)
# A nurse acts as a multiplier to doctor efficiency. We combine them into "effective staff"
effective_staff = doctors_on_duty + (nurses_on_duty * 0.3) 
queue_pressure = patients_in_waiting_room / effective_staff

# Step C: Triage Queue Multiplier
triage_multiplier = np.where(triage_level == 1, 0.0,
                    np.where(triage_level == 2, 2.0,
                    np.where(triage_level == 3, 6.0,
                    np.where(triage_level == 4, 12.0, 18.0))))

base_queue_wait = queue_pressure * triage_multiplier

# Step D: Percentage-based multipliers for Age and Complaints
age_multiplier = np.where(patient_age <= 5,  0.50,
                 np.where(patient_age <= 12, 0.75,
                 np.where(patient_age >= 75, 0.60,
                 np.where(patient_age >= 65, 0.80, 1.0))))

complaint_multiplier = np.where(
    np.isin(chief_complaint_category, ["Cardiac", "Trauma"]), 0.3,
    np.where(np.isin(chief_complaint_category, ["Respiratory", "Neurological"]), 0.7, 1.0)
)

# Step E: Operational Efficiency Penalty
night_sluggishness = np.where((hour_of_day >= 23) | (hour_of_day <= 6), 1.6, 1.0)
weekend_sluggishness = np.where(is_weekend == 1, 1.4, 1.0)

# SEASONAL EFFICIENCY (Winter means staff is dealing with contagious isolation rooms, slowing things down)
winter_sluggishness = np.where(season == "Winter", 1.25, 1.0)

efficiency_penalty = night_sluggishness * weekend_sluggishness * winter_sluggishness

# Combine elements
calculated_wait_time = min_physical_wait + (base_queue_wait * age_multiplier * complaint_multiplier * efficiency_penalty)

# Add proportional noise
noise = np.random.normal(0, 0.15 * calculated_wait_time, N_ROWS)
wait_time = calculated_wait_time + noise

# Cap at realistic max (8 hours)
wait_time = clamp(np.round(wait_time, 1), lo=1.0, hi=480.0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ASSEMBLE & SAVE
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
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
    "wait_time_minutes":        wait_time,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f" Dataset saved to: {OUTPUT_FILE}")