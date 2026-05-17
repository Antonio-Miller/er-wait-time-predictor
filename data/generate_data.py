"""
generate_data.py
================
Synthetic ER Wait Time data generator — v5.

Design goals
------------
- Weekends and peak hours (08–10 and 17–19) should increase wait times.
- Age should have a realistic U-shaped effect:
    very young and elderly patients wait slightly longer on average.
- Directional relationships should be stable enough that a model learns
    intuitive weights for day/time/age features.
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

N_ROWS      = 10_000
OUTPUT_DIR  = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_er_data.csv")


def clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

hour_of_day = rng.integers(0, 24, N_ROWS)
day_of_week = rng.integers(0, 7,  N_ROWS)
month       = rng.integers(1, 13, N_ROWS)
is_weekend  = (day_of_week >= 5).astype(int)

hour_sin  = np.sin(2 * np.pi * hour_of_day / 24)
hour_cos  = np.cos(2 * np.pi * hour_of_day / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PATIENT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

# Realistic ER age mix: mostly adults, fewer very young children and elderly.
age_group = rng.choice(
    ["young_child", "child_teen", "adult", "older_adult", "elderly"],
    size=N_ROWS,
    p=[0.05, 0.13, 0.50, 0.22, 0.10],
)

patient_age = np.empty(N_ROWS, dtype=int)

mask = age_group == "young_child"
patient_age[mask] = rng.integers(0, 6, np.sum(mask))

mask = age_group == "child_teen"
patient_age[mask] = rng.integers(6, 18, np.sum(mask))

mask = age_group == "adult"
patient_age[mask] = np.round(
    clamp(rng.normal(37, 12, np.sum(mask)), 18, 64)
).astype(int)

mask = age_group == "older_adult"
patient_age[mask] = np.round(
    clamp(rng.normal(58, 7, np.sum(mask)), 45, 74)
).astype(int)

mask = age_group == "elderly"
patient_age[mask] = np.round(
    clamp(rng.normal(80, 7, np.sum(mask)), 70, 100)
).astype(int)

triage_level = rng.choice(
    [1, 2, 3, 4, 5],
    size=N_ROWS,
    p=[0.05, 0.10, 0.25, 0.35, 0.25],
)

complaint_categories = [
    "Trauma", "Cardiac", "Respiratory", "Abdominal",
    "Neurological", "Musculoskeletal", "Infection", "Other",
]
complaint_weights = [0.12, 0.10, 0.12, 0.15, 0.08, 0.18, 0.15, 0.10]
chief_complaint_category = rng.choice(
    complaint_categories, size=N_ROWS, p=complaint_weights
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STAFFING  — fixed before demand, never auto-corrects
# ─────────────────────────────────────────────────────────────────────────────

is_day = ((hour_of_day >= 8) & (hour_of_day <= 20)).astype(int)

doc_weekday = np.where(is_day, rng.integers(5, 8, N_ROWS),
                               rng.integers(2, 5, N_ROWS))
doc_weekend = np.where(is_day, rng.integers(3, 6, N_ROWS),
                               rng.integers(1, 4, N_ROWS))
doctors_on_duty = clamp(
    np.where(is_weekend, doc_weekend, doc_weekday), 1, 8
)

nurse_base     = doctors_on_duty * rng.uniform(2.0, 2.8, N_ROWS) \
                 + rng.normal(0, 0.8, N_ROWS)
nurses_on_duty = clamp(np.round(nurse_base).astype(int), 2, 16)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ARRIVAL RATE  λ  (patients / hour)
#
#     Peak order (intentional):
#       17–20h  ×3.0  >  07–10h  ×2.2  >  22–00h  ×1.25 >  00–02h  ×1.0
#     Deep quiet 03–06h  ×0.20
# ─────────────────────────────────────────────────────────────────────────────

LAMBDA_BASE = 7.0

hour_mult = np.select(
    [
        (hour_of_day >= 3)  & (hour_of_day <= 6),    # deep quiet
        (hour_of_day >= 7)  & (hour_of_day <= 10),   # morning surge
        (hour_of_day >= 11) & (hour_of_day <= 13),   # midday dip
        (hour_of_day >= 14) & (hour_of_day <= 16),   # afternoon build
        (hour_of_day >= 17) & (hour_of_day <= 20),   # evening PEAK
        (hour_of_day >= 21) & (hour_of_day <= 23),   # late evening
        (hour_of_day >= 0)  & (hour_of_day <= 2),    # midnight spike (capped)
    ],
    [0.20, 2.20, 1.10, 1.50, 3.00, 1.25, 1.00],
    default=1.00,
)

weekend_lam_mult = np.where(is_weekend == 1, 1.45, 1.00)

season_lam_mult = np.where(
    np.isin(month, [12, 1, 2]), 1.55,
    np.where(np.isin(month, [6, 7, 8]), 1.15, 1.00)
)

lam = (
    LAMBDA_BASE
    * hour_mult
    * weekend_lam_mult
    * season_lam_mult
    * rng.uniform(0.88, 1.12, N_ROWS)
)
lam = clamp(lam, 0.5, 45.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SERVICE RATE  μ  (patients / hour per doctor)
# ─────────────────────────────────────────────────────────────────────────────

MU_PER_DOC = 2.4

night_factor  = np.where((hour_of_day >= 23) | (hour_of_day <= 6), 0.82, 1.00)
winter_factor = np.where(np.isin(month, [12, 1, 2]), 0.88, 1.00)
mu_per_doc    = MU_PER_DOC * night_factor * winter_factor


# ─────────────────────────────────────────────────────────────────────────────
# 6.  M/M/c QUEUE WAIT
# ─────────────────────────────────────────────────────────────────────────────

c   = doctors_on_duty.astype(float)
mu  = mu_per_doc
rho = lam / (c * mu)

log_c_factorial = np.array([
    float(np.sum(np.log(np.arange(1, int(ci) + 1)))) if ci > 1 else 0.0
    for ci in c
])

log_erlang_num = c * np.log(np.maximum(c * rho, 1e-9)) - log_c_factorial

erlang_c = np.where(
    rho >= 1.0,
    1.0,
    np.exp(log_erlang_num) / (np.exp(log_erlang_num) + (1.0 - rho)),
)
erlang_c = clamp(erlang_c, 0.0, 1.0)

W_q_hours = np.where(
    rho < 1.0,
    erlang_c / (c * mu * (1.0 - rho)),
    (rho - 0.95) * 2.2 / mu,
)

mm_c_wait_minutes = clamp(W_q_hours * 60.0, 0.0, 360.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  WAITING ROOM SIZE  (Little's Law)
# ─────────────────────────────────────────────────────────────────────────────

W_total_hours = W_q_hours + 1.0 / np.maximum(mu, 0.1)
L_system      = lam * W_total_hours
patients_in_waiting_room = clamp(
    np.round(L_system * 0.60 + rng.normal(0, 3, N_ROWS)).astype(int),
    0, 80,
)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TRIAGE BASE WAIT
#
#     Triage 1: 1–5 min  — physical first-response time is never 0.
#               A resus team still needs 1–3 min to be at the bedside,
#               plus 1–2 min initial assessment. Realistic, not dangerous.
#     Triage 2: 5–12 min
#     Triage 3: 15–25 min
#     Triage 4: 25–45 min
#     Triage 5: 45–70 min
# ─────────────────────────────────────────────────────────────────────────────

triage_base = np.where(
    triage_level == 1, rng.uniform(1,  5,  N_ROWS),
    np.where(
        triage_level == 2, rng.uniform(5, 12, N_ROWS),
        np.where(
            triage_level == 3, rng.uniform(15, 25, N_ROWS),
            np.where(
                triage_level == 4, rng.uniform(25, 45, N_ROWS),
                rng.uniform(45, 70, N_ROWS),
            ),
        ),
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PRIORITY FRACTION  (queue position, triage 3–5 only)
# ─────────────────────────────────────────────────────────────────────────────

priority_fraction = np.where(
    triage_level == 1, 0.00,
    np.where(
        triage_level == 2, 0.05,
        np.where(
            triage_level == 3, 0.38,
            np.where(
                triage_level == 4, 0.72,
                1.00,
            ),
        ),
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# 10. COMPLAINT ADJUSTMENT TO PRIORITY FRACTION (triage 3+ only)
# ─────────────────────────────────────────────────────────────────────────────

complaint_pf_adj = np.where(
    np.isin(chief_complaint_category, ["Cardiac", "Trauma"]),            -0.28,
    np.where(
        np.isin(chief_complaint_category, ["Neurological", "Respiratory"]), -0.14,
        np.where(
            np.isin(chief_complaint_category, ["Abdominal", "Infection"]),   0.05,
            np.where(
                np.isin(chief_complaint_category, ["Musculoskeletal", "Other"]), 0.12,
                0.00,
            ),
        ),
    ),
)
complaint_pf_adj = np.where(triage_level <= 2, 0.0, complaint_pf_adj)


# ─────────────────────────────────────────────────────────────────────────────
# 11. AGE PRIORITY EFFECT
#
#     Priority relationship for triage 3+:
#       - minors and elderly are prioritized and get modest wait reductions.
#       - typical adults (18-59) are baseline queue population and wait longer.
# ─────────────────────────────────────────────────────────────────────────────

effective_priority = clamp(priority_fraction + complaint_pf_adj, 0.0, 1.0)

age_wait_adjustment = np.where(
    patient_age < 5,  -rng.uniform(8, 14, N_ROWS),
    np.where(
        patient_age <= 17, -rng.uniform(4, 8, N_ROWS),
        np.where(
            patient_age > 70, -rng.uniform(6, 11, N_ROWS),
            np.where(
                patient_age >= 60, -rng.uniform(3, 6, N_ROWS),
                rng.uniform(5, 10, N_ROWS),
            ),
        ),
    ),
)
age_wait_adjustment = np.where(triage_level <= 2, 0.0, age_wait_adjustment)


# ─────────────────────────────────────────────────────────────────────────────
# 12. OPERATIONAL OVERHEAD  (triage 3–5 only)
# ─────────────────────────────────────────────────────────────────────────────

night_overhead   = np.where((hour_of_day >= 23) | (hour_of_day <= 6), 7.0, 0.0)
weekend_overhead = np.where(is_weekend == 1, 16.0, 0.0)
winter_overhead  = np.where(np.isin(month, [12, 1, 2]),  8.0, 0.0)
peak_overhead    = np.where(
    ((hour_of_day >= 8) & (hour_of_day <= 10))
    | ((hour_of_day >= 17) & (hour_of_day <= 19)),
    12.0,
    0.0,
)

operational_overhead = (
    (night_overhead + weekend_overhead + winter_overhead + peak_overhead)
    * np.where(triage_level <= 2, 0.0, 1.0)
)


# ─────────────────────────────────────────────────────────────────────────────
# 13. FINAL WAIT TIME
#
#     Triage 1–2:  triage_base + tiny noise
#     Triage 3–5:  triage_base
#                + mm_c_queue_wait × effective_priority
#                + age_wait_adjustment
#                + operational_overhead
#                + noise
# ─────────────────────────────────────────────────────────────────────────────

noise_scale = np.where(
    triage_level <= 2,
    1.2,
    np.maximum(4.0, triage_base * 0.12),
)
noise = rng.normal(0, noise_scale, N_ROWS)

wait_time = clamp(
    np.round(
        triage_base
        + np.where(
            triage_level <= 2,
            0.0,
            mm_c_wait_minutes * effective_priority
            + age_wait_adjustment
            + operational_overhead,
        )
        + noise,
        1,
    ),
    1.0, 480.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# 14. DATAFRAME & SAVE
# ─────────────────────────────────────────────────────────────────────────────

df = pd.DataFrame({
    "hour_of_day":              hour_of_day,
    "day_of_week":              day_of_week,
    "is_weekend":               is_weekend,
    "month":                    month,
    "hour_sin":                 hour_sin,
    "hour_cos":                 hour_cos,
    "month_sin":                month_sin,
    "month_cos":                month_cos,
    "patient_age":              patient_age,
    "triage_level":             triage_level,
    "chief_complaint_category": chief_complaint_category,
    "patients_in_waiting_room": patients_in_waiting_room,
    "doctors_on_duty":          doctors_on_duty,
    "nurses_on_duty":           nurses_on_duty,
    "arrival_rate_per_hour":    lam,
    "wait_time_minutes":        wait_time,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

# ── Validation ────────────────────────────────────────────────────────────────
print(f"Dataset saved: {OUTPUT_FILE}")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols\n")

print("─── Overall wait time ───────────────────────────────")
print(df["wait_time_minutes"].describe().round(1))

print("\n─── Mean wait by triage level ───────────────────────")
print(df.groupby("triage_level")["wait_time_minutes"].mean().round(1))

print("\n─── Mean wait by hour ───────────────────────────────")
hourly = df.groupby("hour_of_day")["wait_time_minutes"].mean().round(1)
print(hourly.to_string())
peak1 = hourly[7:11].mean()
peak2 = hourly[17:21].mean()
quiet = hourly[3:7].mean()
print(f"\n  07–10h avg: {peak1:.1f} min  |  17–20h avg: {peak2:.1f} min  |  03–06h avg: {quiet:.1f} min")

print("\n─── Weekday vs Weekend ──────────────────────────────")
print(df.groupby("is_weekend")["wait_time_minutes"].mean().round(1))

print("\n─── Mean wait by season ─────────────────────────────")
season_label = {m: s for m, s in zip(
    range(1, 13),
    ["Win","Win","Spr","Spr","Spr","Sum","Sum","Sum","Aut","Aut","Aut","Win"]
)}
df["season"] = df["month"].map(season_label)
print(df.groupby("season")["wait_time_minutes"].mean().round(1))

print("\n─── Mean wait by chief complaint ────────────────────")
print(df.groupby("chief_complaint_category")["wait_time_minutes"]
      .mean().round(1).sort_values())

print("\n─── Age effect check (triage 3–5 only) ──────────────")
t35 = df[df["triage_level"] >= 3]
age_groups = {
    "Under 5":  t35[t35["patient_age"] < 5],
    "5–17":     t35[(t35["patient_age"] >= 5)  & (t35["patient_age"] <= 17)],
    "18–59":    t35[(t35["patient_age"] >= 18) & (t35["patient_age"] < 60)],
    "60–70":    t35[(t35["patient_age"] >= 60) & (t35["patient_age"] <= 70)],
    "71+":      t35[t35["patient_age"] > 70],
}
for label, grp in age_groups.items():
    print(f"  {label:8s}  mean={grp['wait_time_minutes'].mean():.1f} min  n={len(grp)}")

print("\n─── Triage 1 stats ──────────────────────────────────")
t1 = df[df["triage_level"] == 1]["wait_time_minutes"]
print(f"  min={t1.min():.1f}  max={t1.max():.1f}  mean={t1.mean():.1f}")

print("\n─── Triage 1 weekend evening peak (stress test) ─────")
stress = df[(df["triage_level"] == 1) & (df["is_weekend"] == 1)
            & (df["hour_of_day"].between(17, 20))]["wait_time_minutes"]
print(f"  n={len(stress)}  min={stress.min():.1f}  max={stress.max():.1f}  mean={stress.mean():.1f}")