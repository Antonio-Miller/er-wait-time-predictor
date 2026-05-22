"""Validation script for the synthetic ER wait-time dataset.

Run from the assignment2 folder:
    python scripts/check_synthetic_data.py

The script regenerates the dataset and checks that the synthetic labels follow
intuitive healthcare patterns:
- weekends have longer waits than weekdays
- peak hours have longer waits than non-peak hours
- very young and elderly patients are prioritized and wait less than typical adults
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_SCRIPT = ROOT / "data" / "generate_data.py"
DATA_FILE = ROOT / "data" / "synthetic_er_data.csv"


def run_generator() -> None:
    subprocess.run([sys.executable, str(DATA_SCRIPT)], check=True, cwd=ROOT)


def main() -> None:
    run_generator()
    df = pd.read_csv(DATA_FILE)
    # --- General data quality checks ---------------------------------
    assert df.shape[0] >= 1000, f"Dataset too small: {df.shape[0]} rows (need >=1000)"
    assert not df.isnull().any().any(), "Dataset contains null values"
    # generator can produce long waits in stressed scenarios; accept up to 480 minutes
    assert df["wait_time_minutes"].between(1, 480).all(), "wait_time_minutes out of realistic range (1-480)"
    assert df["patient_age"].between(0, 120).all(), "patient_age out of realistic range (0-120)"
    assert df["hour_of_day"].between(0, 23).all(), "hour_of_day out of range (0-23)"

    # --- Weekend vs Weekday (overall + per-hour consistency + stats) ---
    weekend = df[df["is_weekend"] == 1]["wait_time_minutes"]
    weekday = df[df["is_weekend"] == 0]["wait_time_minutes"]
    weekend_mean = weekend.mean()
    weekday_mean = weekday.mean()
    # Non-parametric test in case distribution is skewed
    u_stat, u_p = stats.mannwhitneyu(weekend, weekday, alternative="greater")
    assert weekend_mean > weekday_mean, (
        f"Weekend mean <= weekday mean: {weekend_mean:.2f} <= {weekday_mean:.2f}"
    )
    assert u_p < 0.05, f"Weekend vs weekday difference not significant (Mann-Whitney p={u_p:.3f})"

    # Per-hour consistency: for hours with enough samples, weekend mean should be higher
    hours = []
    hour_diffs = []
    for h in range(24):
        w = df[(df["hour_of_day"] == h) & (df["is_weekend"] == 1)]["wait_time_minutes"]
        wd = df[(df["hour_of_day"] == h) & (df["is_weekend"] == 0)]["wait_time_minutes"]
        if len(w) >= 10 and len(wd) >= 10:
            hours.append(h)
            hour_diffs.append(w.mean() - wd.mean())
    # Require that the majority of comparable hours show higher weekend waits
    if len(hour_diffs) > 0:
        prop_pos = np.mean(np.array(hour_diffs) > 0)
        assert prop_pos >= 0.55, f"Weekend effect not consistent across hours (positive in {prop_pos*100:.1f}% of hours)"

    # --- Peak hours checks --------------------------------------------
    peak_mask = df["hour_of_day"].between(8, 10) | df["hour_of_day"].between(17, 19)
    peak_mean = df.loc[peak_mask, "wait_time_minutes"].mean()
    off_peak_mean = df.loc[~peak_mask, "wait_time_minutes"].mean()
    assert peak_mean > off_peak_mean, (
        f"Peak mean <= off-peak mean: {peak_mean:.2f} <= {off_peak_mean:.2f}"
    )
    # Peak gradient: check approach/increase before morning peak and decrease after
    hourly_means = df.groupby("hour_of_day")["wait_time_minutes"].mean()
    # Morning slope (hours 5->8): should be positive
    x_m = np.arange(5, 9)
    y_m = hourly_means.reindex(x_m).values
    if not np.isnan(y_m).any():
        slope_m, _, _, p_m, _ = stats.linregress(x_m, y_m)
        assert slope_m > 0, f"Morning slope approaching peak not positive (slope={slope_m:.3f}, p={p_m:.3f})"
    # Afternoon/evening slope (hours 16->20): should be positive approaching the full evening peak
    x_e = np.arange(16, 21)
    y_e = hourly_means.reindex(x_e).values
    if not np.isnan(y_e).any():
        slope_e, _, _, p_e, _ = stats.linregress(x_e, y_e)
        assert slope_e > 0, f"Evening slope approaching peak not positive (slope={slope_e:.3f}, p={p_e:.3f})"
    # Peak weekend vs peak weekday
    peak_weekend = df[peak_mask & (df["is_weekend"] == 1)]["wait_time_minutes"]
    peak_weekday = df[peak_mask & (df["is_weekend"] == 0)]["wait_time_minutes"]
    if len(peak_weekend) >= 20 and len(peak_weekday) >= 20:
        u_pk, p_pk = stats.mannwhitneyu(peak_weekend, peak_weekday, alternative="greater")
        assert peak_weekend.mean() > peak_weekday.mean(), "Peak weekend mean should be > peak weekday mean"
        assert p_pk < 0.05, f"Peak weekend vs weekday not significant (p={p_pk:.3f})"

    # --- Age-priority checks -----------------------------------------
    age_bins = pd.cut(
        df["patient_age"],
        bins=[-1, 4, 17, 59, 70, 120],
        labels=["0-4", "5-17", "18-59", "60-70", "71+"],
    )
    age_means = df.groupby(age_bins, observed=False)["wait_time_minutes"].mean()
    # Priority pattern: all non-adult groups should wait less than adults (18-59)
    assert age_means["0-4"] <= age_means["18-59"], "0-4 not <= 18-59"
    assert age_means["5-17"] <= age_means["18-59"], "5-17 not <= 18-59"
    assert age_means["60-70"] <= age_means["18-59"], "60-70 not <= 18-59"
    assert age_means["71+"] <= age_means["18-59"], "71+ not <= 18-59"

    # Age effect consistency across weekday/weekend
    for lbl in ["0-4", "5-17", "60-70", "71+"]:
        a_weekend = df[(age_bins == lbl) & (df["is_weekend"] == 1)]["wait_time_minutes"]
        a_weekday = df[(age_bins == lbl) & (df["is_weekend"] == 0)]["wait_time_minutes"]
        # if both sides have enough samples, ensure group mean is <= adult mean for that day type
        if len(a_weekend) >= 30 and len(a_weekday) >= 30:
            adult_weekend = df[(age_bins == "18-59") & (df["is_weekend"] == 1)]["wait_time_minutes"]
            adult_weekday = df[(age_bins == "18-59") & (df["is_weekend"] == 0)]["wait_time_minutes"]
            if len(adult_weekend) > 0 and len(a_weekend) > 0:
                assert a_weekend.mean() <= adult_weekend.mean() + 1e-6, (
                    f"Age group {lbl} on weekends has mean > adult mean on weekends"
                )
            if len(adult_weekday) > 0 and len(a_weekday) > 0:
                assert a_weekday.mean() <= adult_weekday.mean() + 1e-6, (
                    f"Age group {lbl} on weekdays has mean > adult mean on weekdays"
                )

    # --- Summary table output ---------------------------------------
    print("Synthetic data checks passed. Summary:")
    print(f"Rows: {df.shape[0]}  |  Weekend mean: {weekend_mean:.2f}  |  Weekday mean: {weekday_mean:.2f}")
    print(f"Peak mean: {peak_mean:.2f}  |  Off-peak mean: {off_peak_mean:.2f}")
    print("\nMean wait by age group:")
    print(age_means.round(2).to_string())
    print("\nHourly means (0-23):")
    print(hourly_means.round(1).to_string())


if __name__ == "__main__":
    main()
