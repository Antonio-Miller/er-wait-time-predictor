"""
train_models.py
===============
Trains and compares 3 regression models for ER Wait Time Prediction:
  1. Linear Regression   (baseline)
  2. Decision Tree Regressor
  3. Random Forest Regressor

Outputs:
  - Printed results table (MAE, RMSE, R²)
  - model/plots/model_comparison.png
  - model/plots/actual_vs_predicted.png
  - model/plots/feature_importance.png
  - model/plots/decision_tree.png
  - model/best_model.pkl
  - model/metadata.json

Usage:
    python train_models.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor, plot_tree
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import OneHotEncoder
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.age_features import (
    AGE_GROUPS,
    AGE_OFFSETS,
    AgeCalibratedRegressor,
    AgeFeatureEngineer,
    add_shared_features,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join("data", "synthetic_er_data.csv")
MODEL_DIR  = "model"
PLOTS_DIR  = os.path.join(MODEL_DIR, "plots")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns loaded.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DEFINE FEATURES & TARGET
# ─────────────────────────────────────────────────────────────────────────────

TARGET = "wait_time_minutes"

BASE_NUMERIC_FEATURES = [

    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",

    "patient_age",
    "triage_level",

    "patients_in_waiting_room",

    "doctors_on_duty",
    "nurses_on_duty",
]

CATEGORICAL_FEATURES = [
    "chief_complaint_category",
    "age_group",
]

RAW_FEATURES = BASE_NUMERIC_FEATURES + ["chief_complaint_category"]

X = df[RAW_FEATURES]
y = df[TARGET]

# ─────────────────────────────────────────────────────────────────────────────
# 2.5  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

print("Engineering features...")
X = add_shared_features(X)

# Final feature lists
NUMERIC_FEATURES = [feature for feature in BASE_NUMERIC_FEATURES if feature != "patient_age"] + [

    "staff_to_patient_ratio",
    "age_priority_score",

    "hour_sin",
    "hour_cos",

    "month_sin",
    "month_cos",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print(f"  Features: {len(ALL_FEATURES)} total "
      f"({len(NUMERIC_FEATURES)} numeric, "
      f"{len(CATEGORICAL_FEATURES)} categorical)\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN / TEST SPLIT  (80 / 20)
# ─────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  DEFINE THE 3 MODELS
# ─────────────────────────────────────────────────────────────────────────────

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree":     DecisionTreeRegressor(
                             max_depth=8,
                             min_samples_split=20,
                             min_samples_leaf=10,
                             random_state=42
                         ),

    "Random Forest":     RandomForestRegressor(
                             n_estimators=200,
                             max_depth=12,
                             min_samples_split=20,
                             min_samples_leaf=10,
                             random_state=42,
                             n_jobs=-1
                         ),
}

# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAIN, PREDICT & EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

results   = {}
pipelines = {}

baseline_prediction = y_train.mean()
naive_mae = mean_absolute_error(y_test, np.full_like(y_test, baseline_prediction))

print(f"Naive Baseline MAE: {naive_mae:.2f} minutes (predicting the mean)")
print("Training models with 5-fold cross-validation...\n")

for name, regressor in models.items():
    pipe = AgeCalibratedRegressor(Pipeline([
        ("features", Pipeline([
            ("age_features", AgeFeatureEngineer()),
            ("preprocessor", preprocessor),
        ])),
        ("model",        regressor),
    ])
    )

    cv_scores   = cross_val_score(pipe, X_train, y_train,
                                  cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_mae_mean = -cv_scores.mean()
    cv_mae_std  =  cv_scores.std()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        "MAE": mae, "RMSE": rmse, "R²": r2,
        "CV_MAE": cv_mae_mean, "CV_STD": cv_mae_std,
        "y_pred": y_pred
    }
    pipelines[name] = pipe
    print(f"  {name:<20} | CV MAE: {cv_mae_mean:6.2f} ±{cv_mae_std:4.2f} | Test MAE: {mae:6.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 95)
print(f"{'Model':<24} {'CV MAE (Mean ± STD)':<22} {'Test MAE':>10} {'Improvement %':>15}")
print("─" * 95)
print(f"{'Naive Baseline (Mean)':<24} {'N/A':<22} {naive_mae:>10.2f} {0:>14.1f}%")

for name, m in results.items():
    cv_str      = f"{m['CV_MAE']:>6.2f} ± {m['CV_STD']:>4.2f}"
    improvement = ((naive_mae - m["MAE"]) / naive_mae) * 100
    print(f"{name:<24} {cv_str:<22} {m['MAE']:>10.2f} {improvement:>14.1f}%")

print("─" * 95)

best_name = min(results, key=lambda n: results[n]["CV_MAE"])
print(f"\nBest model (by CV MAE): {best_name}")
print(f"  Reduces error by {((naive_mae - results[best_name]['MAE'])/naive_mae)*100:.1f}% "
      f"vs naive baseline.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 7.5  AGE SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

sanity_probe = pd.DataFrame([{
    "hour_of_day": 19,
    "day_of_week": 5,
    "is_weekend": 1,
    "month": 1,
    "patient_age": 2,
    "triage_level": 3,
    "chief_complaint_category": "Other",
    "patients_in_waiting_room": 18,
    "doctors_on_duty": 3,
    "nurses_on_duty": 6,
}, {
    "hour_of_day": 19,
    "day_of_week": 5,
    "is_weekend": 1,
    "month": 1,
    "patient_age": 35,
    "triage_level": 3,
    "chief_complaint_category": "Other",
    "patients_in_waiting_room": 18,
    "doctors_on_duty": 3,
    "nurses_on_duty": 6,
}, {
    "hour_of_day": 19,
    "day_of_week": 5,
    "is_weekend": 1,
    "month": 1,
    "patient_age": 80,
    "triage_level": 3,
    "chief_complaint_category": "Other",
    "patients_in_waiting_room": 18,
    "doctors_on_duty": 3,
    "nurses_on_duty": 6,
}])
sanity_probe = add_shared_features(sanity_probe)
sanity_preds = pipelines[best_name].predict(sanity_probe)
print("Age sanity check predictions (2yo / 35yo / 80yo):", np.round(sanity_preds, 1).tolist())
assert sanity_preds[1] > sanity_preds[0], "Expected 35-year-old to wait longer than 2-year-old in the probe case"
assert sanity_preds[1] > sanity_preds[2], "Expected 35-year-old to wait longer than 80-year-old in the probe case"
assert sanity_preds[1] == max(sanity_preds), "Expected 35-year-old to have the longest predicted wait in the probe case"

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOT 1 — Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

model_names = list(results.keys())
mae_vals  = [results[n]["MAE"]  for n in model_names]
rmse_vals = [results[n]["RMSE"] for n in model_names]
r2_vals   = [results[n]["R²"]   for n in model_names]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Comparison — ER Wait Time Prediction", fontsize=14, fontweight="bold")

for ax, vals, label, color in zip(
    axes,
    [mae_vals, rmse_vals, r2_vals],
    ["MAE (minutes)", "RMSE (minutes)", "R²"],
    COLORS
):
    bars = ax.bar(model_names, vals, color=color, alpha=0.85,
                  edgecolor="white", linewidth=1.2)
    ax.set_title(label, fontsize=12)
    ax.set_ylabel(label)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
path1 = os.path.join(PLOTS_DIR, "model_comparison.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path1}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLOT 2 — Actual vs Predicted
# ─────────────────────────────────────────────────────────────────────────────

y_pred_best = results[best_name]["y_pred"]
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_best, alpha=0.3, s=15, color=COLORS[2], label="Predictions")
lims = [0, max(y_test.max(), y_pred_best.max()) + 10]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Wait Time (min)", fontsize=12)
ax.set_ylabel("Predicted Wait Time (min)", fontsize=12)
ax.set_title(f"Actual vs Predicted — {best_name}", fontsize=13, fontweight="bold")
ax.legend()
ax.set_xlim(lims); ax.set_ylim(lims)
ax.text(0.05, 0.92,
        f"MAE  = {results[best_name]['MAE']:.2f} min\n"
        f"RMSE = {results[best_name]['RMSE']:.2f} min\n"
        f"R²   = {results[best_name]['R²']:.4f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
plt.tight_layout()
path2 = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path2}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOT 3 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

ohe_feature_names = (
    pipelines[best_name]
    .base_estimator_
    .named_steps["features"]
    .named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(CATEGORICAL_FEATURES)
    .tolist()
)
all_feature_names = NUMERIC_FEATURES + ohe_feature_names
best_estimator    = pipelines[best_name].base_estimator_.named_steps["model"]

if hasattr(best_estimator, "feature_importances_"):
    importances = best_estimator.feature_importances_
    feat_imp    = (pd.Series(importances, index=all_feature_names)
                   .sort_values(ascending=True).tail(15))
    fig, ax = plt.subplots(figsize=(8, 6))
    feat_imp.plot(kind="barh", ax=ax, color=COLORS[0], alpha=0.85, edgecolor="white")
    ax.set_title(f"Feature Importance — {best_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path3 = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path3}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. PLOT 4 — Decision Tree Visualisation (depth=3 for slides)
# ─────────────────────────────────────────────────────────────────────────────

print("\nGenerating Decision Tree visualisation (depth=3 for readability)...")
shallow_tree_pipe = Pipeline([
    ("features", Pipeline([
        ("age_features", AgeFeatureEngineer()),
        ("preprocessor", preprocessor),
    ])),
    ("model", DecisionTreeRegressor(max_depth=3, random_state=42)),
])
shallow_tree_pipe.fit(X_train, y_train)

ohe_names_shallow = (
    shallow_tree_pipe.named_steps["features"]
    .named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(CATEGORICAL_FEATURES).tolist()
)
feature_names_shallow = NUMERIC_FEATURES + ohe_names_shallow

fig, ax = plt.subplots(figsize=(24, 8))
plot_tree(shallow_tree_pipe.named_steps["model"],
          feature_names=feature_names_shallow,
          filled=True, rounded=True, fontsize=9, ax=ax,
          impurity=False, precision=1)
ax.set_title(
    "Decision Tree (depth=3) — ER Wait Time Prediction\n"
    "(Shallow version for illustration; full model uses depth=8)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
path4 = os.path.join(PLOTS_DIR, "decision_tree.png")
plt.savefig(path4, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {path4}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE BEST MODEL + METADATA
# ─────────────────────────────────────────────────────────────────────────────

joblib.dump(pipelines[best_name], MODEL_PATH)
print(f"\nBest model pipeline saved to: {MODEL_PATH}")

metadata = {
    "best_model_name":      best_name,
    "numeric_features":     NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "all_features":         ALL_FEATURES,
    "target":               TARGET,
    "baseline_mae":         round(naive_mae, 4),
    "results": {
        name: {
            "MAE":    round(m["MAE"], 4),
            "RMSE":   round(m["RMSE"], 4),
            "R2":     round(m["R²"], 4),
            "CV_MAE": round(m["CV_MAE"], 4),
            "CV_STD": round(m["CV_STD"], 4),
            "improvement_pct": round(((naive_mae - m["MAE"]) / naive_mae) * 100, 2)
        }
        for name, m in results.items()
    }
}
meta_path = os.path.join(MODEL_DIR, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {meta_path}")
print("\nAll done!\n")