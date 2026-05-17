"""Age-aware feature engineering and calibration helpers for ER wait-time models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

AGE_GROUPS = ["0-4", "5-17", "18-59", "60-70", "71+"]
AGE_OFFSETS = {
    "0-4": -8.0,
    "5-17": -5.0,
    "18-59": 15.0,
    "60-70": -4.0,
    "71+": -8.0,
}


def add_shared_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["staff_to_patient_ratio"] = (
        (frame["doctors_on_duty"] + frame["nurses_on_duty"])
        / (frame["patients_in_waiting_room"] + 1)
    )
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour_of_day"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour_of_day"] / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    return frame


class AgeFeatureEngineer(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy()
        frame["age_group"] = pd.cut(
            frame["patient_age"],
            bins=[-1, 4, 17, 59, 70, 120],
            labels=AGE_GROUPS,
        )
        age = frame["patient_age"].astype(float)
        age_score = np.abs(age - 38.0) / 38.0
        frame["age_priority_score"] = np.clip(age_score, 0.0, 1.0)
        return frame


class AgeCalibratedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, age_offsets=None):
        self.base_estimator = base_estimator
        self.age_offsets = age_offsets if age_offsets is not None else AGE_OFFSETS

    def fit(self, X, y):
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y)
        return self

    def predict(self, X):
        base_prediction = self.base_estimator_.predict(X)
        age_group = pd.cut(
            pd.Series(X["patient_age"]),
            bins=[-1, 4, 17, 59, 70, 120],
            labels=AGE_GROUPS,
        )
        age_offsets = age_group.map(self.age_offsets).fillna(0.0).astype(float).to_numpy()
        return base_prediction + age_offsets
