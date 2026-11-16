"""
model.py
Trains a GradientBoostingRegressor to predict overpopulation risk scores by
zip code from shelter intake/demographic features.

Exports
-------
FEATURE_COLS   list[str]  — ordered feature columns used for training/inference
train_model()  → (pipeline, metrics_dict)
predict()      → DataFrame with original columns + predicted_risk + risk_tier
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS: list[str] = [
    "stray_intake",
    "owner_surrender",
    "total_intake",
    "adoption_rate",
    "return_to_owner_rate",
    "median_household_income",
    "population_density",
    "prior_spay_neuter_rate",
    "vet_clinics_per_10k",
    "avg_litter_size",
    "repeat_intakes",
]

TARGET_COL = "risk_score"


def _risk_tier(score: float) -> str:
    if score >= 70:
        return "🔴 High"
    if score >= 45:
        return "🟡 Medium"
    return "🟢 Low"


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """
    Fit model on *df* and return (fitted_pipeline, metrics).

    metrics keys: mae_train, mae_test, r2_train, r2_test, cv_r2_mean, cv_r2_std,
                  feature_importances (dict col→importance)
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.85,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

    importances = dict(
        zip(FEATURE_COLS, pipeline.named_steps["gbr"].feature_importances_)
    )

    metrics = {
        "mae_train": round(mean_absolute_error(y_train, y_pred_train), 3),
        "mae_test": round(mean_absolute_error(y_test, y_pred_test), 3),
        "r2_train": round(r2_score(y_train, y_pred_train), 4),
        "r2_test": round(r2_score(y_test, y_pred_test), 4),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std": round(cv_scores.std(), 4),
        "feature_importances": importances,
    }

    return pipeline, metrics


def predict(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add *predicted_risk* and *risk_tier* columns to a copy of *df*.
    """
    result = df.copy()
    preds = pipeline.predict(df[FEATURE_COLS])
    result["predicted_risk"] = np.clip(preds, 0, 100).round(1)
    result["risk_tier"] = result["predicted_risk"].apply(_risk_tier)
    return result
