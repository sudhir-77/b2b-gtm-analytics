"""K-means segmentation on pre-holdout account features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = (
    "mean_log_seats",
    "mean_usage",
    "mean_penetration",
    "mean_touches",
)


def account_features(panel: pd.DataFrame, max_month_inclusive: int) -> pd.DataFrame:
    """Average behavior through the last month the holdout is allowed to see."""
    history = panel.loc[panel["month"] <= max_month_inclusive]
    grouped = history.groupby("account_id", sort=True)
    features = grouped.agg(
        mean_log_seats=("log_seats", "mean"),
        mean_usage=("usage", "mean"),
        mean_penetration=("penetration", "mean"),
        mean_touches=("touches", "mean"),
        segment=("segment", "first"),
    )
    return features.reset_index()


def fit_segments(features: pd.DataFrame, k: int = 4, seed: int = 42) -> dict:
    matrix = features.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = model.fit_predict(scaled)
    score = float(silhouette_score(scaled, labels))
    assigned = features.copy()
    assigned["cluster"] = labels.astype(int)
    scaled_frame = pd.DataFrame(scaled, columns=[f"z_{c}" for c in FEATURE_COLUMNS])
    scaled_frame.insert(0, "account_id", features["account_id"].to_numpy())
    scaled_frame["cluster"] = labels.astype(int)
    return {
        "silhouette": score,
        "k": int(k),
        "assignments": assigned,
        "scaled": scaled_frame,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
