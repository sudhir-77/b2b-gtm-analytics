"""Expansion propensity on a saved time split.

Features at month t are lagged one month, plus whether the field play starts
in month t (known when the play is launched). The test months are never used
to fit the booster, the logistic baseline, or the one-hot columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.scoring import binary_auroc

RAW_FEATURES = (
    "usage_lag",
    "touches_lag",
    "penetration_lag",
    "log_seats_lag",
    "treatment_onset",
)


def attach_splits(panel: pd.DataFrame, splits: dict[str, list[int]]) -> pd.DataFrame:
    mapping = {month: name for name, months in splits.items() for month in months}
    frame = panel.loc[panel["month"] > 0].copy()
    frame["split"] = frame["month"].map(lambda m: mapping.get(int(m), "exclude"))
    return frame


def _design(frame: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(frame["segment"], prefix="segment")
    base = frame.loc[:, list(RAW_FEATURES)].reset_index(drop=True)
    design = pd.concat([base, dummies.reset_index(drop=True)], axis=1)
    if columns is None:
        columns = list(design.columns)
    else:
        for column in columns:
            if column not in design.columns:
                design[column] = 0.0
        design = design.loc[:, columns]
    return design.astype(float), columns


def _assert_train_excludes_test(frame: pd.DataFrame) -> None:
    train_months = set(frame.loc[frame["split"] == "train", "month"].astype(int))
    test_months = set(frame.loc[frame["split"] == "test", "month"].astype(int))
    overlap = train_months.intersection(test_months)
    if overlap:
        raise ValueError(f"propensity train months overlap the test window: {sorted(overlap)}")
    if not test_months:
        raise ValueError("propensity frame has no test rows")


def fit_propensity(frame: pd.DataFrame, n_estimators: int, seed: int) -> dict:
    _assert_train_excludes_test(frame)
    train = frame.loc[frame["split"] == "train"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    x_train, columns = _design(train)
    y_train = train["expanded"].to_numpy(dtype=int)
    y_test = test["expanded"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    logistic = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=seed)
    logistic.fit(x_train_scaled, y_train)

    booster = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    booster.fit(x_train, y_train)

    # Validation rows are scored for the artifact only. They are not fit
    # and they are not the reported test metric.
    pieces = [_score_split(train, logistic, booster, scaler, columns)]
    val = frame.loc[frame["split"] == "val"]
    if not val.empty:
        pieces.append(_score_split(val, logistic, booster, scaler, columns))
    scored_test = _score_split(test, logistic, booster, scaler, columns)
    pieces.append(scored_test)
    predictions = pd.concat(pieces, ignore_index=True)
    p_lgbm = scored_test["p_lgbm"].to_numpy()
    p_logistic = scored_test["p_logistic"].to_numpy()
    lgbm_auc = binary_auroc(y_test, p_lgbm)
    logistic_auc = binary_auroc(y_test, p_logistic)
    return {
        "lgbm_auroc": lgbm_auc,
        "logistic_auroc": logistic_auc,
        "auroc_lift": float(lgbm_auc - logistic_auc),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_positive_rate": float(np.mean(y_test)),
        "predictions": predictions,
    }


def _score_split(frame, logistic, booster, scaler, columns: list[str]) -> pd.DataFrame:
    design, _ = _design(frame, columns)
    p_logistic = logistic.predict_proba(scaler.transform(design))[:, 1]
    p_lgbm = booster.predict_proba(design)[:, 1]
    return _prediction_frame(frame, p_lgbm, p_logistic)


def _prediction_frame(frame: pd.DataFrame, p_lgbm: np.ndarray, p_logistic: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": frame["account_id"].to_numpy(),
            "month": frame["month"].to_numpy(),
            "split": frame["split"].to_numpy(),
            "y_true": frame["expanded"].to_numpy(dtype=int),
            "p_lgbm": p_lgbm,
            "p_logistic": p_logistic,
        }
    )
