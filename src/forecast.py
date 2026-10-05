"""Three-month revenue forecast from a single pre-holdout origin.

Training targets never fall in the last three months. Test predictions use
only information known at month ``n_months - 4``. The seasonal-naive baseline
is that month's revenue from twelve months earlier. MAPE is computed on
company-wide monthly totals, which is the number a revenue plan would use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.scoring import mape
from src.splits import forecast_origin


def _slice_month(panel: pd.DataFrame, month: int, columns: list[str]) -> pd.DataFrame:
    out = panel.loc[panel["month"] == month, ["account_id", *columns]].copy()
    rename = {column: f"{column}_{month}" for column in columns}
    return out.rename(columns=rename)


def build_forecast_table(panel: pd.DataFrame, n_months: int) -> pd.DataFrame:
    origin_max = forecast_origin(n_months)
    test_months = [n_months - 3, n_months - 2, n_months - 1]
    rows: list[pd.DataFrame] = []
    horizons = (1, 2, 3)
    # Training origins need two lags and a seasonal lag of the target.
    for horizon in horizons:
        for origin in range(12, origin_max + 1):
            target = origin + horizon
            if target in test_months and origin != origin_max:
                continue
            if target > n_months - 1:
                continue
            if target in test_months:
                split = "test"
            elif target <= origin_max:
                split = "train"
            else:
                continue
            seasonal_month = target - 12
            if seasonal_month < 0 or seasonal_month > origin:
                continue
            piece = _assemble(panel, origin, target, horizon, seasonal_month, split)
            rows.append(piece)
    if not rows:
        raise ValueError("forecast table is empty")
    table = pd.concat(rows, ignore_index=True)
    _assert_no_leakage(table, test_months, origin_max)
    return table


def _assemble(
    panel: pd.DataFrame,
    origin: int,
    target: int,
    horizon: int,
    seasonal_month: int,
    split: str,
) -> pd.DataFrame:
    cur = panel.loc[
        panel["month"] == origin,
        ["account_id", "segment", "revenue", "seats", "usage", "log_seats"],
    ].rename(
        columns={
            "revenue": "revenue_origin",
            "seats": "seats_origin",
            "usage": "usage_origin",
            "log_seats": "log_seats_origin",
        }
    )
    lag1 = panel.loc[panel["month"] == origin - 1, ["account_id", "revenue"]].rename(
        columns={"revenue": "revenue_lag1"}
    )
    lag2 = panel.loc[panel["month"] == origin - 2, ["account_id", "revenue"]].rename(
        columns={"revenue": "revenue_lag2"}
    )
    seasonal = panel.loc[panel["month"] == seasonal_month, ["account_id", "revenue"]].rename(
        columns={"revenue": "revenue_seasonal"}
    )
    actual = panel.loc[panel["month"] == target, ["account_id", "revenue"]].rename(
        columns={"revenue": "y_true"}
    )
    merged = cur.merge(lag1, on="account_id").merge(lag2, on="account_id")
    merged = merged.merge(seasonal, on="account_id").merge(actual, on="account_id")
    merged["origin_month"] = origin
    merged["month"] = target
    merged["horizon"] = horizon
    merged["target_calendar_month"] = target % 12
    merged["split"] = split
    return merged


def _assert_no_leakage(table: pd.DataFrame, test_months: list[int], origin_max: int) -> None:
    test = table.loc[table["split"] == "test"]
    train = table.loc[table["split"] == "train"]
    if test.empty or train.empty:
        raise ValueError("forecast table needs both train and test rows")
    if int(test["origin_month"].max()) > origin_max:
        raise ValueError("test forecast uses an origin inside the holdout")
    if set(test["month"].astype(int)).difference(test_months):
        raise ValueError("test forecast targets left the holdout window")
    if set(train["month"].astype(int)).intersection(test_months):
        raise ValueError("forecast training targets include the holdout")


def fit_forecast(table: pd.DataFrame, n_estimators: int, seed: int) -> dict:
    train = table.loc[table["split"] == "train"].copy()
    test = table.loc[table["split"] == "test"].copy()
    feature_cols = [
        "horizon",
        "revenue_origin",
        "revenue_lag1",
        "revenue_lag2",
        "revenue_seasonal",
        "seats_origin",
        "usage_origin",
        "log_seats_origin",
        "target_calendar_month",
    ]
    dummies_train = pd.get_dummies(train["segment"], prefix="segment")
    dummies_test = pd.get_dummies(test["segment"], prefix="segment")
    columns = list(dummies_train.columns)
    for column in columns:
        if column not in dummies_test.columns:
            dummies_test[column] = 0
    x_train = pd.concat(
        [train[feature_cols].reset_index(drop=True), dummies_train.reset_index(drop=True)],
        axis=1,
    )
    x_test = pd.concat(
        [test[feature_cols].reset_index(drop=True), dummies_test[columns].reset_index(drop=True)],
        axis=1,
    )
    y_train = train["y_true"].to_numpy(dtype=float)
    model = LGBMRegressor(
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
    model.fit(x_train, y_train)
    pred = np.clip(model.predict(x_test), 0.0, None)
    account = test[
        ["account_id", "month", "y_true", "revenue_seasonal", "origin_month", "horizon"]
    ].copy()
    account["y_pred_gbm"] = pred
    account["y_pred_seasonal_naive"] = account["revenue_seasonal"].to_numpy()
    account = account.drop(columns=["revenue_seasonal"])

    monthly = (
        account.groupby("month", sort=True)
        .agg(
            y_true=("y_true", "sum"),
            y_pred_gbm=("y_pred_gbm", "sum"),
            y_pred_seasonal_naive=("y_pred_seasonal_naive", "sum"),
        )
        .reset_index()
    )
    gbm_mape = mape(monthly["y_true"].to_numpy(), monthly["y_pred_gbm"].to_numpy())
    naive_mape = mape(monthly["y_true"].to_numpy(), monthly["y_pred_seasonal_naive"].to_numpy())
    return {
        "gbm_mape": gbm_mape,
        "seasonal_naive_mape": naive_mape,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "account_predictions": account,
        "monthly": monthly,
    }
