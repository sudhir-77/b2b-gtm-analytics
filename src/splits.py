"""Time splits, written once and reused for every fit.

Month 0 has no lag, so it is never a modeling row. The last three months are
the test window for both the propensity model and the revenue forecast. The
three months before that are a validation window used only for early checks
inside unit tests; the fitted models use fixed hyperparameters and do not
read the test months.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def time_splits(n_months: int) -> dict[str, list[int]]:
    if n_months < 18:
        raise ValueError("n_months must be at least 18")
    test = [n_months - 3, n_months - 2, n_months - 1]
    val = [n_months - 6, n_months - 5, n_months - 4]
    train = list(range(1, n_months - 6))
    _assert_disjoint(train, val, test)
    return {"train": train, "val": val, "test": test}


def forecast_origin(n_months: int) -> int:
    """Last month the forecast may use as a feature origin for the holdout."""
    return n_months - 4


def _assert_disjoint(*groups: list[int]) -> None:
    seen: set[int] = set()
    for group in groups:
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(f"split months overlap: {sorted(overlap)}")
        seen.update(group)


def assign_propensity_split(months: pd.Series, splits: dict[str, list[int]]) -> pd.Series:
    mapping = {}
    for name, values in splits.items():
        for month in values:
            mapping[int(month)] = name
    assigned = months.map(lambda m: mapping.get(int(m), "exclude"))
    return assigned


def save_splits(path: Path, n_months: int, seed: int) -> dict:
    splits = time_splits(n_months)
    payload = {
        "seed": int(seed),
        "n_months": int(n_months),
        "propensity": splits,
        "forecast": {
            "train_target_max_month": int(forecast_origin(n_months)),
            "test_months": splits["test"],
            "feature_origin_month": int(forecast_origin(n_months)),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_splits(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_frames_disjoint(train_ids: pd.Series, test_ids: pd.Series) -> None:
    overlap = set(train_ids).intersection(set(test_ids))
    if overlap:
        raise ValueError(f"train and test ids overlap, for example {next(iter(overlap))}")
