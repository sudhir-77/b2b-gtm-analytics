import numpy as np
import pandas as pd

from src.propensity import fit_propensity
from src.simulate import expansion_probability


def test_lightgbm_beats_logistic_on_the_xor_rule():
    rng = np.random.default_rng(0)
    n = 4000
    usage = rng.uniform(0.0, 1.0, size=n)
    touches = rng.integers(0, 9, size=n)
    onset = (rng.random(n) < 0.08).astype(float)
    probability = expansion_probability(usage, touches, onset)
    expanded = (rng.random(n) < probability).astype(int)
    segments = rng.choice(["enterprise", "mid_market", "smb", "digital"], size=n)
    split = np.where(np.arange(n) < 3000, "train", "test")
    frame = pd.DataFrame(
        {
            "account_id": np.arange(n),
            "month": np.where(split == "train", 5, 22),
            "segment": segments,
            "expanded": expanded,
            "usage_lag": usage,
            "touches_lag": touches,
            "penetration_lag": rng.uniform(0.2, 0.8, size=n),
            "log_seats_lag": rng.normal(4.5, 0.4, size=n),
            "treatment_onset": onset,
            "split": split,
        }
    )
    result = fit_propensity(frame, n_estimators=60, seed=0)
    assert result["lgbm_auroc"] >= 0.75
    assert result["lgbm_auroc"] >= result["logistic_auroc"] + 0.02
    test_months = set(result["predictions"].loc[result["predictions"]["split"] == "test", "month"])
    train_months = set(result["predictions"].loc[result["predictions"]["split"] == "train", "month"])
    assert test_months.isdisjoint(train_months)


def test_fit_rejects_a_leaky_split():
    frame = pd.DataFrame(
        {
            "account_id": [1, 2],
            "month": [5, 5],
            "segment": ["smb", "smb"],
            "expanded": [0, 1],
            "usage_lag": [0.2, 0.8],
            "touches_lag": [1, 6],
            "penetration_lag": [0.3, 0.4],
            "log_seats_lag": [3.0, 4.0],
            "treatment_onset": [0, 0],
            "split": ["train", "test"],
        }
    )
    frame.loc[frame["split"] == "test", "month"] = 5
    try:
        fit_propensity(frame, n_estimators=5, seed=0)
    except ValueError:
        return
    raise AssertionError("overlapping train and test months were accepted")
