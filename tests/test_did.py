import numpy as np
import pandas as pd

from src.did import evaluate_did, naive_before_after, twoway_demean
from src.simulate import simulate
from tests.helpers import make_config


def test_twoway_demean_removes_group_means():
    values = np.array([1.0, 3.0, 5.0, 7.0])
    entity = np.array([1, 1, 2, 2])
    time = np.array([0, 1, 0, 1])
    residual = twoway_demean(values, entity, time)
    assert abs(residual[entity == 1].mean()) < 1e-8
    assert abs(residual[time == 0].mean()) < 1e-8


def test_zero_noise_did_recovers_the_planted_effect():
    cfg = make_config(n_accounts=180, noise_sigma=0.0, revenue_noise=0.0, usage_events_per_account_month=1)
    panel = simulate(cfg).account_month
    result = evaluate_did(panel, cfg.field_play_effect)
    assert abs(result["estimate"] - cfg.field_play_effect) < 1e-6
    assert result["relative_error"] < 1e-6
    assert result["ci_covers_truth"] == 1.0 or result["se"] < 1e-6
    assert result["naive_relative_error"] > 0.2


def test_naive_before_after_includes_the_time_trend():
    # One treated account, no control adjustment. Trend of 0.01 per month
    # plus a 0.08 shift at month 2.
    rows = []
    for month, treated, y in ((0, 0, 0.00), (1, 0, 0.01), (2, 1, 0.10), (3, 1, 0.11)):
        rows.append(
            {
                "account_id": 1,
                "month": month,
                "cohort": 2,
                "treated": treated,
                "log_seats": y,
            }
        )
    estimate = naive_before_after(pd.DataFrame(rows))
    assert estimate == (0.10 + 0.11) / 2 - (0.00 + 0.01) / 2
