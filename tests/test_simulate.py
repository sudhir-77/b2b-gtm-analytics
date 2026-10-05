import numpy as np
import pandas as pd

from src.simulate import expansion_probability, simulate, write_duckdb
from tests.helpers import make_config


def test_expansion_probability_xor_regions():
    high_usage_low_touch = expansion_probability(np.array([0.8]), np.array([0]), np.array([0]))
    low_usage_high_touch = expansion_probability(np.array([0.2]), np.array([6]), np.array([0]))
    both = expansion_probability(np.array([0.8]), np.array([6]), np.array([0]))
    neither = expansion_probability(np.array([0.2]), np.array([0]), np.array([0]))
    onset = expansion_probability(np.array([0.2]), np.array([0]), np.array([1]))
    assert high_usage_low_touch[0] > 0.85
    assert low_usage_high_touch[0] > 0.85
    assert both[0] < 0.15
    assert neither[0] < 0.15
    assert onset[0] > neither[0]


def test_simulator_invariants_and_reproducibility(tmp_path):
    cfg = make_config(n_accounts=100, usage_events_per_account_month=2, noise_sigma=0.0, revenue_noise=0.0)
    first = simulate(cfg)
    second = simulate(cfg)
    pd.testing.assert_series_equal(
        first.account_month["log_seats"],
        second.account_month["log_seats"],
        check_names=False,
    )
    panel = first.account_month
    assert len(panel) == cfg.n_accounts * cfg.n_months
    assert panel["seats"].min() > 0
    assert panel["revenue"].min() > 0
    assert set(panel["segment"].unique()) == {seg.name for seg in cfg.segments}
    cohorts = sorted(c for c in panel["cohort"].unique() if c >= 0)
    assert len(cohorts) > 1

    treated = panel["cohort"] >= 0
    assert (panel.loc[~treated, "treated"] == 0).all()
    pre = panel.loc[treated & (panel["month"] < panel["cohort"])]
    post = panel.loc[treated & (panel["month"] >= panel["cohort"])]
    assert (pre["treated"] == 0).all()
    assert (post["treated"] == 1).all()
    onset = panel.loc[panel["treatment_onset"] == 1]
    assert (onset["month"] == onset["cohort"]).all()

    merged = panel.merge(first.accounts[["account_id", "log_seats_base"]], on="account_id")
    season = cfg.season_amp * np.sin(2.0 * np.pi * merged["month"] / 12.0)
    expected = (
        merged["log_seats_base"]
        + cfg.trend_per_month * merged["month"]
        + season
        + cfg.field_play_effect * merged["treated"]
    )
    assert np.allclose(merged["log_seats"], expected)

    counts = write_duckdb(first, tmp_path / "gtm.duckdb")
    assert counts == first.expected_counts
    assert sum(counts.values()) == first.total_rows
