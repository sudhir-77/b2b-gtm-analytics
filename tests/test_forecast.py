import numpy as np

from src.forecast import build_forecast_table, fit_forecast
from src.scoring import mape
from src.simulate import simulate
from tests.helpers import make_config


def test_forecast_holdout_does_not_leak_into_training():
    cfg = make_config(n_accounts=80, usage_events_per_account_month=1, noise_sigma=0.01)
    panel = simulate(cfg).account_month
    table = build_forecast_table(panel, cfg.n_months)
    test_months = {cfg.n_months - 3, cfg.n_months - 2, cfg.n_months - 1}
    train_targets = set(table.loc[table["split"] == "train", "month"].astype(int))
    assert train_targets.isdisjoint(test_months)
    test = table.loc[table["split"] == "test"]
    assert set(test["month"].astype(int)) == test_months
    assert int(test["origin_month"].max()) == cfg.n_months - 4
    result = fit_forecast(table, n_estimators=30, seed=cfg.seed)
    recomputed = mape(
        result["monthly"]["y_true"].to_numpy(),
        result["monthly"]["y_pred_gbm"].to_numpy(),
    )
    assert recomputed == result["gbm_mape"]
    monthly_from_accounts = result["account_predictions"].groupby("month")[["y_true", "y_pred_gbm"]].sum()
    assert np.allclose(monthly_from_accounts["y_true"], result["monthly"].set_index("month")["y_true"])
