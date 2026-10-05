from sklearn.metrics import silhouette_score

from src.segment import account_features, fit_segments
from src.simulate import simulate
from src.splits import forecast_origin
from tests.helpers import make_config


def test_pre_holdout_segments_are_separated():
    cfg = make_config(n_accounts=240, usage_events_per_account_month=1)
    panel = simulate(cfg).account_month
    origin = forecast_origin(cfg.n_months)
    features = account_features(panel, max_month_inclusive=origin)
    assert features["account_id"].is_unique
    result = fit_segments(features, k=4, seed=cfg.seed)
    assert result["silhouette"] >= 0.25
    scaled = result["scaled"]
    columns = [column for column in scaled.columns if column.startswith("z_")]
    recomputed = silhouette_score(scaled[columns].to_numpy(), scaled["cluster"].to_numpy())
    assert abs(recomputed - result["silhouette"]) < 1e-8
    # Holdout months do not enter the feature averages.
    leaked = account_features(panel, max_month_inclusive=cfg.n_months - 1)
    early = features.set_index("account_id")["mean_log_seats"]
    late = leaked.set_index("account_id")["mean_log_seats"]
    assert not early.equals(late)
