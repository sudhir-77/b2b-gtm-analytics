import json

import pandas as pd
from sklearn.metrics import roc_auc_score, silhouette_score

from src.pipeline import run
from src.scoring import mape
from tests.helpers import ROOT


def _echo(prompt: str) -> str:
    return prompt.split("\n\n", 1)[1]


def test_quick_pipeline_meets_analytical_criteria(tmp_path):
    metrics = run(
        quick=True,
        device="cpu",
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        config_path=ROOT / "config.yaml",
        measure_coverage=False,
        generate_fn=_echo,
    )
    by_name = {item["name"]: item for item in metrics["success_criteria"]}
    for name in (
        "Field-play effect estimate (DiD) vs. simulated truth",
        "Expansion propensity AUROC on held-out months",
        "3-month revenue forecast MAPE",
        "Segmentation silhouette score",
        "Numbers in the LLM summary that match computed tables",
    ):
        assert by_name[name]["passed"], name
    assert "Test coverage of src/" not in by_name
    assert metrics["hardware"]["peak_rss_gb"] < 3.0
    assert metrics["metrics"]["summary"]["model_loaded"] is False

    results = tmp_path / "results"
    predictions = pd.read_csv(results / "propensity_predictions.csv")
    test = predictions.loc[predictions["split"] == "test"]
    train = predictions.loc[predictions["split"] == "train"]
    assert set(test["month"]).isdisjoint(set(train["month"]))
    assert roc_auc_score(test["y_true"], test["p_lgbm"]) == metrics["metrics"]["propensity"]["lgbm_auroc"]

    monthly = pd.read_csv(results / "forecast_monthly.csv")
    assert mape(monthly["y_true"], monthly["y_pred_gbm"]) == metrics["metrics"]["forecast"]["gbm_mape"]
    accounts = pd.read_csv(results / "forecast_account_month.csv")
    summed = accounts.groupby("month")["y_true"].sum()
    direct = monthly.set_index("month")["y_true"]
    assert (summed - direct.loc[summed.index]).abs().max() < 1e-4

    scaled = pd.read_csv(results / "segmentation_scaled.csv")
    columns = [column for column in scaled.columns if column.startswith("z_")]
    score = silhouette_score(scaled[columns].to_numpy(), scaled["cluster"].to_numpy())
    assert abs(score - metrics["metrics"]["segmentation"]["silhouette"]) < 1e-4

    summary = (results / "exec_summary.md").read_text(encoding="utf-8")
    facts = json.loads((results / "computed_tables.json").read_text(encoding="utf-8"))
    from src.factcheck import fact_check

    assert fact_check(summary, facts)["passed"]
    assert (tmp_path / "data" / "splits.json").exists()
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_rows"] == metrics["metrics"]["simulation"]["duckdb_rows"]
