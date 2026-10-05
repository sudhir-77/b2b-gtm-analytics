"""End-to-end run: simulate, measure, model, and fact-check a summary."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import Config, load_config
from src.did import evaluate_did
from src.download import download_model
from src.forecast import build_forecast_table, fit_forecast
from src.factcheck import fact_check
from src.hardware import peak_rss_gb, snapshot
from src.metrics_sql import run_metrics, summarize_metrics
from src.plots import plot_did, plot_forecast, plot_nrr, plot_roc, plot_segments
from src.propensity import attach_splits, fit_propensity
from src.reporting import render_resume_notes, success_criteria, update_readme, write_json
from src.segment import account_features, fit_segments
from src.simulate import simulate, write_duckdb
from src.splits import forecast_origin, save_splits
from src.summarize import generate_text, publish_summary

ROOT = Path(__file__).resolve().parents[1]


def build_facts(cfg: Config, total_rows: int, sql: dict, did: dict, propensity: dict, forecast: dict, silhouette: float) -> dict:
    return {
        "sample": {
            "n_accounts": int(cfg.n_accounts),
            "n_months": int(cfg.n_months),
            "duckdb_rows": int(total_rows),
        },
        "retention": {
            "nrr_12m": sql["nrr_12m"],
            "mean_monthly_nrr": sql["mean_monthly_nrr"],
            "mean_adoption": sql["mean_adoption"],
            "mean_adoption_pct": sql["mean_adoption"] * 100.0,
            "mean_penetration": sql["mean_penetration"],
            "mean_penetration_pct": sql["mean_penetration"] * 100.0,
            "median_penetration": sql["median_penetration"],
        },
        "field_play": {
            "truth": did["truth"],
            "truth_pct": did["truth"] * 100.0,
            "estimate": did["estimate"],
            "se": did["se"],
            "ci_low": did["ci_low"],
            "ci_high": did["ci_high"],
            "relative_error": did["relative_error"],
            "relative_error_pct": did["relative_error"] * 100.0,
            "naive_estimate": did["naive_estimate"],
            "naive_relative_error": did["naive_relative_error"],
            "naive_relative_error_pct": did["naive_relative_error"] * 100.0,
        },
        "propensity": {
            "lgbm_auroc": propensity["lgbm_auroc"],
            "logistic_auroc": propensity["logistic_auroc"],
            "auroc_lift": propensity["auroc_lift"],
        },
        "forecast": {
            "gbm_mape": forecast["gbm_mape"],
            "gbm_mape_pct": forecast["gbm_mape"] * 100.0,
            "seasonal_naive_mape": forecast["seasonal_naive_mape"],
            "seasonal_naive_mape_pct": forecast["seasonal_naive_mape"] * 100.0,
            "horizon_months": 3,
        },
        "segmentation": {
            "silhouette": silhouette,
            "k": 4,
        },
    }


def _coverage_percent(results_dir: Path) -> tuple[float, bool]:
    if os.environ.get("GTM_IN_PYTEST") == "1":
        return 0.0, False
    cov_path = results_dir / "coverage.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "not smoke",
        "--cov=src",
        f"--cov-report=json:{cov_path}",
        "-q",
        "--tb=line",
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if not cov_path.exists():
        return 0.0, False
    payload = json.loads(cov_path.read_text(encoding="utf-8"))
    percent = float(payload["totals"]["percent_covered"])
    return percent, completed.returncode == 0 and percent >= 70.0


def run(
    quick: bool = False,
    device: str = "cpu",
    data_dir: Path | None = None,
    results_dir: Path | None = None,
    model_dir: Path | None = None,
    config_path: Path | None = None,
    measure_coverage: bool = True,
    generate_fn=None,
) -> dict:
    started = time.perf_counter()
    if quick or os.environ.get("GTM_IN_PYTEST") == "1":
        measure_coverage = False
    data_dir = data_dir or (ROOT / "data")
    results_dir = results_dir or (ROOT / "results")
    config_path = config_path or (ROOT / "config.yaml")
    model_dir = model_dir or (data_dir / "models" / "SmolLM2-360M-Instruct")
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path, quick=quick)
    print(f"simulating {cfg.n_accounts} accounts x {cfg.n_months} months", flush=True)
    data = simulate(cfg)
    db_path = data_dir / "gtm.duckdb"
    counts = write_duckdb(data, db_path)
    total_rows = int(sum(counts.values()))
    write_json(data_dir / "manifest.json", {"counts": counts, "total_rows": total_rows})
    if total_rows != data.total_rows:
        raise RuntimeError("manifest row count does not match the simulator")

    panel = data.account_month
    del data
    gc.collect()

    tables = run_metrics(db_path)
    sql = summarize_metrics(tables)
    tables["monthly_nrr"].to_csv(results_dir / "monthly_nrr.csv", index=False)
    tables["adoption"].to_csv(results_dir / "adoption.csv", index=False)
    tables["nrr_12"].to_csv(results_dir / "nrr_12.csv", index=False)

    # The split file is written before any model sees a row.
    split_payload = save_splits(data_dir / "splits.json", cfg.n_months, cfg.seed)
    propensity_splits = split_payload["propensity"]

    origin = forecast_origin(cfg.n_months)
    features = account_features(panel, max_month_inclusive=origin)
    segmented = fit_segments(features, k=4, seed=cfg.seed)
    did = evaluate_did(panel, cfg.field_play_effect)
    print(
        f"did estimate {did['estimate']:.4f} vs truth {did['truth']:.4f}",
        flush=True,
    )

    model_frame = attach_splits(panel, propensity_splits)
    propensity = fit_propensity(model_frame, n_estimators=cfg.lgbm_n_estimators, seed=cfg.seed)
    print(
        f"auroc lgbm {propensity['lgbm_auroc']:.3f} logistic {propensity['logistic_auroc']:.3f}",
        flush=True,
    )

    forecast_table = build_forecast_table(panel, cfg.n_months)
    forecast = fit_forecast(forecast_table, n_estimators=cfg.forecast_n_estimators, seed=cfg.seed)
    print(
        f"mape gbm {forecast['gbm_mape']:.3%} naive {forecast['seasonal_naive_mape']:.3%}",
        flush=True,
    )

    plot_nrr(tables["monthly_nrr"], results_dir / "nrr_monthly.png")
    plot_segments(segmented["assignments"], results_dir / "segments.png")
    plot_roc(propensity["predictions"], results_dir / "propensity_roc.png")
    plot_forecast(forecast["monthly"], results_dir / "forecast.png")
    plot_did(panel, results_dir / "did_trends.png")

    panel.loc[:, ["account_id", "month", "log_seats", "treated", "cohort", "segment"]].to_csv(
        results_dir / "did_panel.csv", index=False
    )
    propensity["predictions"].to_csv(results_dir / "propensity_predictions.csv", index=False)
    forecast["account_predictions"].to_csv(results_dir / "forecast_account_month.csv", index=False)
    forecast["monthly"].to_csv(results_dir / "forecast_monthly.csv", index=False)
    segmented["assignments"].to_csv(results_dir / "segmentation.csv", index=False)
    segmented["scaled"].to_csv(results_dir / "segmentation_scaled.csv", index=False)

    facts = build_facts(cfg, total_rows, sql, did, propensity, forecast, segmented["silhouette"])
    write_json(results_dir / "computed_tables.json", facts)

    del panel, forecast_table, model_frame, tables
    gc.collect()

    model_loaded = False
    if generate_fn is None:
        print("downloading or verifying local model weights", flush=True)
        download_model(model_dir, cfg.model_repo, cfg.model_sha256)
        model_loaded = True

        def generate_fn(prompt: str) -> str:
            return generate_text(
                prompt,
                model_dir=model_dir,
                device=device,
                max_new_tokens=cfg.llm_max_new_tokens,
                seed=cfg.seed,
            )

    print("drafting executive summary", flush=True)
    published = publish_summary(facts, generate_fn, attempts=2)
    summary_path = results_dir / "exec_summary.md"
    summary_path.write_text(published["text"].rstrip() + "\n", encoding="utf-8")
    # Re-check the file that will be read by an evaluator, not only the in-memory string.
    on_disk = fact_check(summary_path.read_text(encoding="utf-8"), facts)
    if not on_disk["passed"]:
        raise RuntimeError("exec_summary.md failed the fact check after writing")

    write_json(
        results_dir / "summary_drafts.json",
        {
            "source": published["source"],
            "drafts_rejected": published["drafts_rejected"],
            "model_loaded": model_loaded,
            "drafts": published["drafts"],
            "canonical": published["canonical"],
        },
    )

    propensity_counts = model_split_counts(propensity["predictions"])

    criteria = success_criteria(
        did,
        propensity,
        forecast,
        segmented["silhouette"],
        on_disk["match_rate"],
        coverage_pct=None,
    )
    hardware = snapshot(device)
    metrics = _assemble_metrics(
        cfg,
        criteria,
        facts,
        did,
        propensity,
        forecast,
        segmented["silhouette"],
        on_disk,
        published,
        model_loaded,
        propensity_counts,
        forecast,
        hardware,
        runtime=0.0,
    )
    notes = render_resume_notes(facts)
    checked_notes = fact_check(notes, facts)
    if not checked_notes["passed"]:
        raise RuntimeError(f"resume notes failed the fact check: {checked_notes['unmatched']}")
    (results_dir / "RESUME_NOTES.md").write_text(notes, encoding="utf-8")

    coverage_pct = None
    tests_ok = None
    if measure_coverage:
        print("measuring test coverage", flush=True)
        coverage_pct, tests_ok = _coverage_percent(results_dir)
        criteria = success_criteria(
            did,
            propensity,
            forecast,
            segmented["silhouette"],
            on_disk["match_rate"],
            coverage_pct=coverage_pct if tests_ok or coverage_pct else coverage_pct,
        )
        # A failing test run does not get to claim the coverage bar.
        if tests_ok is False:
            for item in criteria:
                if item["name"] == "Test coverage of src/":
                    item["passed"] = False
    metrics = _assemble_metrics(
        cfg,
        criteria,
        facts,
        did,
        propensity,
        forecast,
        segmented["silhouette"],
        on_disk,
        published,
        model_loaded,
        propensity_counts,
        forecast,
        hardware,
        runtime=time.perf_counter() - started,
    )
    metrics["hardware"]["peak_rss_gb"] = round(peak_rss_gb(), 4)
    if tests_ok is not None:
        metrics["metrics"]["coverage"] = {
            "percent": coverage_pct,
            "tests_passed": bool(tests_ok),
        }
    write_json(results_dir / "metrics.json", metrics)

    default_results = ROOT / "results"
    if not quick and results_dir.resolve() == default_results.resolve():
        update_readme(ROOT / "README.md", metrics)
        (ROOT / "RESUME_NOTES.md").write_text(notes, encoding="utf-8")
    print(f"wrote {results_dir / 'metrics.json'}", flush=True)
    return metrics


def model_split_counts(predictions: pd.DataFrame) -> dict[str, int]:
    counts = predictions["split"].value_counts().to_dict()
    return {
        "train": int(counts.get("train", 0)),
        "val": int(counts.get("val", 0)),
        "test": int(counts.get("test", 0)),
    }


def _assemble_metrics(
    cfg: Config,
    criteria: list[dict],
    facts: dict,
    did: dict,
    propensity: dict,
    forecast: dict,
    silhouette: float,
    on_disk: dict,
    published: dict,
    model_loaded: bool,
    propensity_counts: dict,
    forecast_result: dict,
    hardware: dict,
    runtime: float,
) -> dict:
    return {
        "project": "b2b-gtm-analytics",
        "success_criteria": criteria,
        "metrics": {
            "simulation": facts["sample"],
            "sql": facts["retention"],
            "did": {
                "estimate": did["estimate"],
                "truth": did["truth"],
                "se": did["se"],
                "ci_low": did["ci_low"],
                "ci_high": did["ci_high"],
                "ci_covers_truth": did["ci_covers_truth"],
                "relative_error": did["relative_error"],
                "naive_estimate": did["naive_estimate"],
                "naive_relative_error": did["naive_relative_error"],
            },
            "propensity": {
                "lgbm_auroc": propensity["lgbm_auroc"],
                "logistic_auroc": propensity["logistic_auroc"],
                "auroc_lift": propensity["auroc_lift"],
                "test_positive_rate": propensity["test_positive_rate"],
            },
            "forecast": {
                "gbm_mape": forecast["gbm_mape"],
                "seasonal_naive_mape": forecast["seasonal_naive_mape"],
            },
            "segmentation": {"silhouette": silhouette, "k": 4},
            "summary": {
                "number_match_rate": on_disk["match_rate"],
                "n_numbers": on_disk["n_numbers"],
                "source": published["source"],
                "drafts_rejected": published["drafts_rejected"],
                "model_loaded": model_loaded,
            },
        },
        "baseline": {
            "naive_before_after_estimate": did["naive_estimate"],
            "naive_before_after_relative_error": did["naive_relative_error"],
            "logistic_auroc": propensity["logistic_auroc"],
            "seasonal_naive_mape": forecast["seasonal_naive_mape"],
        },
        "splits": {
            "propensity": propensity_counts,
            "forecast": {
                "train": int(forecast_result["n_train"]),
                "val": 0,
                "test": int(forecast_result["n_test"]),
            },
        },
        "seeds": [int(cfg.seed)],
        "hardware": hardware,
        "runtime_seconds": float(runtime),
    }
