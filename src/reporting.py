"""Metrics JSON, resume notes, and the README results block."""

from __future__ import annotations

import json
from pathlib import Path

from src.factcheck import fact_check

RESULTS_START = "<!-- RESULTS_START -->"
RESULTS_END = "<!-- RESULTS_END -->"
BANNED = ("leveraged", "robust", "cutting-edge", "seamless", "spearheaded", "\u2014", "\u2013")


def success_criteria(
    did: dict,
    propensity: dict,
    forecast: dict,
    silhouette: float,
    match_rate: float,
    coverage_pct: float | None,
) -> list[dict]:
    criteria = [
        {
            "name": "Field-play effect estimate (DiD) vs. simulated truth",
            "value": float(did["relative_error"]),
            "op": "<=",
            "threshold": 0.15,
            "threshold_high": None,
            "passed": bool(did["relative_error"] <= 0.15 and did["ci_covers_truth"]),
        },
        {
            "name": "Expansion propensity AUROC on held-out months",
            "value": float(propensity["lgbm_auroc"]),
            "op": ">=",
            "threshold": 0.75,
            "threshold_high": None,
            "passed": bool(
                propensity["lgbm_auroc"] >= 0.75
                and propensity["lgbm_auroc"] >= propensity["logistic_auroc"] + 0.02
            ),
        },
        {
            "name": "3-month revenue forecast MAPE",
            "value": float(forecast["gbm_mape"]),
            "op": "<=",
            "threshold": 0.08,
            "threshold_high": None,
            "passed": bool(
                forecast["gbm_mape"] <= 0.08
                and forecast["gbm_mape"] < forecast["seasonal_naive_mape"]
            ),
        },
        {
            "name": "Segmentation silhouette score",
            "value": float(silhouette),
            "op": ">=",
            "threshold": 0.25,
            "threshold_high": None,
            "passed": bool(silhouette >= 0.25),
        },
        {
            "name": "Numbers in the LLM summary that match computed tables",
            "value": float(match_rate),
            "op": ">=",
            "threshold": 1.0,
            "threshold_high": None,
            "passed": bool(match_rate >= 1.0),
        },
    ]
    if coverage_pct is not None:
        criteria.append(
            {
                "name": "Test coverage of src/",
                "value": float(coverage_pct),
                "op": ">=",
                "threshold": 70.0,
                "threshold_high": None,
                "passed": bool(coverage_pct >= 70.0),
            }
        )
    return criteria


def render_resume_notes(facts: dict) -> str:
    sample = facts["sample"]
    field = facts["field_play"]
    propensity = facts["propensity"]
    forecast = facts["forecast"]
    did_pct = f"{field['relative_error_pct']:.1f}"
    naive_pct = f"{field['naive_relative_error_pct']:.1f}"
    auc = f"{propensity['lgbm_auroc']:.3f}"
    mape_pct = f"{forecast['gbm_mape_pct']:.1f}"
    naive_mape_pct = f"{forecast['seasonal_naive_mape_pct']:.1f}"
    text = f"""# Resume notes

## Bullets

- Built a simulated B2B SaaS dataset ({int(sample['n_accounts'])} accounts, {int(sample['n_months'])} months) with a SQL metric layer for NRR, adoption, and seat penetration.
- Estimated a staggered field program's effect on seat expansion with difference-in-differences ({did_pct}% relative error vs. the simulated truth), where a naive before/after comparison was off by {naive_pct}%.
- Scored expansion propensity with LightGBM ({auc} AUROC) and drafted executive summaries with a local LLM gated by an automated numeric fact-check.
- Forecasted three months of simulated revenue with gradient boosting ({mape_pct}% MAPE), lower than a seasonal naive baseline ({naive_mape_pct}% MAPE).

## Skills and tools

Python, SQL, DuckDB, pandas, scikit-learn, LightGBM, statsmodels, matplotlib, transformers.

## Project summary

This is a simulated B2B SaaS panel of accounts, seats, usage, sales touches, and invoices with a known field-play effect. The analysis reports net revenue retention, segments accounts, scores expansion propensity, estimates the staggered play with difference-in-differences, and forecasts revenue, then checks every number in a local-model executive summary.

## GitHub description

Simulated B2B SaaS growth analysis: NRR, segmentation, expansion propensity, staggered DiD, and a fact-checked local summary.
"""
    lowered = text.lower()
    for word in BANNED:
        if word in lowered or word in text:
            raise ValueError(f"resume notes contain banned wording: {word}")
    return text


def render_results_markdown(metrics: dict) -> str:
    lines = [
        "| Criterion | Value | Threshold | Passed |",
        "|---|---:|---:|:---|",
    ]
    for item in metrics["success_criteria"]:
        high = item.get("threshold_high")
        threshold = f"{item['op']} {item['threshold']}"
        if high is not None:
            threshold = f"{item['threshold']} to {high}"
        lines.append(
            f"| {item['name']} | {item['value']:.4f} | {threshold} | {'yes' if item['passed'] else 'no'} |"
        )
    headline = metrics["metrics"]
    lines.extend(
        [
            "",
            "Headline figures from the same file:",
            "",
            "| Figure | Value |",
            "|---|---:|",
            f"| DiD estimate | {headline['did']['estimate']:.4f} |",
            f"| DiD relative error | {headline['did']['relative_error']:.4f} |",
            f"| Naive relative error | {headline['did']['naive_relative_error']:.4f} |",
            f"| LightGBM AUROC | {headline['propensity']['lgbm_auroc']:.4f} |",
            f"| Logistic AUROC | {headline['propensity']['logistic_auroc']:.4f} |",
            f"| Forecast MAPE | {headline['forecast']['gbm_mape']:.4f} |",
            f"| Seasonal naive MAPE | {headline['forecast']['seasonal_naive_mape']:.4f} |",
            f"| Silhouette | {headline['segmentation']['silhouette']:.4f} |",
            f"| Summary number match rate | {headline['summary']['number_match_rate']:.4f} |",
        ]
    )
    return "\n".join(lines)


def update_readme(path: Path, metrics: dict) -> None:
    block = render_results_markdown(metrics)
    replacement = f"{RESULTS_START}\n\n{block}\n\n{RESULTS_END}"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if RESULTS_START in text and RESULTS_END in text:
        pre, rest = text.split(RESULTS_START, 1)
        _, post = rest.split(RESULTS_END, 1)
        text = pre + replacement + post
    else:
        text = text.rstrip() + "\n\n" + replacement + "\n"
    path.write_text(text, encoding="utf-8")


def resume_matches_metrics(notes: str, metrics: dict) -> bool:
    return bool(fact_check(notes, metrics)["passed"])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
