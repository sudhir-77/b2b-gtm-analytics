from src.factcheck import fact_check
from src.summarize import canonical_summary, publish_summary


def _facts():
    return {
        "sample": {"n_accounts": 5000, "n_months": 24, "duckdb_rows": 2100000},
        "retention": {
            "nrr_12m": 1.1274,
            "mean_monthly_nrr": 1.012,
            "mean_adoption": 0.614,
            "mean_adoption_pct": 61.4,
            "mean_penetration": 0.482,
            "mean_penetration_pct": 48.2,
            "median_penetration": 0.45,
        },
        "field_play": {
            "truth": 0.08,
            "truth_pct": 8.0,
            "estimate": 0.0812,
            "se": 0.001,
            "ci_low": 0.079,
            "ci_high": 0.083,
            "relative_error": 0.015,
            "relative_error_pct": 1.5,
            "naive_estimate": 0.19,
            "naive_relative_error": 1.375,
            "naive_relative_error_pct": 137.5,
        },
        "propensity": {"lgbm_auroc": 0.861, "logistic_auroc": 0.742, "auroc_lift": 0.119},
        "forecast": {
            "gbm_mape": 0.024,
            "gbm_mape_pct": 2.4,
            "seasonal_naive_mape": 0.138,
            "seasonal_naive_mape_pct": 13.8,
            "horizon_months": 3,
        },
        "segmentation": {"silhouette": 0.512, "k": 4},
    }


def test_canonical_summary_uses_only_table_numbers():
    facts = _facts()
    text = canonical_summary(facts)
    checked = fact_check(text, facts)
    assert checked["passed"], checked["unmatched"]
    assert "simulated" in text.lower()


def test_publish_keeps_a_clean_draft_and_rejects_a_bad_one():
    facts = _facts()
    canonical = canonical_summary(facts)

    def echo(prompt: str) -> str:
        return prompt.split("\n\n", 1)[1]

    clean = publish_summary(facts, echo, attempts=2)
    assert clean["source"] == "llm"
    assert fact_check(clean["text"], facts)["passed"]
    assert clean["text"] == canonical

    def hallucinate(prompt: str) -> str:
        return "The program returned 123456 dollars and grew 999 percent."

    gated = publish_summary(facts, hallucinate, attempts=2)
    assert gated["source"] == "gated_fallback"
    assert gated["text"] == canonical
    assert gated["drafts_rejected"] == 2
    assert fact_check(gated["text"], facts)["passed"]
