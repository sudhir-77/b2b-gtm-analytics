from src.factcheck import fact_check
from src.reporting import render_resume_notes, success_criteria, update_readme
from tests.test_summarize import _facts


def _did(**kwargs):
    base = {
        "relative_error": 0.01,
        "ci_covers_truth": 1.0,
        "estimate": 0.08,
        "naive_relative_error": 1.2,
    }
    base.update(kwargs)
    return base


def _prop(**kwargs):
    base = {"lgbm_auroc": 0.86, "logistic_auroc": 0.74}
    base.update(kwargs)
    return base


def _forecast(**kwargs):
    base = {"gbm_mape": 0.03, "seasonal_naive_mape": 0.14}
    base.update(kwargs)
    return base


def test_success_flags_require_the_baseline_comparisons():
    good = success_criteria(_did(), _prop(), _forecast(), 0.4, 1.0, 82.0)
    by_name = {item["name"]: item for item in good}
    assert all(item["passed"] for item in good)
    missed = success_criteria(_did(ci_covers_truth=0.0), _prop(), _forecast(), 0.4, 1.0, None)
    assert missed[0]["passed"] is False
    close_models = success_criteria(_did(), _prop(lgbm_auroc=0.76, logistic_auroc=0.75), _forecast(), 0.4, 1.0, None)
    assert close_models[1]["passed"] is False
    weak_forecast = success_criteria(_did(), _prop(), _forecast(gbm_mape=0.05, seasonal_naive_mape=0.04), 0.4, 1.0, None)
    assert weak_forecast[2]["passed"] is False
    assert by_name["Test coverage of src/"]["threshold"] == 70.0


def test_resume_notes_quote_only_computed_numbers():
    facts = _facts()
    notes = render_resume_notes(facts)
    assert fact_check(notes, facts)["passed"]
    assert "simulated" in notes.lower()
    assert "—" not in notes


def test_readme_results_block_is_replaced(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("Intro\n\n<!-- RESULTS_START -->\nold\n<!-- RESULTS_END -->\n", encoding="utf-8")
    metrics = {
        "success_criteria": success_criteria(_did(), _prop(), _forecast(), 0.4, 1.0, 80.0),
        "metrics": {
            "did": {"estimate": 0.081, "relative_error": 0.01, "naive_relative_error": 1.2},
            "propensity": {"lgbm_auroc": 0.86, "logistic_auroc": 0.74},
            "forecast": {"gbm_mape": 0.03, "seasonal_naive_mape": 0.14},
            "segmentation": {"silhouette": 0.4},
            "summary": {"number_match_rate": 1.0},
        },
    }
    update_readme(readme, metrics)
    text = readme.read_text(encoding="utf-8")
    assert "\nold\n" not in text
    assert "LightGBM AUROC" in text
    assert text.startswith("Intro")
