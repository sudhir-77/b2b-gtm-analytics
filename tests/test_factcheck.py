from src.factcheck import drop_unverified_sentences, extract_numbers, fact_check


TABLES = {
    "n_accounts": 5000,
    "n_months": 24,
    "nrr_12m": 1.1274,
    "adoption_pct": 61.4,
    "relative_error_pct": 1.37,
    "truth": 0.08,
}


def test_extracts_commas_decimals_and_percents():
    tokens = extract_numbers("Panel of 5,000 accounts, NRR 1.127, error 1.4%.")
    rendered = [(token["value"], token["decimals"], token["percent"]) for token in tokens]
    assert rendered == [(5000.0, 0, False), (1.127, 3, False), (1.4, 1, True)]


def test_matching_numbers_pass_and_hallucinations_fail():
    good = fact_check("Retention is 1.127 on 5,000 accounts over 24 months.", TABLES)
    assert good["passed"]
    assert good["match_rate"] == 1.0
    bad = fact_check("Retention is 1.127 and the secret lift is 999.", TABLES)
    assert not bad["passed"]
    assert "999" in bad["unmatched"]


def test_percent_token_matches_fraction_or_points():
    as_points = fact_check("The error is 1.4%.", TABLES)
    assert as_points["passed"]
    as_fraction = fact_check("Truth is 0.080.", {"truth": 0.08})
    assert as_fraction["passed"]


def test_empty_summary_fails():
    assert fact_check("No figures are stated here.", TABLES)["passed"] is False


def test_drop_unverified_sentences_keeps_clean_ones():
    text = "NRR is 1.127. The imagined gain is 999. Adoption is 61.4 percent."
    kept = drop_unverified_sentences(text, TABLES)
    assert "999" not in kept
    assert "1.127" in kept
    assert fact_check(kept, TABLES)["passed"]
