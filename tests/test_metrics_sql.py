import duckdb

from src.metrics_sql import load_queries, summarize_metrics


def test_named_queries_exist():
    queries = load_queries()
    assert set(queries) == {"monthly_nrr", "nrr_12", "adoption", "penetration"}


def test_sql_metrics_match_hand_computed_values(tmp_path):
    db = tmp_path / "tiny.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE account_month (
            account_id INTEGER, month INTEGER, revenue DOUBLE,
            seats DOUBLE, employees DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO account_month VALUES
            (1, 0, 100, 25, 100),
            (2, 0, 100, 50, 100),
            (1, 1, 150, 75, 100),
            (2, 1, 50, 100, 100),
            (1, 12, 180, 25, 100),
            (2, 12, 90, 50, 100)
        """
    )
    con.execute(
        """
        CREATE TABLE usage_events (account_id INTEGER, month INTEGER, feature_id INTEGER)
        """
    )
    con.execute(
        """
        INSERT INTO usage_events VALUES
            (1, 0, 0), (1, 0, 0), (1, 0, 1),
            (2, 0, 0), (2, 0, 1), (2, 0, 2)
        """
    )
    queries = load_queries()
    tables = {name: con.execute(sql).fetchdf() for name, sql in queries.items()}
    con.close()
    summary = summarize_metrics(tables)
    # Month 1 revenue 200 / month 0 revenue 200.
    assert float(tables["monthly_nrr"].loc[tables["monthly_nrr"]["month"] == 1, "nrr"].iloc[0]) == 1.0
    # Month 12 revenue 270 / month 0 revenue 200.
    assert summary["nrr_12m"] == 1.35
    # One of two accounts used 3 distinct features.
    assert summary["mean_adoption"] == 0.5
    # Penetrations 0.25, 0.50, 0.75, 1.00, 0.25, 0.50. Mean = 0.5416...
    assert summary["mean_penetration"] == (0.25 + 0.50 + 0.75 + 1.0 + 0.25 + 0.50) / 6
