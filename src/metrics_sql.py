"""Run the documented SQL metric layer and return small result tables."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

SQL_PATH = Path(__file__).resolve().parent / "sql" / "metrics.sql"


def load_queries(path: Path | None = None) -> dict[str, str]:
    text = (path or SQL_PATH).read_text(encoding="utf-8")
    parts = re.split(r"(?m)^-- name: (\w+)\s*$", text)
    queries: dict[str, str] = {}
    iterator = iter(parts[1:])
    for name, body in zip(iterator, iterator):
        queries[name] = body.strip().rstrip(";")
    if not queries:
        raise ValueError(f"no named queries in {path or SQL_PATH}")
    return queries


def run_metrics(db_path: Path) -> dict[str, pd.DataFrame]:
    queries = load_queries()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return {name: con.execute(sql).fetchdf() for name, sql in queries.items()}
    finally:
        con.close()


def summarize_metrics(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    monthly = tables["monthly_nrr"]
    annual = tables["nrr_12"]
    adoption = tables["adoption"]
    penetration = tables["penetration"]
    latest = annual.loc[annual["month"].idxmax()]
    return {
        "mean_monthly_nrr": float(monthly["nrr"].mean()),
        "nrr_12m": float(latest["nrr_12"]),
        "nrr_12m_end_month": int(latest["month"]),
        "mean_adoption": float(adoption["adoption"].mean()),
        "mean_penetration": float(penetration["mean_penetration"].iloc[0]),
        "median_penetration": float(penetration["median_penetration"].iloc[0]),
    }
