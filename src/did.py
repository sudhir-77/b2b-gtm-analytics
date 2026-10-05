"""Staggered difference-in-differences on log seat-credits.

The estimator is two-way fixed effects (account and month) with a constant
treatment effect. Under the simulator that effect is homogeneous, there is
no anticipation, and untreated trends are common, so TWFE recovers tau.
Standard errors are cluster-robust by account. The 95% interval is
estimate +/- 1.96 * SE.

The baseline is a before/after mean on accounts that eventually receive the
field play. It ignores the control group and the common time trend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.scoring import relative_error


def demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=values) / counts
    return values - means[inverse]


def twoway_demean(
    values: np.ndarray,
    entity: np.ndarray,
    time: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    current = np.asarray(values, dtype=float).copy()
    for _ in range(max_iter):
        previous = current
        current = demean(current, entity)
        current = demean(current, time)
        if np.max(np.abs(current - previous)) < tol:
            break
    return current


def _cluster_se(x: np.ndarray, y: np.ndarray, groups: np.ndarray, beta: float) -> float:
    """Cluster-robust SE for a through-origin regression, one regressor."""
    resid = y - x * beta
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    g = int(len(counts))
    scores = np.bincount(inverse, weights=x * resid)
    meat = float(np.sum(scores**2))
    bread = 1.0 / float(np.dot(x, x))
    # Liang-Zeger correction with G/(G-1).
    variance = bread * meat * bread * (g / max(g - 1, 1))
    return float(np.sqrt(max(variance, 0.0)))


def estimate_twfe(panel: pd.DataFrame, outcome: str = "log_seats") -> dict[str, float]:
    required = {"account_id", "month", "treated", outcome}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel is missing {sorted(missing)}")
    y = twoway_demean(panel[outcome].to_numpy(), panel["account_id"].to_numpy(), panel["month"].to_numpy())
    x = twoway_demean(panel["treated"].to_numpy(dtype=float), panel["account_id"].to_numpy(), panel["month"].to_numpy())
    denom = float(np.dot(x, x))
    if denom <= 0:
        raise ValueError("treatment has no within-account variation after fixed effects")
    beta = float(np.dot(x, y) / denom)
    # statsmodels is the reported SE path; the closed form above is the point estimate.
    groups = panel["account_id"].to_numpy()
    se_manual = _cluster_se(x, y, groups, beta)
    try:
        model = sm.OLS(y, x[:, None], hasconst=False).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
        se_model = float(np.asarray(model.bse).reshape(-1)[0])
    except (ValueError, np.linalg.LinAlgError):
        se_model = float("nan")
    candidates = [value for value in (se_model, se_manual) if np.isfinite(value)]
    # Prefer the larger of the two cluster calculations so the interval is not
    # accidentally tighter than either standard formula.
    se = float(max(candidates)) if candidates else 0.0
    crit = 1.96
    return {
        "estimate": beta,
        "se": se,
        "ci_low": beta - crit * se,
        "ci_high": beta + crit * se,
        "n_rows": float(len(panel)),
        "n_accounts": float(panel["account_id"].nunique()),
    }


def naive_before_after(panel: pd.DataFrame, outcome: str = "log_seats") -> float:
    """Mean outcome after treatment minus mean outcome before it, treated accounts only."""
    ever = panel["cohort"] >= 0
    treated_rows = panel.loc[ever]
    pre = treated_rows.loc[treated_rows["treated"] == 0, outcome]
    post = treated_rows.loc[treated_rows["treated"] == 1, outcome]
    if pre.empty or post.empty:
        raise ValueError("naive before/after needs pre and post rows")
    return float(post.mean() - pre.mean())


def evaluate_did(panel: pd.DataFrame, truth: float) -> dict[str, float]:
    twfe = estimate_twfe(panel)
    naive = naive_before_after(panel)
    estimate = twfe["estimate"]
    covers = float(twfe["ci_low"] <= truth <= twfe["ci_high"])
    return {
        "estimate": estimate,
        "se": twfe["se"],
        "ci_low": twfe["ci_low"],
        "ci_high": twfe["ci_high"],
        "ci_covers_truth": covers,
        "relative_error": relative_error(estimate, truth),
        "naive_estimate": naive,
        "naive_relative_error": relative_error(naive, truth),
        "truth": float(truth),
        "n_rows": twfe["n_rows"],
        "n_accounts": twfe["n_accounts"],
    }
