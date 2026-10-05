"""Simulated B2B SaaS accounts, usage, touches, and invoices.

Generative model (continuous seat-credits, so the DiD outcome is the log
the billing system stores):

    log(seats_it) = alpha_i + delta * t + gamma * sin(2 * pi * t / 12)
                    + tau * D_it + eps_it

D_it is 1 on and after the account's field-play cohort month. Never-treated
accounts stay at 0. tau is ``field_play_effect`` (default 0.08). Assignment
does not depend on segment, usage, or the noise draws. Trends and seasonality
are common, so two-way fixed effects can recover tau.

Usage follows a segment adoption curve that finishes its ramp early, then
moves with a small seasonal term. The expansion-qualified label is a Bernoulli
draw from a nonlinear rule on lagged usage and touches (product-led or
sales-assisted, not both), plus a bump in the month the field play starts.
That label is the propensity target. It is not written back into seats, so
the seat panel stays a clean staggered-adoption design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config

FEATURE_NAMES = (
    "admin",
    "analytics",
    "integrations",
    "automation",
    "reporting",
    "api",
)
CHANNELS = ("email", "call", "meeting")
USAGE_HIGH = 0.50
TOUCHES_HIGH = 3


@dataclass
class SimulatedData:
    accounts: pd.DataFrame
    account_month: pd.DataFrame
    usage_events: pd.DataFrame
    touches: pd.DataFrame
    invoices: pd.DataFrame
    expected_counts: dict[str, int]

    @property
    def total_rows(self) -> int:
        return int(sum(self.expected_counts.values()))


def expansion_probability(
    usage_lag: np.ndarray,
    touches_lag: np.ndarray,
    treatment_onset: np.ndarray,
) -> np.ndarray:
    """P(expansion-qualified) from lagged product usage and sales touches.

    The two qualified motions are mutually exclusive: high usage with few
    touches (product-led), or low usage with many touches (sales-assisted).
    Accounts with both or neither are mostly unqualified. A field-play onset
    month adds a linear bump. Trees can represent the interaction; a linear
    logistic model cannot.
    """
    usage = np.asarray(usage_lag, dtype=float)
    touches = np.asarray(touches_lag, dtype=float)
    onset = np.asarray(treatment_onset, dtype=float)
    high_u = usage >= USAGE_HIGH
    high_t = touches >= TOUCHES_HIGH
    xor = np.logical_xor(high_u, high_t).astype(float)
    logit = -2.35 + 4.6 * xor + 2.0 * onset
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def _assign_segments(rng: np.random.Generator, cfg: Config) -> np.ndarray:
    shares = np.array([seg.share for seg in cfg.segments], dtype=float)
    cuts = np.cumsum(shares)
    draws = rng.random(cfg.n_accounts)
    idx = np.searchsorted(cuts, draws, side="right")
    idx = np.clip(idx, 0, len(cfg.segments) - 1)
    return idx


def _assign_cohorts(rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """Cohort month, or -1 when the account is never treated."""
    n = cfg.n_accounts
    cohorts = np.full(n, -1, dtype=np.int32)
    treated = rng.random(n) >= cfg.control_fraction
    n_cohorts = len(cfg.treatment_cohorts)
    pick = rng.integers(0, n_cohorts, size=n)
    cohort_values = np.asarray(cfg.treatment_cohorts, dtype=np.int32)
    cohorts[treated] = cohort_values[pick[treated]]
    return cohorts


def simulate(cfg: Config, rng: np.random.Generator | None = None) -> SimulatedData:
    cfg.validate()
    rng = rng or np.random.default_rng(cfg.seed)
    n = cfg.n_accounts
    t = cfg.n_months
    months = np.arange(t, dtype=np.int32)

    seg_idx = _assign_segments(rng, cfg)
    cohorts = _assign_cohorts(rng, cfg)
    seg_names = np.array([seg.name for seg in cfg.segments])
    names = seg_names[seg_idx]

    alpha = np.empty(n, dtype=float)
    usage_asym = np.empty(n, dtype=float)
    usage_sd = np.empty(n, dtype=float)
    touch_mean = np.empty(n, dtype=float)
    target_pen = np.empty(n, dtype=float)
    price = np.empty(n, dtype=float)
    for i, seg in enumerate(cfg.segments):
        mask = seg_idx == i
        count = int(mask.sum())
        if count == 0:
            continue
        alpha[mask] = rng.normal(seg.log_seats_mean, seg.log_seats_sd, size=count)
        usage_asym[mask] = seg.usage_asymptote
        usage_sd[mask] = seg.usage_sd
        touch_mean[mask] = seg.touch_mean
        target_pen[mask] = seg.target_penetration
        price[mask] = seg.price_per_seat

    employees = np.exp(alpha) / target_pen

    treated = np.zeros((n, t), dtype=np.int8)
    onset = np.zeros((n, t), dtype=np.int8)
    for i, cohort in enumerate(cohorts):
        if cohort < 0:
            continue
        treated[i, cohort:] = 1
        onset[i, cohort] = 1

    season = cfg.season_amp * np.sin(2.0 * np.pi * months / 12.0)
    eps = rng.normal(0.0, cfg.noise_sigma, size=(n, t))
    log_seats = (
        alpha[:, None]
        + cfg.trend_per_month * months[None, :]
        + season[None, :]
        + cfg.field_play_effect * treated
        + eps
    )
    seats = np.exp(log_seats)

    ramp = np.minimum(1.0, (months + 1) / float(cfg.adoption_ramp_months))
    usage_noise = rng.normal(0.0, 1.0, size=(n, t)) * usage_sd[:, None]
    usage_season = 0.02 * np.sin(2.0 * np.pi * months / 12.0 + 0.4)
    usage = usage_asym[:, None] * (0.62 + 0.38 * ramp[None, :]) + usage_season[None, :]
    usage = np.clip(usage + usage_noise, 0.02, 0.98)

    touches = rng.poisson(touch_mean[:, None], size=(n, t)).astype(np.int32)
    penetration = seats / employees[:, None]

    rev_season = 1.0 + 0.03 * np.sin(2.0 * np.pi * months / 12.0 + 0.5)
    rev_noise = rng.normal(0.0, cfg.revenue_noise, size=(n, t))
    revenue = price[:, None] * seats * rev_season[None, :] * (1.0 + rev_noise)
    revenue = np.maximum(revenue, 0.01)

    usage_lag = np.full((n, t), np.nan)
    touches_lag = np.full((n, t), np.nan)
    penetration_lag = np.full((n, t), np.nan)
    log_seats_lag = np.full((n, t), np.nan)
    usage_lag[:, 1:] = usage[:, :-1]
    touches_lag[:, 1:] = touches[:, :-1]
    penetration_lag[:, 1:] = penetration[:, :-1]
    log_seats_lag[:, 1:] = log_seats[:, :-1]

    prob = np.zeros((n, t), dtype=float)
    prob[:, 1:] = expansion_probability(usage_lag[:, 1:], touches_lag[:, 1:], onset[:, 1:])
    expanded = np.zeros((n, t), dtype=np.int8)
    expanded[:, 1:] = (rng.random((n, t - 1)) < prob[:, 1:]).astype(np.int8)

    account_ids = np.arange(n, dtype=np.int32)
    month_grid = np.broadcast_to(months, (n, t))
    id_grid = np.broadcast_to(account_ids[:, None], (n, t))

    account_month = pd.DataFrame(
        {
            "account_id": id_grid.ravel(),
            "month": month_grid.ravel().astype(np.int32),
            "segment": np.repeat(names, t),
            "cohort": np.repeat(cohorts, t).astype(np.int32),
            "treated": treated.ravel().astype(np.int8),
            "treatment_onset": onset.ravel().astype(np.int8),
            "seats": seats.ravel(),
            "log_seats": log_seats.ravel(),
            "usage": usage.ravel(),
            "touches": touches.ravel().astype(np.int32),
            "employees": np.repeat(employees, t),
            "penetration": penetration.ravel(),
            "revenue": revenue.ravel(),
            "expanded": expanded.ravel().astype(np.int8),
            "usage_lag": usage_lag.ravel(),
            "touches_lag": touches_lag.ravel(),
            "penetration_lag": penetration_lag.ravel(),
            "log_seats_lag": log_seats_lag.ravel(),
        }
    )

    accounts = pd.DataFrame(
        {
            "account_id": account_ids,
            "segment": names,
            "cohort": cohorts,
            "employees": employees,
            "log_seats_base": alpha,
            "price_per_seat": price,
            "usage_asymptote": usage_asym,
        }
    )

    events_per = cfg.usage_events_per_account_month
    n_events = n * t * events_per
    event_accounts = np.repeat(id_grid.ravel(), events_per)
    event_months = np.repeat(month_grid.ravel(), events_per)
    usage_flat = np.repeat(usage.ravel(), events_per)
    # Higher usage opens more of the feature surface. Sampling with a
    # usage-dependent cap makes distinct-feature adoption a real metric.
    n_active = np.clip(np.rint(1 + usage_flat * 5).astype(np.int32), 1, len(FEATURE_NAMES))
    feature_id = np.empty(n_events, dtype=np.int32)
    # Vectorized sample from 0..n_active-1 without a Python loop over rows.
    feature_id = (rng.random(n_events) * n_active).astype(np.int32)
    feature_id = np.minimum(feature_id, n_active - 1)
    usage_events = pd.DataFrame(
        {
            "account_id": event_accounts.astype(np.int32),
            "month": event_months.astype(np.int32),
            "event_idx": np.tile(np.arange(events_per, dtype=np.int32), n * t),
            "feature_id": feature_id,
        }
    )

    touch_counts = touches.ravel()
    n_touch_rows = int(touch_counts.sum())
    if n_touch_rows:
        touch_accounts = np.repeat(id_grid.ravel(), touch_counts)
        touch_months = np.repeat(month_grid.ravel(), touch_counts)
        # touch index within the account-month
        offsets = np.repeat(np.cumsum(touch_counts) - touch_counts, touch_counts)
        seq = np.arange(n_touch_rows, dtype=np.int32) - offsets.astype(np.int32)
        channel = rng.integers(0, len(CHANNELS), size=n_touch_rows)
        touches_df = pd.DataFrame(
            {
                "account_id": touch_accounts.astype(np.int32),
                "month": touch_months.astype(np.int32),
                "touch_idx": seq,
                "channel_id": channel.astype(np.int32),
            }
        )
    else:
        touches_df = pd.DataFrame(
            columns=["account_id", "month", "touch_idx", "channel_id"]
        )

    invoices = pd.DataFrame(
        {
            "account_id": id_grid.ravel().astype(np.int32),
            "month": month_grid.ravel().astype(np.int32),
            "revenue": revenue.ravel(),
            "seats": seats.ravel(),
        }
    )

    expected = {
        "accounts": int(n),
        "account_month": int(n * t),
        "usage_events": int(n_events),
        "touches": int(n_touch_rows),
        "invoices": int(n * t),
    }
    return SimulatedData(
        accounts=accounts,
        account_month=account_month,
        usage_events=usage_events,
        touches=touches_df,
        invoices=invoices,
        expected_counts=expected,
    )


def write_duckdb(data: SimulatedData, path) -> dict[str, int]:
    """Persist the panel and return verified row counts."""
    import duckdb

    path = path if hasattr(path, "suffix") else path
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    try:
        con.register("accounts_df", data.accounts)
        con.register("am_df", data.account_month)
        con.register("events_df", data.usage_events)
        con.register("touches_df", data.touches)
        con.register("invoices_df", data.invoices)
        con.execute("CREATE TABLE accounts AS SELECT * FROM accounts_df")
        con.execute("CREATE TABLE account_month AS SELECT * FROM am_df")
        con.execute("CREATE TABLE usage_events AS SELECT * FROM events_df")
        con.execute("CREATE TABLE touches AS SELECT * FROM touches_df")
        con.execute("CREATE TABLE invoices AS SELECT * FROM invoices_df")
        counts = {}
        for table, expected in data.expected_counts.items():
            got = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if int(got) != int(expected):
                raise RuntimeError(f"{table} row count {got} != {expected}")
            counts[table] = int(got)
        return counts
    finally:
        con.close()
