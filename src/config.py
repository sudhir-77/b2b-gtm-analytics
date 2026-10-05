"""Load the simulator config and apply the quick-run profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SegmentConfig:
    name: str
    share: float
    log_seats_mean: float
    log_seats_sd: float
    usage_asymptote: float
    usage_sd: float
    touch_mean: float
    target_penetration: float
    price_per_seat: float


@dataclass
class Config:
    seed: int
    n_accounts: int
    n_months: int
    field_play_effect: float
    control_fraction: float
    treatment_cohorts: list[int]
    trend_per_month: float
    season_amp: float
    noise_sigma: float
    revenue_noise: float
    usage_events_per_account_month: int
    adoption_ramp_months: int
    lgbm_n_estimators: int
    forecast_n_estimators: int
    llm_max_new_tokens: int
    model_repo: str
    model_sha256: str
    segments: list[SegmentConfig] = field(default_factory=list)
    quick: bool = False

    def validate(self) -> None:
        shares = sum(seg.share for seg in self.segments)
        if abs(shares - 1.0) > 1e-6:
            raise ValueError(f"segment shares must sum to 1, got {shares}")
        if self.n_months < 18:
            raise ValueError("n_months must be at least 18 so the time split fits")
        if self.n_accounts < 80:
            raise ValueError("n_accounts must be at least 80")
        if not (0 < self.control_fraction < 1):
            raise ValueError("control_fraction must be between 0 and 1")
        if any(c < 1 or c >= self.n_months - 1 for c in self.treatment_cohorts):
            raise ValueError("treatment cohorts must sit inside the panel")


def _segment(raw: dict) -> SegmentConfig:
    return SegmentConfig(
        name=str(raw["name"]),
        share=float(raw["share"]),
        log_seats_mean=float(raw["log_seats_mean"]),
        log_seats_sd=float(raw["log_seats_sd"]),
        usage_asymptote=float(raw["usage_asymptote"]),
        usage_sd=float(raw["usage_sd"]),
        touch_mean=float(raw["touch_mean"]),
        target_penetration=float(raw["target_penetration"]),
        price_per_seat=float(raw["price_per_seat"]),
    )


def load_config(path: Path, quick: bool = False) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    quick_raw = dict(raw.get("quick") or {})
    cfg = Config(
        seed=int(raw["seed"]),
        n_accounts=int(raw["n_accounts"]),
        n_months=int(raw["n_months"]),
        field_play_effect=float(raw["field_play_effect"]),
        control_fraction=float(raw["control_fraction"]),
        treatment_cohorts=[int(c) for c in raw["treatment_cohorts"]],
        trend_per_month=float(raw["trend_per_month"]),
        season_amp=float(raw["season_amp"]),
        noise_sigma=float(raw["noise_sigma"]),
        revenue_noise=float(raw["revenue_noise"]),
        usage_events_per_account_month=int(raw["usage_events_per_account_month"]),
        adoption_ramp_months=int(raw["adoption_ramp_months"]),
        lgbm_n_estimators=int(raw["lgbm_n_estimators"]),
        forecast_n_estimators=int(raw["forecast_n_estimators"]),
        llm_max_new_tokens=int(raw["llm_max_new_tokens"]),
        model_repo=str(raw["model_repo"]),
        model_sha256=str(raw["model_sha256"]),
        segments=[_segment(s) for s in raw["segments"]],
        quick=quick,
    )
    if quick:
        for key in (
            "n_accounts",
            "usage_events_per_account_month",
            "lgbm_n_estimators",
            "forecast_n_estimators",
            "llm_max_new_tokens",
        ):
            if key in quick_raw:
                setattr(cfg, key, type(getattr(cfg, key))(quick_raw[key]))
    cfg.validate()
    return cfg
