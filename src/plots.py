"""Charts written next to the metric tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_nrr(monthly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(monthly["month"], monthly["nrr"], marker="o", color="#1f4e79")
    ax.axhline(1.0, color="#888888", linewidth=1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly net revenue retention")
    ax.set_title("Simulated monthly NRR")
    _save(fig, path)


def plot_segments(assignments: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for cluster, group in assignments.groupby("cluster"):
        ax.scatter(
            group["mean_usage"],
            group["mean_log_seats"],
            s=12,
            alpha=0.7,
            label=f"cluster {cluster}",
        )
    ax.set_xlabel("Mean usage")
    ax.set_ylabel("Mean log seats")
    ax.set_title("Account segments (pre-holdout features)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_roc(predictions: pd.DataFrame, path: Path) -> None:
    test = predictions.loc[predictions["split"] == "test"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for column, label in (("p_lgbm", "LightGBM"), ("p_logistic", "Logistic")):
        fpr, tpr, _ = roc_curve(test["y_true"], test[column])
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], color="#888888", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Expansion propensity on held-out months")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_forecast(monthly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(monthly))
    width = 0.25
    ax.bar([i - width for i in x], monthly["y_true"], width=width, label="Actual")
    ax.bar(list(x), monthly["y_pred_gbm"], width=width, label="GBM")
    ax.bar(
        [i + width for i in x],
        monthly["y_pred_seasonal_naive"],
        width=width,
        label="Seasonal naive",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(int(m)) for m in monthly["month"]])
    ax.set_xlabel("Holdout month")
    ax.set_ylabel("Total revenue")
    ax.set_title("Simulated three-month revenue forecast")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_did(panel: pd.DataFrame, path: Path) -> None:
    frame = panel.copy()
    frame["arm"] = (frame["cohort"] >= 0).map({True: "eventually treated", False: "never treated"})
    means = frame.groupby(["month", "arm"], sort=True)["log_seats"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, group in means.groupby("arm"):
        ax.plot(group["month"], group["log_seats"], marker="o", label=arm)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean log seats")
    ax.set_title("Simulated log seats by field-play assignment")
    ax.legend(frameon=False)
    _save(fig, path)
