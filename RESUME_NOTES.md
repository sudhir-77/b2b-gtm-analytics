# Resume notes

## Bullets

- Built a simulated B2B SaaS dataset (5000 accounts, 24 months) with a SQL metric layer for NRR, adoption, and seat penetration.
- Estimated a staggered field program's effect on seat expansion with difference-in-differences (0.3% relative error vs. the simulated truth), where a naive before/after comparison was off by 120.1%.
- Scored expansion propensity with LightGBM (0.903 AUROC) and drafted executive summaries with a local LLM gated by an automated numeric fact-check.
- Forecasted three months of simulated revenue with gradient boosting (3.6% MAPE), lower than a seasonal naive baseline (13.8% MAPE).

## Skills and tools

Python, SQL, DuckDB, pandas, scikit-learn, LightGBM, statsmodels, matplotlib, transformers.

## Project summary

This is a simulated B2B SaaS panel of accounts, seats, usage, sales touches, and invoices with a known field-play effect. The analysis reports net revenue retention, segments accounts, scores expansion propensity, estimates the staggered play with difference-in-differences, and forecasts revenue, then checks every number in a local-model executive summary.

## GitHub description

Simulated B2B SaaS growth analysis: NRR, segmentation, expansion propensity, staggered DiD, and a fact-checked local summary.
