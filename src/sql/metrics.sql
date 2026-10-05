-- Metric definitions for the simulated B2B panel.
-- Revenue is positive for every account-month and the panel is balanced
-- (no logo churn). Monthly NRR therefore reflects expansion and contraction
-- only. That limitation is stated in the README.

-- name: monthly_nrr
-- Net revenue retention for month t:
--   numerator   = revenue in month t from accounts that also had revenue in t-1
--   denominator = those same accounts' revenue in month t-1
-- Because every account is present in every month, this equals total
-- revenue_t / total revenue_{t-1}.
SELECT
    c.month AS month,
    SUM(c.revenue) / SUM(p.revenue) AS nrr
FROM account_month AS c
JOIN account_month AS p
    ON c.account_id = p.account_id
   AND c.month = p.month + 1
GROUP BY c.month
ORDER BY c.month

-- name: nrr_12
-- Twelve-month net revenue retention ending in month t:
--   revenue_t / revenue_{t-12} on the same accounts.
SELECT
    c.month AS month,
    SUM(c.revenue) / SUM(p.revenue) AS nrr_12
FROM account_month AS c
JOIN account_month AS p
    ON c.account_id = p.account_id
   AND c.month = p.month + 12
GROUP BY c.month
ORDER BY c.month

-- name: adoption
-- Feature adoption: share of accounts that used at least 3 distinct product
-- features in the month. usage_events has one row per simulated event.
SELECT
    month,
    AVG(CASE WHEN n_features >= 3 THEN 1.0 ELSE 0.0 END) AS adoption
FROM (
    SELECT
        account_id,
        month,
        COUNT(DISTINCT feature_id) AS n_features
    FROM usage_events
    GROUP BY account_id, month
) AS feat
GROUP BY month
ORDER BY month

-- name: penetration
-- Seat penetration = seat-credits / employees at the account's baseline
-- headcount. Reported as the mean and median across account-months.
SELECT
    AVG(seats / employees) AS mean_penetration,
    MEDIAN(seats / employees) AS median_penetration
FROM account_month
