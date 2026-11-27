-- =============================================================================
-- queries.sql
-- Key analytical queries for the Predictive Maintenance pipeline
-- SQL Server / T-SQL
-- =============================================================================

-- ── Q1: Latest sensor snapshot per machine (used for real-time dashboard) ─────
CREATE OR ALTER VIEW vw_latest_sensor AS
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY machine_id
               ORDER BY timestamp DESC
           ) AS rn
    FROM sensor_readings
)
SELECT
    r.machine_id,
    m.machine_name,
    m.location,
    r.timestamp        AS last_reading,
    r.temperature,
    r.vibration,
    r.pressure,
    r.rpm,
    r.current,
    r.oil_level,
    DATEDIFF(MINUTE, r.timestamp, GETDATE()) AS minutes_since_reading
FROM ranked r
JOIN machines m ON r.machine_id = m.machine_id
WHERE r.rn = 1;
GO


-- ── Q2: Rolling 24-hr averages + anomaly flags (core feature view) ────────────
CREATE OR ALTER VIEW vw_sensor_features AS
SELECT
    machine_id,
    timestamp,
    temperature,
    vibration,
    current,
    oil_level,

    -- 6-hour rolling averages
    AVG(temperature) OVER (
        PARTITION BY machine_id
        ORDER BY timestamp
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS temp_avg_6h,

    AVG(vibration) OVER (
        PARTITION BY machine_id
        ORDER BY timestamp
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS vib_avg_6h,

    -- 24-hour rolling averages
    AVG(temperature) OVER (
        PARTITION BY machine_id
        ORDER BY timestamp
        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
    ) AS temp_avg_24h,

    AVG(current) OVER (
        PARTITION BY machine_id
        ORDER BY timestamp
        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
    ) AS current_avg_24h,

    -- Rate of change (current vs 1 hour ago)
    temperature - LAG(temperature, 1) OVER (
        PARTITION BY machine_id ORDER BY timestamp
    ) AS temp_delta_1h,

    vibration - LAG(vibration, 1) OVER (
        PARTITION BY machine_id ORDER BY timestamp
    ) AS vib_delta_1h,

    -- Anomaly flags (1 if outside normal range)
    CASE WHEN temperature > 95  OR temperature < 60  THEN 1 ELSE 0 END AS temp_anomaly,
    CASE WHEN vibration   > 4.5 OR vibration   < 0.1 THEN 1 ELSE 0 END AS vib_anomaly,
    CASE WHEN oil_level   < 30                         THEN 1 ELSE 0 END AS low_oil_flag,

    -- Count of anomalies in last 6 hours
    SUM(CASE WHEN temperature > 95 THEN 1 ELSE 0 END) OVER (
        PARTITION BY machine_id
        ORDER BY timestamp
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS temp_spikes_6h,

    failure_within_24h

FROM sensor_readings;
GO


-- ── Q3: Machine health score report ──────────────────────────────────────────
-- Combines latest predictions with maintenance history
SELECT
    p.machine_id,
    m.machine_name,
    m.location,
    p.failure_prob,
    p.risk_level,
    p.top_feature_1,
    p.action_recommended,
    DATEDIFF(DAY, ml.last_maintenance, GETDATE()) AS days_since_maintenance,
    ml.total_downtime_ytd,
    ml.total_cost_ytd,
    -- Priority score for maintenance scheduling (higher = more urgent)
    ROUND(
        p.failure_prob * 100 +
        CASE p.risk_level
            WHEN 'CRITICAL' THEN 40
            WHEN 'HIGH'     THEN 20
            WHEN 'MEDIUM'   THEN 5
            ELSE 0
        END +
        DATEDIFF(DAY, ml.last_maintenance, GETDATE()) * 0.5,
    2) AS priority_score
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY machine_id ORDER BY prediction_ts DESC
           ) AS rn
    FROM failure_predictions
) p
JOIN machines m ON p.machine_id = m.machine_id
LEFT JOIN (
    SELECT
        machine_id,
        MAX(maintenance_date)      AS last_maintenance,
        SUM(downtime_hours)        AS total_downtime_ytd,
        SUM(cost_inr)              AS total_cost_ytd
    FROM maintenance_log
    WHERE maintenance_date >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1)
    GROUP BY machine_id
) ml ON p.machine_id = ml.machine_id
WHERE p.rn = 1
ORDER BY priority_score DESC;
GO


-- ── Q4: Month-over-month failure frequency trend ──────────────────────────────
SELECT
    machine_id,
    FORMAT(timestamp, 'yyyy-MM')              AS month,
    COUNT(*)                                   AS total_readings,
    SUM(failure_within_24h)                    AS failure_events,
    ROUND(AVG(CAST(failure_within_24h AS FLOAT)) * 100, 2) AS failure_rate_pct,
    -- Month-over-month change
    SUM(failure_within_24h) - LAG(SUM(failure_within_24h)) OVER (
        PARTITION BY machine_id
        ORDER BY FORMAT(timestamp, 'yyyy-MM')
    ) AS failure_mom_change
FROM sensor_readings
GROUP BY machine_id, FORMAT(timestamp, 'yyyy-MM')
ORDER BY machine_id, month;
GO


-- ── Q5: Machines with high anomaly counts (last 24 hours) ────────────────────
-- Use this to quickly flag which machines need attention RIGHT NOW
SELECT
    machine_id,
    COUNT(*)                                          AS readings_24h,
    SUM(CASE WHEN temperature > 95  THEN 1 ELSE 0 END) AS temp_spikes,
    SUM(CASE WHEN vibration   > 4.5 THEN 1 ELSE 0 END) AS vib_spikes,
    SUM(CASE WHEN oil_level   < 30  THEN 1 ELSE 0 END) AS low_oil_count,
    MAX(temperature)                                  AS max_temp,
    MAX(vibration)                                    AS max_vibration,
    AVG(oil_level)                                    AS avg_oil_level
FROM sensor_readings
WHERE timestamp >= DATEADD(HOUR, -24, GETDATE())
GROUP BY machine_id
HAVING
    SUM(CASE WHEN temperature > 95  THEN 1 ELSE 0 END) > 3
    OR SUM(CASE WHEN vibration > 4.5 THEN 1 ELSE 0 END) > 3
    OR SUM(CASE WHEN oil_level < 30  THEN 1 ELSE 0 END) > 5
ORDER BY temp_spikes + vib_spikes DESC;
GO
