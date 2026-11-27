-- =============================================================================
-- schema.sql
-- Predictive Maintenance Database Schema
-- SQL Server / T-SQL compatible
-- =============================================================================

USE manufacturing_db;
GO

-- ── 1. Machines master table ──────────────────────────────────────────────────
CREATE TABLE machines (
    machine_id       VARCHAR(10)  NOT NULL PRIMARY KEY,
    machine_name     VARCHAR(100) NOT NULL,
    machine_type     VARCHAR(50)  NOT NULL,   -- e.g., 'CNC Lathe', 'Conveyor'
    location         VARCHAR(50),
    install_date     DATE,
    last_service     DATE,
    status           VARCHAR(20)  DEFAULT 'active'
        CHECK (status IN ('active', 'maintenance', 'decommissioned')),
    created_at       DATETIME     DEFAULT GETDATE()
);
GO

-- ── 2. Sensor readings (partitioned by month for performance) ─────────────────
CREATE TABLE sensor_readings (
    reading_id       BIGINT       IDENTITY(1,1),
    machine_id       VARCHAR(10)  NOT NULL
        REFERENCES machines(machine_id),
    timestamp        DATETIME     NOT NULL,
    temperature      DECIMAL(6,2),    -- °C
    vibration        DECIMAL(6,3),    -- mm/s
    pressure         DECIMAL(6,2),    -- bar
    rpm              INT,             -- rotations per minute
    current          DECIMAL(6,2),    -- amperes
    oil_level        DECIMAL(5,1),    -- percentage
    failure_within_24h BIT          DEFAULT 0,
    ingested_at      DATETIME     DEFAULT GETDATE()
);
GO

-- Indexes for common query patterns
CREATE NONCLUSTERED INDEX ix_sensor_machine_time
    ON sensor_readings (machine_id, timestamp DESC)
    INCLUDE (temperature, vibration, current);
GO

CREATE NONCLUSTERED INDEX ix_sensor_timestamp
    ON sensor_readings (timestamp DESC);
GO

-- ── 3. Maintenance log ────────────────────────────────────────────────────────
CREATE TABLE maintenance_log (
    log_id           INT          IDENTITY(1,1) PRIMARY KEY,
    machine_id       VARCHAR(10)  NOT NULL
        REFERENCES machines(machine_id),
    maintenance_date DATE         NOT NULL,
    maintenance_type VARCHAR(50),
    technician       VARCHAR(100),
    downtime_hours   DECIMAL(4,1),
    cost_inr         DECIMAL(10,2),
    notes            VARCHAR(500),
    created_at       DATETIME     DEFAULT GETDATE()
);
GO

-- ── 4. Model predictions (output store) ───────────────────────────────────────
CREATE TABLE failure_predictions (
    pred_id          INT          IDENTITY(1,1) PRIMARY KEY,
    machine_id       VARCHAR(10)  NOT NULL,
    prediction_ts    DATETIME     NOT NULL,    -- when prediction was made
    failure_prob     DECIMAL(5,4) NOT NULL,    -- 0.0000 to 1.0000
    risk_level       VARCHAR(10)
        CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    top_feature_1    VARCHAR(50),
    top_feature_2    VARCHAR(50),
    top_feature_3    VARCHAR(50),
    action_recommended VARCHAR(200),
    model_version    VARCHAR(20),
    created_at       DATETIME     DEFAULT GETDATE()
);
GO

-- ── 5. Production orders ──────────────────────────────────────────────────────
CREATE TABLE production_orders (
    order_id         VARCHAR(20)  PRIMARY KEY,
    machine_id       VARCHAR(10)  REFERENCES machines(machine_id),
    product_code     VARCHAR(50),
    quantity         INT,
    scheduled_start  DATETIME,
    scheduled_end    DATETIME,
    actual_start     DATETIME,
    actual_end       DATETIME,
    status           VARCHAR(20)
        CHECK (status IN ('pending','in_progress','completed','delayed','cancelled')),
    priority         INT          DEFAULT 3   -- 1 = highest
);
GO
