"""
config.py
Central configuration for the Predictive Maintenance Pipeline.
All environment-specific values live here — never hardcode in src/ files.
"""

import os
from dataclasses import dataclass, field
from typing import List


# ── Database ──────────────────────────────────────────────────────────────────

@dataclass
class DBConfig:
    server:   str = os.getenv("DB_SERVER",   "localhost")
    database: str = os.getenv("DB_NAME",     "manufacturing_db")
    username: str = os.getenv("DB_USER",     "sa")
    password: str = os.getenv("DB_PASSWORD", "yourpassword")
    driver:   str = "ODBC Driver 17 for SQL Server"

    @property
    def connection_string(self) -> str:
        return (
            f"mssql+pyodbc://{self.username}:{self.password}"
            f"@{self.server}/{self.database}"
            f"?driver={self.driver.replace(' ', '+')}"
        )


# ── Machine / Sensor ──────────────────────────────────────────────────────────

@dataclass
class SensorConfig:
    machines: List[str] = field(default_factory=lambda: [
        "M001", "M002", "M003", "M004", "M005",
        "M006", "M007", "M008",
    ])
    sensor_cols: List[str] = field(default_factory=lambda: [
        "temperature",   # °C
        "vibration",     # mm/s
        "pressure",      # bar
        "rpm",           # rotations per minute
        "current",       # amperes
        "oil_level",     # percentage
    ])
    # Normal operating ranges — used for anomaly flagging
    thresholds: dict = field(default_factory=lambda: {
        "temperature": {"min": 60,  "max": 95},
        "vibration":   {"min": 0.1, "max": 4.5},
        "pressure":    {"min": 2.0, "max": 8.0},
        "rpm":         {"min": 800, "max": 3500},
        "current":     {"min": 5,   "max": 25},
        "oil_level":   {"min": 30,  "max": 100},
    })


# ── Model ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    prediction_horizon_hours: int   = 24      # predict failure in next N hours
    failure_label_col:        str   = "failure_within_24h"
    test_size:                float = 0.2
    cv_folds:                 int   = 5
    random_state:             int   = 42
    n_estimators:             int   = 200
    max_depth:                int   = 12
    smote_sampling_strategy:  float = 0.4    # minority:majority ratio after SMOTE
    model_path:               str   = "outputs/churn_model.pkl"
    primary_metric:           str   = "recall"   # recall > accuracy for maintenance


# ── Rolling Feature Windows ───────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    rolling_windows: List[int] = field(default_factory=lambda: [1, 6, 24])  # hours
    lag_periods:     List[int] = field(default_factory=lambda: [1, 3, 6])
    rate_of_change_cols: List[str] = field(default_factory=lambda: [
        "temperature", "vibration", "current"
    ])


# ── Reporting ─────────────────────────────────────────────────────────────────

@dataclass
class ReportConfig:
    output_dir:       str       = "outputs"
    top_n_at_risk:    int       = 20
    email_recipients: List[str] = field(default_factory=lambda: [
        "maintenance-lead@company.com",
        "production-manager@company.com",
    ])
    smtp_host:        str = "smtp.company.com"
    smtp_port:        int = 587
    sender_email:     str = "data-pipeline@company.com"


# ── Singleton instances ───────────────────────────────────────────────────────

db_cfg      = DBConfig()
sensor_cfg  = SensorConfig()
model_cfg   = ModelConfig()
feature_cfg = FeatureConfig()
report_cfg  = ReportConfig()
