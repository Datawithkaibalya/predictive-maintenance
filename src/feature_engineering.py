"""
src/feature_engineering.py
Creates ML-ready features from raw sensor readings.

Key feature groups:
  1. Rolling statistics  (mean, std, max over 1h / 6h / 24h windows)
  2. Rate of change      (delta from previous reading)
  3. Anomaly flags       (binary: outside normal range)
  4. Cross-sensor ratios (temperature/current, vibration/rpm)
  5. Time features       (hour of day, day of week — shift patterns matter)
  6. Lag features        (sensor value N hours ago)
"""

import logging
import numpy as np
import pandas as pd
from typing import List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import sensor_cfg, feature_cfg

logger = logging.getLogger(__name__)


# ── Rolling statistics ────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame,
                         cols: List[str],
                         windows: List[int]) -> pd.DataFrame:
    """
    Add rolling mean, std, and max for each sensor column
    across multiple time windows (in hours, assuming 1 reading/hour).
    """
    for col in cols:
        for w in windows:
            grp = df.groupby("machine_id")[col]
            df[f"{col}_mean_{w}h"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{col}_std_{w}h"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
            df[f"{col}_max_{w}h"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).max()
            )
    logger.info(f"Rolling features added: {len(cols)} cols × {len(windows)} windows × 3 stats")
    return df


# ── Rate of change ────────────────────────────────────────────────────────────

def add_rate_of_change(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Delta = current value - value N periods ago.
    Captures sensor trends (rising temperature = bad sign).
    """
    for col in cols:
        for lag in feature_cfg.lag_periods:
            df[f"{col}_delta_{lag}h"] = (
                df.groupby("machine_id")[col]
                  .transform(lambda x: x.diff(lag))
                  .fillna(0)
            )
    logger.info(f"Rate-of-change features added for: {cols}")
    return df


# ── Anomaly flags ─────────────────────────────────────────────────────────────

def add_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flags: 1 if sensor reading is outside the normal operating range.
    Also adds rolling count of anomalies in last 6 hours (trend signal).
    """
    thresholds = sensor_cfg.thresholds
    for col, bounds in thresholds.items():
        if col not in df.columns:
            continue
        flag_col = f"{col}_anomaly"
        df[flag_col] = (
            (df[col] < bounds["min"]) | (df[col] > bounds["max"])
        ).astype(int)

        # Count anomalies in last 6 hours per machine
        df[f"{col}_anomaly_count_6h"] = (
            df.groupby("machine_id")[flag_col]
              .transform(lambda x: x.rolling(6, min_periods=1).sum())
        )

    # Composite: total anomaly count across all sensors (single risk indicator)
    anomaly_cols = [c for c in df.columns if c.endswith("_anomaly") and "count" not in c]
    df["total_anomaly_count"] = df[anomaly_cols].sum(axis=1)
    logger.info(f"Anomaly flags added for {len(thresholds)} sensors")
    return df


# ── Cross-sensor ratios ───────────────────────────────────────────────────────

def add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratios between sensors carry physical meaning:
      - temp/current: high temp with low current → cooling failure
      - vibration/rpm: normalised vibration intensity
    """
    df["temp_per_current"]  = df["temperature"] / df["current"].replace(0, np.nan)
    df["vib_per_rpm"]       = df["vibration"]   / df["rpm"].replace(0, np.nan) * 1000
    df["pressure_per_rpm"]  = df["pressure"]    / df["rpm"].replace(0, np.nan) * 1000
    df[["temp_per_current", "vib_per_rpm", "pressure_per_rpm"]] = (
        df[["temp_per_current", "vib_per_rpm", "pressure_per_rpm"]].fillna(0)
    )
    return df


# ── Time features ─────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift patterns heavily influence machine load.
    Night shifts often have less maintenance, so failures cluster there.
    """
    df["hour"]      = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_night"]  = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["is_weekend"]= (df["dayofweek"] >= 5).astype(int)
    return df


# ── Time since last maintenance ───────────────────────────────────────────────

def add_maintenance_features(df: pd.DataFrame,
                              maintenance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge maintenance log to compute hours since last service per machine.
    Machines with longer gaps are more likely to fail.
    """
    if maintenance_df is None or maintenance_df.empty:
        df["hours_since_maintenance"] = -1
        return df

    maint = (
        maintenance_df
        .sort_values("maintenance_date")
        .groupby("machine_id")["maintenance_date"]
        .last()
        .reset_index()
        .rename(columns={"maintenance_date": "last_maint"})
    )
    df = df.merge(maint, on="machine_id", how="left")
    df["hours_since_maintenance"] = (
        (df["timestamp"] - pd.to_datetime(df["last_maint"]))
        .dt.total_seconds() / 3600
    ).fillna(-1)
    df.drop(columns=["last_maint"], inplace=True)
    return df


# ── Master feature builder ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   maintenance_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Runs all feature engineering steps in order.
    Returns a DataFrame ready for model training or inference.
    """
    logger.info(f"Starting feature engineering on {len(df):,} rows...")
    df = df.sort_values(["machine_id", "timestamp"]).copy()

    sensor_cols = [c for c in sensor_cfg.sensor_cols if c in df.columns]

    df = add_rolling_features(df, sensor_cols, feature_cfg.rolling_windows)
    df = add_rate_of_change(df,  feature_cfg.rate_of_change_cols)
    df = add_anomaly_flags(df)
    df = add_cross_sensor_features(df)
    df = add_time_features(df)
    df = add_maintenance_features(df, maintenance_df)

    # Drop rows with NaN (first few rows per machine have no rolling history)
    before = len(df)
    df.dropna(inplace=True)
    logger.info(f"Dropped {before - len(df)} rows with NaN after feature engineering")
    logger.info(f"Final shape: {df.shape} | features: {df.shape[1]}")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return only the feature columns (exclude IDs, labels, timestamps)."""
    exclude = {"machine_id", "timestamp", "failure_within_24h", "ingested_at"}
    return [c for c in df.columns if c not in exclude]
