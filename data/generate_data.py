"""
data/generate_data.py
Generates realistic synthetic sensor data for 8 machines over 90 days.
Saves to CSV (offline mode) and optionally inserts into SQL Server.

Usage:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import sensor_cfg

np.random.seed(42)


# ── Constants ─────────────────────────────────────────────────────────────────

MACHINES    = sensor_cfg.machines
DAYS        = 90
FREQ        = "1h"           # one reading per hour
FAILURE_RATE = 0.06          # ~6% of readings precede a failure


# ── Helper: normal operating signal with noise ────────────────────────────────

def _normal_signal(n: int, mean: float, std: float,
                   lo: float, hi: float) -> np.ndarray:
    """Gaussian signal clipped to operating range."""
    return np.clip(np.random.normal(mean, std, n), lo, hi)


# ── Helper: degrade signal as machine approaches failure ─────────────────────

def _degrade(series: np.ndarray, failure_indices: np.ndarray,
             window: int = 24, factor: float = 1.4) -> np.ndarray:
    """
    Amplify sensor readings in the 'window' hours before a failure event.
    Simulates a real machine degrading before it breaks.
    """
    out = series.copy()
    for idx in failure_indices:
        start = max(0, idx - window)
        # Linear ramp: starts at 1x, peaks at factor just before failure
        ramp = np.linspace(1.0, factor, idx - start)
        out[start:idx] = np.clip(out[start:idx] * ramp, None, series.max() * 1.5)
    return out


# ── Generate data for one machine ─────────────────────────────────────────────

def generate_machine_data(machine_id: str) -> pd.DataFrame:
    timestamps = pd.date_range(
        start=datetime(2024, 1, 1),
        periods=DAYS * 24,
        freq=FREQ,
    )
    n = len(timestamps)

    # Base sensor readings (normal operation)
    temp      = _normal_signal(n, mean=75,   std=5,   lo=60,  hi=120)
    vibration = _normal_signal(n, mean=2.0,  std=0.5, lo=0.1, hi=8.0)
    pressure  = _normal_signal(n, mean=5.0,  std=0.8, lo=2.0, hi=10.0)
    rpm       = _normal_signal(n, mean=2000, std=200, lo=800, hi=4000)
    current   = _normal_signal(n, mean=15,   std=3,   lo=5,   hi=30)
    oil_level = np.clip(
        100 - np.cumsum(np.random.exponential(0.03, n)), 20, 100
    )  # oil gradually depletes

    # Randomly assign failure events
    failure_mask = np.zeros(n, dtype=int)
    failure_count = max(1, int(n * FAILURE_RATE / 24))
    failure_indices = np.random.choice(
        range(48, n - 24), size=failure_count, replace=False
    )
    for fi in failure_indices:
        # Mark the 24 hours before failure as label=1
        failure_mask[max(0, fi - 24): fi] = 1

    # Degrade signals before failures
    temp      = _degrade(temp,      failure_indices, factor=1.35)
    vibration = _degrade(vibration, failure_indices, factor=1.60)
    current   = _degrade(current,   failure_indices, factor=1.25)

    # Add maintenance events (oil change resets oil_level)
    for i in range(0, n, 24 * 15):   # every 15 days
        oil_level[i:] = np.clip(
            oil_level[i:] + np.random.uniform(40, 60), 0, 100
        )

    df = pd.DataFrame({
        "timestamp":          timestamps,
        "machine_id":         machine_id,
        "temperature":        np.round(temp,      2),
        "vibration":          np.round(vibration, 3),
        "pressure":           np.round(pressure,  2),
        "rpm":                np.round(rpm,       0).astype(int),
        "current":            np.round(current,   2),
        "oil_level":          np.round(oil_level, 1),
        "failure_within_24h": failure_mask,
    })
    return df


# ── Generate for all machines ─────────────────────────────────────────────────

def generate_all(save_csv: bool = True) -> pd.DataFrame:
    print(f"Generating sensor data for {len(MACHINES)} machines over {DAYS} days...")
    frames = []
    for mid in MACHINES:
        df = generate_machine_data(mid)
        frames.append(df)
        print(f"  [{mid}] {len(df):,} rows | failures: {df['failure_within_24h'].sum()}")

    full = pd.concat(frames, ignore_index=True)
    full.sort_values(["machine_id", "timestamp"], inplace=True)
    full.reset_index(drop=True, inplace=True)

    if save_csv:
        os.makedirs("data", exist_ok=True)
        path = "data/sensor_data.csv"
        full.to_csv(path, index=False)
        print(f"\nSaved {len(full):,} rows to {path}")

    print(f"\nTotal rows     : {len(full):,}")
    print(f"Failure ratio  : {full['failure_within_24h'].mean():.2%}")
    print(f"Date range     : {full['timestamp'].min()} → {full['timestamp'].max()}")
    return full


# ── Also generate a maintenance log ──────────────────────────────────────────

def generate_maintenance_log() -> pd.DataFrame:
    records = []
    for mid in MACHINES:
        for day in range(0, DAYS, np.random.randint(12, 20)):
            records.append({
                "machine_id":        mid,
                "maintenance_date":  datetime(2024, 1, 1) + timedelta(days=day),
                "maintenance_type":  np.random.choice(
                    ["oil_change", "belt_replacement", "sensor_calibration",
                     "bearing_check", "full_inspection"],
                    p=[0.35, 0.25, 0.20, 0.15, 0.05],
                ),
                "technician":        np.random.choice(
                    ["Rajan", "Sunita", "Arjun", "Priya", "Deepak"]
                ),
                "downtime_hours":    round(np.random.uniform(0.5, 6), 1),
                "cost_inr":          round(np.random.uniform(2000, 25000), 0),
            })
    df = pd.DataFrame(records)
    df.to_csv("data/maintenance_log.csv", index=False)
    print(f"Maintenance log saved: {len(df)} records")
    return df


if __name__ == "__main__":
    generate_all(save_csv=True)
    generate_maintenance_log()
