"""
src/optimizer.py
Production Scheduling Optimizer.

Given machine health scores from the ML model, this module:
  1. Computes a composite Health Score (0–100) per machine
  2. Recommends maintenance windows (earliest safe slot)
  3. Re-ranks pending production orders by machine risk
  4. Flags orders that should be re-routed to healthy machines

Business logic:
  - Health Score < 30  → machine should not take new orders
  - Health Score 30-60 → assign only low-priority orders, schedule maintenance
  - Health Score > 60  → normal operation
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ── Health Score ──────────────────────────────────────────────────────────────

def compute_health_scores(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite health score (0 = about to fail, 100 = perfect condition).

    Score components (weighted):
      - Failure probability     40%  (inverted)
      - Vibration anomaly count 20%
      - Temperature anomaly     20%
      - Oil level               20%
    """
    df = predictions_df.copy()

    # Component 1: failure probability (inverted — lower prob = healthier)
    df["score_failure"] = (1 - df["failure_prob"]) * 100

    # Component 2: vibration (normalize anomaly count 0–6h, invert)
    if "vibration_anomaly_count_6h" in df.columns:
        max_vib = df["vibration_anomaly_count_6h"].max() or 1
        df["score_vibration"] = (
            1 - df["vibration_anomaly_count_6h"] / max_vib
        ) * 100
    else:
        df["score_vibration"] = 100.0

    # Component 3: temperature
    if "temperature_anomaly_count_6h" in df.columns:
        max_temp = df["temperature_anomaly_count_6h"].max() or 1
        df["score_temp"] = (
            1 - df["temperature_anomaly_count_6h"] / max_temp
        ) * 100
    else:
        df["score_temp"] = 100.0

    # Component 4: oil level (directly usable 0–100)
    if "oil_level" in df.columns:
        df["score_oil"] = df["oil_level"].clip(0, 100)
    else:
        df["score_oil"] = 100.0

    # Weighted composite
    df["health_score"] = (
        df["score_failure"]   * 0.40 +
        df["score_vibration"] * 0.20 +
        df["score_temp"]      * 0.20 +
        df["score_oil"]       * 0.20
    ).round(1)

    df["health_category"] = pd.cut(
        df["health_score"],
        bins=[-0.1, 30, 60, 80, 100.1],
        labels=["CRITICAL", "WARNING", "FAIR", "GOOD"],
    )
    return df


# ── Maintenance Window Recommendation ─────────────────────────────────────────

def recommend_maintenance_windows(
    health_df: pd.DataFrame,
    production_calendar: Optional[pd.DataFrame] = None,
    reference_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    For each machine below the WARNING threshold, recommend the earliest
    maintenance window that minimises production disruption.

    Simple heuristic:
      - CRITICAL → stop immediately, next available slot (within 4 hours)
      - WARNING  → next shift change (next 8-16 hours)
      - FAIR     → schedule within 48 hours
    """
    if reference_time is None:
        reference_time = datetime.now()

    recs = []
    for _, row in health_df.iterrows():
        cat = str(row["health_category"])
        if cat == "CRITICAL":
            window_start = reference_time + timedelta(hours=0)
            urgency      = "IMMEDIATE — stop production"
            action       = f"Emergency maintenance: check {row.get('top_feature_1','sensors')}"
        elif cat == "WARNING":
            window_start = reference_time + timedelta(hours=8)
            urgency      = "URGENT — next shift change"
            action       = f"Schedule maintenance: inspect {row.get('top_feature_1','sensors')}"
        elif cat == "FAIR":
            window_start = reference_time + timedelta(hours=48)
            urgency      = "PLANNED — within 48h"
            action       = f"Preventive check: {row.get('top_feature_1','sensors')}"
        else:
            window_start = None
            urgency      = "None"
            action       = "No action needed"

        recs.append({
            "machine_id":        row["machine_id"],
            "health_score":      row["health_score"],
            "health_category":   cat,
            "failure_prob":      row.get("failure_prob", None),
            "maintenance_urgency": urgency,
            "recommended_window":  window_start.strftime("%Y-%m-%d %H:%M") if window_start else "—",
            "recommended_action":  action,
        })

    result = pd.DataFrame(recs).sort_values("health_score")
    logger.info(
        f"Maintenance recommendations: "
        f"CRITICAL={( result['health_category']=='CRITICAL').sum()} | "
        f"WARNING={( result['health_category']=='WARNING').sum()} | "
        f"FAIR={( result['health_category']=='FAIR').sum()}"
    )
    return result


# ── Production Order Re-scheduling ────────────────────────────────────────────

def optimize_production_orders(
    orders_df: pd.DataFrame,
    health_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Re-score production orders based on assigned machine health.

    Logic:
      - If machine health_score < 30 → flag order as AT_RISK, suggest re-route
      - If machine health_score 30-60 → flag as MONITOR, reduce order load
      - Else → OK
    """
    health_lookup = health_df.set_index("machine_id")["health_score"].to_dict()
    cat_lookup    = health_df.set_index("machine_id")["health_category"].to_dict()

    df = orders_df.copy()
    df["machine_health_score"]    = df["machine_id"].map(health_lookup).fillna(100)
    df["machine_health_category"] = df["machine_id"].map(cat_lookup).fillna("GOOD")

    def _order_flag(row):
        cat = row["machine_health_category"]
        if cat == "CRITICAL":
            return "RE-ROUTE NOW"
        elif cat == "WARNING":
            return "MONITOR CLOSELY"
        else:
            return "OK"

    df["order_flag"]   = df.apply(_order_flag, axis=1)
    df["order_risk"]   = df["machine_health_score"].apply(
        lambda s: "HIGH" if s < 30 else ("MEDIUM" if s < 60 else "LOW")
    )

    # Sort: flagged orders first, then by priority
    flag_order = {"RE-ROUTE NOW": 0, "MONITOR CLOSELY": 1, "OK": 2}
    df["_flag_sort"] = df["order_flag"].map(flag_order)
    df = df.sort_values(["_flag_sort", "priority"]).drop(columns=["_flag_sort"])

    flagged = (df["order_flag"] != "OK").sum()
    logger.info(
        f"Production orders optimised: {len(df)} total | "
        f"{flagged} flagged for action"
    )
    return df


# ── Summary Table ─────────────────────────────────────────────────────────────

def build_operations_summary(
    health_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
    orders_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Build a summary dict for reporting / dashboard."""
    total_machines   = len(health_df)
    critical_count   = (health_df["health_category"] == "CRITICAL").sum()
    warning_count    = (health_df["health_category"] == "WARNING").sum()
    avg_health       = round(health_df["health_score"].mean(), 1)
    at_risk_orders   = (
        (orders_df["order_flag"] != "OK").sum()
        if orders_df is not None else 0
    )

    return {
        "total_machines":  int(total_machines),
        "critical_count":  int(critical_count),
        "warning_count":   int(warning_count),
        "healthy_count":   int(total_machines - critical_count - warning_count),
        "avg_health_score": float(avg_health),
        "at_risk_orders":  int(at_risk_orders),
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
