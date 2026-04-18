"""
main.py
Predictive Maintenance & Production Optimization Pipeline
Entry point — orchestrates all pipeline stages.

Modes:
  python main.py --mode train    # train model on historical data
  python main.py --mode predict  # run inference + generate today's report
  python main.py --mode full     # train + predict (first run)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        ),
    ],
)
logger = logging.getLogger("main")


# ── Imports (after logging) ───────────────────────────────────────────────────
from config import model_cfg, report_cfg
from data.generate_data import generate_all, generate_maintenance_log
from src.feature_engineering import build_features
from src.model import train_model, predict_failure_risk, load_model
from src.optimizer import (
    compute_health_scores,
    recommend_maintenance_windows,
    optimize_production_orders,
    build_operations_summary,
)
from src.report import (
    generate_excel_report,
    generate_health_chart,
    send_email_report,
)


# ── Stage: Load data ──────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sensor readings and maintenance log.
    In production, replace CSV reads with SQLAlchemy queries.
    """
    sensor_path = "data/sensor_data.csv"
    maint_path  = "data/maintenance_log.csv"

    if not os.path.exists(sensor_path):
        logger.info("Sensor data not found — generating synthetic data...")
        generate_all(save_csv=True)
        generate_maintenance_log()

    sensor_df = pd.read_csv(sensor_path, parse_dates=["timestamp"])
    maint_df  = pd.read_csv(maint_path,  parse_dates=["maintenance_date"])
    logger.info(
        f"Loaded: {len(sensor_df):,} sensor rows | "
        f"{len(maint_df)} maintenance records"
    )
    return sensor_df, maint_df


# ── Stage: Feature engineering ────────────────────────────────────────────────

def run_feature_engineering(
    sensor_df: pd.DataFrame,
    maint_df:  pd.DataFrame,
) -> pd.DataFrame:
    logger.info("=== STAGE: Feature Engineering ===")
    t0 = time.time()
    df = build_features(sensor_df, maintenance_df=maint_df)
    logger.info(f"Feature engineering done in {time.time()-t0:.1f}s | shape: {df.shape}")
    return df


# ── Stage: Training ───────────────────────────────────────────────────────────

def run_training(features_df: pd.DataFrame) -> dict:
    logger.info("=== STAGE: Model Training ===")
    t0 = time.time()
    result = train_model(features_df)
    logger.info(
        f"Training done in {time.time()-t0:.1f}s | "
        f"Recall={result['metrics']['recall']} | "
        f"AUC={result['metrics']['roc_auc']}"
    )
    return result


# ── Stage: Inference + Optimization ──────────────────────────────────────────

def run_inference_and_optimize(
    features_df:   pd.DataFrame,
    maint_df:      pd.DataFrame,
    model_artifact: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    logger.info("=== STAGE: Inference + Optimization ===")

    # Run model predictions on latest snapshot (last reading per machine)
    latest = (
        features_df
        .sort_values("timestamp")
        .groupby("machine_id")
        .last()
        .reset_index()
    )
    predictions = predict_failure_risk(latest, model_artifact)

    # Compute health scores
    health_df = compute_health_scores(predictions)

    # Maintenance window recommendations
    maintenance_recs = recommend_maintenance_windows(health_df)

    # Simulated production orders (replace with real DB query in production)
    orders_df = _mock_production_orders(health_df["machine_id"].tolist())
    optimized_orders = optimize_production_orders(orders_df, health_df)

    # Build summary
    summary = build_operations_summary(health_df, maintenance_recs, optimized_orders)
    logger.info(f"Operations summary: {summary}")

    return health_df, maintenance_recs, optimized_orders, summary


def _mock_production_orders(machine_ids: list) -> pd.DataFrame:
    """Simulate a production orders table for demo purposes."""
    import numpy as np
    np.random.seed(7)
    records = []
    for i, mid in enumerate(machine_ids * 3):
        records.append({
            "order_id":   f"PO-{1000 + i}",
            "machine_id": mid,
            "product_code": np.random.choice(["PART-A", "PART-B", "PART-C"]),
            "quantity":   int(np.random.randint(50, 500)),
            "priority":   int(np.random.randint(1, 5)),
            "status":     "pending",
        })
    return pd.DataFrame(records)


# ── Stage: Report ─────────────────────────────────────────────────────────────

def run_reporting(
    health_df:      pd.DataFrame,
    maintenance_df: pd.DataFrame,
    orders_df:      pd.DataFrame,
    summary:        dict,
    model_metrics:  dict,
    send_email:     bool = False,
) -> None:
    logger.info("=== STAGE: Report Generation ===")

    excel_path = generate_excel_report(
        health_df, maintenance_df, orders_df, summary, model_metrics
    )
    chart_path = generate_health_chart(health_df)

    logger.info(f"Report   : {excel_path}")
    logger.info(f"Chart    : {chart_path}")

    if send_email:
        send_email_report(excel_path, chart_path, summary)


# ── CLI Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predictive Maintenance Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "full"],
        default="full",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--email", action="store_true",
        help="Send email report after generation",
    )
    args = parser.parse_args()

    logger.info(f"{'='*60}")
    logger.info(f"Predictive Maintenance Pipeline — mode: {args.mode}")
    logger.info(f"{'='*60}")

    # Load data
    sensor_df, maint_df = load_data()

    # Feature engineering
    features_df = run_feature_engineering(sensor_df, maint_df)

    model_artifact = None
    model_metrics  = {}

    if args.mode in ("train", "full"):
        result         = run_training(features_df)
        model_artifact = result
        model_metrics  = result["metrics"]

    if args.mode in ("predict", "full"):
        if model_artifact is None:
            model_artifact = load_model()
            model_metrics  = model_artifact.get("metrics", {})

        health_df, maintenance_recs, orders_df, summary = (
            run_inference_and_optimize(features_df, maint_df, model_artifact)
        )
        run_reporting(
            health_df, maintenance_recs, orders_df,
            summary, model_metrics,
            send_email=args.email,
        )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
