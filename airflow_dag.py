"""
airflow_dag.py
Apache Airflow DAG — schedules the maintenance pipeline to run daily at 6 AM.

Workflow:
  extract_data → feature_engineering → run_model → optimize → generate_report → send_email

Retry logic: 3 retries, 5-minute delay, email alert on failure.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator


# ── Default args ──────────────────────────────────────────────────────────────

default_args = {
    "owner":             "data-team",
    "depends_on_past":   False,
    "start_date":        datetime(2024, 1, 1),
    "email":             ["data-team@company.com"],
    "email_on_failure":  True,
    "email_on_retry":    False,
    "retries":           3,
    "retry_delay":       timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ── Task functions ─────────────────────────────────────────────────────────────

def task_load_data(**context):
    from data.generate_data import generate_all
    df = generate_all(save_csv=True)
    context["ti"].xcom_push(key="row_count", value=len(df))
    return len(df)


def task_feature_engineering(**context):
    import pandas as pd
    from src.feature_engineering import build_features
    sensor_df = pd.read_csv("data/sensor_data.csv", parse_dates=["timestamp"])
    maint_df  = pd.read_csv("data/maintenance_log.csv", parse_dates=["maintenance_date"])
    features  = build_features(sensor_df, maint_df)
    features.to_parquet("data/features.parquet", index=False)
    return features.shape


def task_train_model(**context):
    import pandas as pd
    from src.model import train_model
    features = pd.read_parquet("data/features.parquet")
    result   = train_model(features)
    return result["metrics"]


def task_inference_optimize(**context):
    import pandas as pd
    from src.model import load_model, predict_failure_risk
    from src.optimizer import (
        compute_health_scores,
        recommend_maintenance_windows,
        build_operations_summary,
    )
    features      = pd.read_parquet("data/features.parquet")
    model_artifact = load_model()
    latest = (
        features.sort_values("timestamp")
                .groupby("machine_id").last().reset_index()
    )
    preds      = predict_failure_risk(latest, model_artifact)
    health_df  = compute_health_scores(preds)
    maint_recs = recommend_maintenance_windows(health_df)
    health_df.to_parquet("data/health_scores.parquet",  index=False)
    maint_recs.to_parquet("data/maintenance_recs.parquet", index=False)


def task_generate_report(**context):
    import pandas as pd
    from src.model import load_model
    from src.report import generate_excel_report, generate_health_chart
    from src.optimizer import build_operations_summary

    health_df  = pd.read_parquet("data/health_scores.parquet")
    maint_recs = pd.read_parquet("data/maintenance_recs.parquet")
    artifact   = load_model()
    summary    = build_operations_summary(health_df, maint_recs)

    excel_path = generate_excel_report(
        health_df, maint_recs, None, summary, artifact["metrics"]
    )
    chart_path = generate_health_chart(health_df)
    context["ti"].xcom_push(key="excel_path", value=excel_path)
    context["ti"].xcom_push(key="chart_path",  value=chart_path)


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="predictive_maintenance_pipeline",
    default_args=default_args,
    description="Daily predictive maintenance and production optimization pipeline",
    schedule_interval="0 6 * * *",    # every day at 6 AM
    catchup=False,
    tags=["maintenance", "ml", "production"],
) as dag:

    load_data = PythonOperator(
        task_id="load_data",
        python_callable=task_load_data,
    )

    feature_eng = PythonOperator(
        task_id="feature_engineering",
        python_callable=task_feature_engineering,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    inference = PythonOperator(
        task_id="inference_and_optimize",
        python_callable=task_inference_optimize,
    )

    report = PythonOperator(
        task_id="generate_report",
        python_callable=task_generate_report,
    )

    notify = EmailOperator(
        task_id="send_email",
        to=["maintenance-lead@company.com", "production-manager@company.com"],
        subject="[Predictive Maintenance] Daily Report Ready — {{ ds }}",
        html_content="""
        <h2>Daily Predictive Maintenance Report</h2>
        <p>The pipeline ran successfully for {{ ds }}.</p>
        <p>Please check the attached report in the shared drive.</p>
        """,
    )

    # Pipeline order
    load_data >> feature_eng >> train >> inference >> report >> notify
