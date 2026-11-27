"""
src/model.py
Trains, evaluates, and runs inference with the failure prediction model.

Model: Random Forest Classifier
Target: failure_within_24h (binary)
Key design choices:
  - SMOTE for class imbalance (failures are rare ~6%)
  - Stratified k-fold cross-validation
  - SHAP for per-machine explainability
  - Recall as primary metric (missing a failure >> false alarm)
"""

import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, recall_score, precision_score, f1_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import shap
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import model_cfg, report_cfg
from src.feature_engineering import get_feature_columns

logger = logging.getLogger(__name__)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> dict:
    """
    Full training pipeline:
      1. Split features / label
      2. Stratified train-test split
      3. SMOTE in pipeline (on training fold only — no data leakage)
      4. Cross-validate with 5 folds
      5. Final fit on full training set
      6. Evaluate on hold-out test set
      7. Save model artifact

    Returns dict with model, scaler, feature columns, and metrics.
    """
    logger.info("Starting model training...")

    feature_cols = get_feature_columns(df)
    label_col    = model_cfg.failure_label_col

    X = df[feature_cols].values
    y = df[label_col].values

    logger.info(
        f"Dataset: {len(y):,} rows | "
        f"failure rate: {y.mean():.2%} | "
        f"features: {len(feature_cols)}"
    )

    # Stratified split — preserves class ratio in test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_cfg.test_size,
        stratify=y,
        random_state=model_cfg.random_state,
    )

    # Pipeline: SMOTE → RandomForest
    # SMOTE is inside the pipeline so it only runs on training folds
    pipeline = ImbPipeline([
        ("smote", SMOTE(
            sampling_strategy=model_cfg.smote_sampling_strategy,
            random_state=model_cfg.random_state,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=model_cfg.n_estimators,
            max_depth=model_cfg.max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=model_cfg.random_state,
        )),
    ])

    # Cross-validation
    cv = StratifiedKFold(
        n_splits=model_cfg.cv_folds,
        shuffle=True,
        random_state=model_cfg.random_state,
    )
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=["recall", "precision", "f1", "roc_auc"],
        return_train_score=False,
    )
    logger.info("Cross-validation results (mean ± std):")
    for metric, vals in cv_results.items():
        if metric.startswith("test_"):
            name = metric.replace("test_", "")
            logger.info(f"  {name:12s}: {vals.mean():.4f} ± {vals.std():.4f}")

    # Final fit on full training data
    pipeline.fit(X_train, y_train)

    # Test set evaluation
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "recall":    round(recall_score(y_test, y_pred),    4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred),        4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob),   4),
        "cv_recall": round(cv_results["test_recall"].mean(), 4),
    }
    logger.info(f"\nTest set metrics: {metrics}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # Save model
    os.makedirs(report_cfg.output_dir, exist_ok=True)
    joblib.dump({
        "pipeline":     pipeline,
        "feature_cols": feature_cols,
        "metrics":      metrics,
    }, model_cfg.model_path)
    logger.info(f"Model saved to {model_cfg.model_path}")

    # Plot feature importance
    _plot_feature_importance(
        pipeline.named_steps["clf"],
        feature_cols,
        top_n=20,
    )

    return {
        "pipeline":     pipeline,
        "feature_cols": feature_cols,
        "metrics":      metrics,
        "X_test":       X_test,
        "y_test":       y_test,
        "y_prob":       y_prob,
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model() -> dict:
    """Load a previously saved model artifact."""
    if not os.path.exists(model_cfg.model_path):
        raise FileNotFoundError(
            f"Model not found at {model_cfg.model_path}. "
            "Run train_model() first."
        )
    return joblib.load(model_cfg.model_path)


def predict_failure_risk(df: pd.DataFrame, model_artifact: dict = None) -> pd.DataFrame:
    """
    Runs inference on feature-engineered DataFrame.
    Returns original DataFrame with added columns:
      - failure_prob   : predicted probability of failure in next 24h
      - risk_level     : LOW / MEDIUM / HIGH / CRITICAL
      - top_features   : top 3 SHAP features driving the prediction
    """
    if model_artifact is None:
        model_artifact = load_model()

    pipeline     = model_artifact["pipeline"]
    feature_cols = model_artifact["feature_cols"]

    # Guard: ensure all required features are present
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features in input: {missing}")

    X = df[feature_cols].values
    probs = pipeline.predict_proba(X)[:, 1]

    df = df.copy()
    df["failure_prob"] = np.round(probs, 4)
    df["risk_level"]   = pd.cut(
        df["failure_prob"],
        bins=[-0.001, 0.30, 0.55, 0.75, 1.001],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )

    # SHAP explanations for top features per row
    clf       = pipeline.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)[:, :, 1]  # class=1 (failure)

    top_features = []
    for i in range(len(df)):
        order = np.argsort(np.abs(shap_vals[i]))[::-1][:3]
        top_features.append([feature_cols[j] for j in order])

    df["top_feature_1"] = [f[0] for f in top_features]
    df["top_feature_2"] = [f[1] for f in top_features]
    df["top_feature_3"] = [f[2] for f in top_features]

    logger.info(
        f"Inference done: {len(df):,} rows | "
        f"HIGH+CRITICAL: {(df['risk_level'].isin(['HIGH','CRITICAL'])).sum()}"
    )
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_feature_importance(clf: RandomForestClassifier,
                              feature_cols: list,
                              top_n: int = 20) -> None:
    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index, top.values, color="#378ADD")
    ax.set_xlabel("Feature importance (mean decrease in impurity)")
    ax.set_title(f"Top {top_n} features — Failure Prediction Model")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(report_cfg.output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance chart saved: {path}")
