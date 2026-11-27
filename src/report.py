"""
src/report.py
Automated reporting module.

Generates:
  1. Excel report with 4 sheets:
       - Executive Summary
       - Machine Health Scores
       - Maintenance Recommendations
       - Production Order Status
  2. PNG charts: health score bar chart, failure probability heatmap
  3. (Optional) Email delivery via SMTP
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, Reference

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import report_cfg

logger = logging.getLogger(__name__)

# Color palette for Excel
COLORS = {
    "CRITICAL": "E24B4A",
    "WARNING":  "EF9F27",
    "FAIR":     "378ADD",
    "GOOD":     "639922",
    "header_bg": "042C53",
    "header_fg": "FFFFFF",
}


# ── Excel helpers ─────────────────────────────────────────────────────────────

def _style_header_row(ws, row: int, n_cols: int) -> None:
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=col)
        cell.font      = Font(bold=True, color=COLORS["header_fg"], size=11)
        cell.fill      = PatternFill("solid", fgColor=COLORS["header_bg"])
        cell.alignment = Alignment(horizontal="center", vertical="center")


def _color_risk_cells(ws, df: pd.DataFrame,
                       cat_col: str, start_row: int) -> None:
    """Color rows based on health/risk category."""
    col_idx = list(df.columns).index(cat_col) + 1
    for i, val in enumerate(df[cat_col]):
        val_str = str(val)
        color   = COLORS.get(val_str, "FFFFFF")
        ws.cell(row=start_row + i, column=col_idx).fill = (
            PatternFill("solid", fgColor=color)
        )
        ws.cell(row=start_row + i, column=col_idx).font = Font(
            bold=True, color="FFFFFF"
        )


def _write_df_to_sheet(ws, df: pd.DataFrame, start_row: int = 2) -> None:
    """Write DataFrame to openpyxl worksheet."""
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True)):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=start_row + r_idx - 1, column=c_idx, value=value)


# ── Main report builder ───────────────────────────────────────────────────────

def generate_excel_report(
    health_df:      pd.DataFrame,
    maintenance_df: pd.DataFrame,
    orders_df:      Optional[pd.DataFrame],
    summary:        dict,
    model_metrics:  dict,
) -> str:
    """
    Build the full Excel workbook.
    Returns the file path of the saved report.
    """
    os.makedirs(report_cfg.output_dir, exist_ok=True)
    date_str  = datetime.now().strftime("%Y%m%d_%H%M")
    file_path = os.path.join(
        report_cfg.output_dir, f"maintenance_report_{date_str}.xlsx"
    )

    wb = openpyxl.Workbook()

    # ── Sheet 1: Executive Summary ────────────────────────────────────────────
    ws_exec = wb.active
    ws_exec.title = "Executive Summary"

    ws_exec["B2"] = "Predictive Maintenance Daily Report"
    ws_exec["B2"].font = Font(bold=True, size=16, color=COLORS["header_bg"])
    ws_exec["B3"] = f"Generated: {summary['generated_at']}"
    ws_exec["B3"].font = Font(color="888888", size=10)

    summary_data = [
        ("Total machines monitored", summary["total_machines"]),
        ("CRITICAL machines",        summary["critical_count"]),
        ("WARNING machines",         summary["warning_count"]),
        ("Healthy machines",         summary["healthy_count"]),
        ("Average health score",     summary["avg_health_score"]),
        ("At-risk production orders",summary["at_risk_orders"]),
        ("Model recall",             f"{model_metrics.get('recall', 'N/A')*100:.1f}%"),
        ("Model AUC-ROC",            model_metrics.get("roc_auc", "N/A")),
    ]
    for i, (label, value) in enumerate(summary_data, start=5):
        ws_exec[f"B{i}"] = label
        ws_exec[f"C{i}"] = value
        ws_exec[f"B{i}"].font = Font(size=11)
        ws_exec[f"C{i}"].font = Font(size=11, bold=True)

    # ── Sheet 2: Machine Health ────────────────────────────────────────────────
    ws_health = wb.create_sheet("Machine Health Scores")
    cols_health = [
        "machine_id", "health_score", "health_category",
        "failure_prob", "top_feature_1", "top_feature_2",
    ]
    cols_present = [c for c in cols_health if c in health_df.columns]
    _write_df_to_sheet(ws_health, health_df[cols_present])
    _style_header_row(ws_health, 2, len(cols_present))
    _color_risk_cells(ws_health, health_df[cols_present], "health_category", 3)

    # Auto-width columns
    for col in ws_health.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws_health.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

    # ── Sheet 3: Maintenance Recommendations ─────────────────────────────────
    ws_maint = wb.create_sheet("Maintenance Recommendations")
    _write_df_to_sheet(ws_maint, maintenance_df)
    _style_header_row(ws_maint, 2, len(maintenance_df.columns))

    # ── Sheet 4: Production Orders ────────────────────────────────────────────
    if orders_df is not None and not orders_df.empty:
        ws_orders = wb.create_sheet("Production Orders")
        _write_df_to_sheet(ws_orders, orders_df)
        _style_header_row(ws_orders, 2, len(orders_df.columns))

    wb.save(file_path)
    logger.info(f"Excel report saved: {file_path}")
    return file_path


# ── Charts ────────────────────────────────────────────────────────────────────

def generate_health_chart(health_df: pd.DataFrame) -> str:
    """Bar chart: machine health scores with color coding."""
    fig, ax = plt.subplots(figsize=(12, 5))

    color_map = {
        "CRITICAL": "#E24B4A",
        "WARNING":  "#EF9F27",
        "FAIR":     "#378ADD",
        "GOOD":     "#639922",
    }
    colors = [
        color_map.get(str(c), "#888780")
        for c in health_df["health_category"]
    ]

    bars = ax.bar(
        health_df["machine_id"],
        health_df["health_score"],
        color=colors, edgecolor="none", width=0.6,
    )
    ax.axhline(30, color="#E24B4A", linestyle="--", linewidth=1, label="CRITICAL threshold")
    ax.axhline(60, color="#EF9F27", linestyle="--", linewidth=1, label="WARNING threshold")

    for bar, score in zip(bars, health_df["health_score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{score:.0f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Health Score (0–100)")
    ax.set_title("Machine Health Scores — Daily Report", fontweight="bold")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(report_cfg.output_dir, "health_scores.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Health chart saved: {path}")
    return path


# ── Email delivery ─────────────────────────────────────────────────────────────

def send_email_report(
    report_path:    str,
    chart_path:     str,
    summary:        dict,
    recipients:     list = None,
) -> None:
    """Send report via SMTP. Configure credentials in config.py."""
    if recipients is None:
        recipients = report_cfg.email_recipients

    msg = MIMEMultipart()
    msg["Subject"] = (
        f"[Predictive Maintenance] Daily Report — "
        f"{summary['critical_count']} CRITICAL, "
        f"{summary['warning_count']} WARNING — "
        f"{summary['generated_at']}"
    )
    msg["From"] = report_cfg.sender_email
    msg["To"]   = ", ".join(recipients)

    body = f"""
    Daily Predictive Maintenance Report

    Summary:
      - Machines monitored : {summary['total_machines']}
      - CRITICAL            : {summary['critical_count']}
      - WARNING             : {summary['warning_count']}
      - Healthy             : {summary['healthy_count']}
      - Avg health score    : {summary['avg_health_score']}
      - At-risk orders      : {summary['at_risk_orders']}

    See attached Excel report and health score chart for full details.
    """
    msg.attach(MIMEText(body, "plain"))

    for fpath in [report_path, chart_path]:
        if fpath and os.path.exists(fpath):
            with open(fpath, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={os.path.basename(fpath)}",
                )
                msg.attach(part)

    try:
        with smtplib.SMTP(report_cfg.smtp_host, report_cfg.smtp_port) as server:
            server.starttls()
            server.sendmail(report_cfg.sender_email, recipients, msg.as_string())
        logger.info(f"Report emailed to {recipients}")
    except Exception as e:
        logger.error(f"Email failed: {e}. Report available at {report_path}")
