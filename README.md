Predictive Maintenance & Production Optimization

An end-to-end ML pipeline, built on simulated sensor data (8 machines, 90 days), that predicts machine failures 24 hours ahead and recommends maintenance windows from machine health scores.

Business Problem

Manufacturing plants suffer from unplanned machine downtime — the most expensive type of failure. Reactive maintenance (fix after failure) leads to:

Production line stoppages
Expensive emergency repairs
Missed delivery deadlines

This system shifts the approach from reactive → predictive, flagging at-risk machines before they fail.

Project Architecture
Sensor Data (simulated CSV, or SQL Server via config.py)
        │
        ▼
  Feature Engineering (src/feature_engineering.py)
   - Rolling averages, rate-of-change, anomaly flags
        │
        ▼
  ML Model (src/model.py)
   - Random Forest Classifier
   - Predicts failure within next 24 hours
   - SHAP explainability
        │
        ▼
  Production Optimizer (src/optimizer.py)
   - Scores each machine's health
   - Recommends maintenance windows
   - Re-schedules production orders
        │
        ▼
  Automated Report (src/report.py)
   - Excel report + matplotlib charts
   - Emailed daily at 6 AM via cron/Airflow
Tech Stack
Layer	Technology
Language	Python 3.10
Data Processing	pandas, numpy
Machine Learning	scikit-learn, imbalanced-learn
Explainability	SHAP
Database	SQL Server (pyodbc / SQLAlchemy)
Visualization	matplotlib, seaborn
Scheduling	Apache Airflow / cron
Reporting	openpyxl, smtplib
Results

Evaluated on a held-out test set (20%, stratified random split) of simulated data:

Metric	Score
Accuracy	0.98
Recall (failures caught)	0.76
Precision	0.88
F1	0.81
ROC-AUC	0.97

Note: data is simulated, so these scores show the pipeline works end to end, not real plant performance. Next step: time-based validation and testing on real sensor data.

Feature importance

Setup & Run
bash
# 1. Clone repo
git clone https://github.com/Datawithkaibalya/predictive-maintenance.git  # use your repo's actual URL
cd predictive-maintenance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data (or connect to real DB via config.py)
python data/generate_data.py

# 4. Run full pipeline
python main.py

# 5. View outputs in /outputs folder
Project Structure
predictive-maintenance/
├── README.md
├── requirements.txt
├── config.py                    # DB credentials, model params
├── main.py                      # Pipeline orchestrator
├── airflow_dag.py               # Daily scheduling (Airflow)
├── data/
│   └── generate_data.py         # Synthetic sensor data generator
├── sql/
│   ├── schema.sql               # Table definitions
│   └── queries.sql              # Analytical SQL queries
├── src/
│   ├── feature_engineering.py   # Feature creation
│   ├── model.py                 # ML training & inference
│   ├── optimizer.py             # Production scheduling optimizer
│   └── report.py                # Automated Excel + email report
├── outputs/                     # Generated reports, charts
└── logs/                        # Pipeline execution logs
Key Technical Decisions

Why Random Forest? Handles many engineered features well, and SHAP explainability gives clear per-machine reasons for each prediction.

Why SMOTE for imbalance? Failures are rare (~5% of records). SMOTE oversamples the minority class inside the training folds only, so there is no leakage into validation.

Why Recall over Accuracy? Missing a real failure costs far more than a false alarm, so recall and precision are tracked instead of accuracy alone.

Author

Kaibalya Prasad Rout. Personal project inspired by real plant maintenance problems. All data is simulated.

Stack: Python · SQL · scikit-learn · SHAP · Airflow
