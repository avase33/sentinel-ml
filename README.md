<div align="center">

```
 ___            _   _            _       __  __ _
/ __| ___ _ _ | |_(_)_ _  ___ | |     |  \/  | |
\__ \/ -_) ' \|  _| | ' \/ -_)| |__   | |\/| | |
|___/\___|_||_|\__|_|_||_\___||____|  |_|  |_|_|
```

### **ML Model Monitoring and Anomaly Detection**

*Keep your deployed models honest. Catch drift before your users do.*

<br/>

[![CI](https://github.com/avase33/sentinel-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/avase33/sentinel-ml/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-Proprietary-red)

<br/>

> **Sentinel-ML** is a production ML monitoring system that watches your deployed models for data drift, performance degradation, and anomalous predictions -- then fires alerts before silent failures become business problems.

</div>

---

## The Problem

ML models decay. The data distribution shifts. Users behave differently than training examples. A model that was 94% accurate last quarter is silently wrong today -- and without monitoring, no one knows.

Sentinel-ML makes model health visible, measurable, and alertable.

---

## Feature Highlights

### Data Drift Detection

- Statistical tests (KS-test, Chi-squared, PSI) on input feature distributions
- Baseline any dataset as the reference distribution
- Continuous comparison of incoming production data against baseline
- Per-feature drift scores with configurable thresholds

### Model Performance Monitoring

- Track accuracy, precision, recall, F1, RMSE, MAE over time
- Rolling window performance: last 24h, 7d, 30d
- Detect performance cliffs vs gradual degradation
- Segmented analysis: performance by user cohort, geography, or feature value

### Anomaly Detection

- Flag predictions that fall outside expected confidence ranges
- Isolation Forest and statistical outlier detection on prediction outputs
- Cluster incoming samples and detect out-of-distribution inputs
- Real-time anomaly scoring via prediction hook

### Alerting

- Rule-based alerts: drift score > threshold, accuracy drop > X%, anomaly rate spikes
- Alert channels: email, Slack webhook, PagerDuty
- Alert deduplication and cooldown periods
- Full alert history with root cause context

### Dashboard

- Real-time metrics on model health across all deployed models
- Feature importance drift visualization
- Prediction distribution histograms
- Exportable monitoring reports

---

## Architecture

```
+--------------------------------------------------------------+
|              Your Deployed Models / Services                 |
|  Prediction hook --> POST /api/v1/log  (each prediction)    |
+------------------------+-------------------------------------+
                         |
+------------------------v-------------------------------------+
|                  Sentinel-ML Backend (Python)               |
|  FastAPI - scikit-learn - pandas - scipy - numpy           |
|                                                              |
|  +-----------+  +----------+  +----------+  +----------+   |
|  |   Drift   |  | Perf     |  | Anomaly  |  |  Alert   |   |
|  |  Detector |  | Tracker  |  | Detector |  |  Engine  |   |
|  +-----------+  +----------+  +----------+  +----------+   |
|                           |                                  |
|                   Celery Workers (async analysis)            |
+------------------------+-------------------------------------+
                         |
              PostgreSQL (time-series metrics)
              +  Redis (queue + recent-window cache)
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Runtime** | Python 3.11 | Core monitoring logic |
| **API** | FastAPI | Prediction logging + metrics endpoints |
| **ML** | scikit-learn, scipy, numpy, pandas | Drift detection and anomaly scoring |
| **Database** | PostgreSQL 16 | Time-series metric storage |
| **Cache/Queue** | Redis 7, Celery | Async analysis pipeline |
| **Dashboard** | React 18, Recharts | Metrics visualization |
| **Alerts** | SMTP, Slack webhooks, PagerDuty API | Multi-channel notification |
| **CI** | GitHub Actions | Lint, test, build |

---

## Quick Start

### Option A: Docker

```bash
git clone https://github.com/avase33/sentinel-ml.git
cd sentinel-ml
cp .env.example .env
docker compose up -d
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

### Option B: Local Development

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env
uvicorn app.main:app --reload
```

---

## Integration

### Log predictions from your model

```python
import requests

# After each prediction in your model service:
requests.post("http://sentinel:8000/api/v1/log", json={
    "model_id": "churn-classifier-v3",
    "inputs": {"tenure_months": 12, "monthly_charges": 65.0, "contract": "month-to-month"},
    "prediction": 0.87,
    "confidence": 0.91,
    "timestamp": "2026-06-22T10:00:00Z"
})
```

### Set a baseline

```python
# POST /api/v1/models/{model_id}/baseline
# Body: your training dataset as JSON or CSV upload
```

### Configure alerts

```python
# POST /api/v1/alerts/rules
{
  "model_id": "churn-classifier-v3",
  "metric": "psi_score",
  "threshold": 0.2,
  "channel": "slack",
  "webhook_url": "https://hooks.slack.com/..."
}
```

---

## Drift Detection Methods

| Method | Use Case | Statistic |
|---|---|---|
| **Kolmogorov-Smirnov** | Continuous features | KS statistic + p-value |
| **Chi-Squared** | Categorical features | Chi2 statistic + p-value |
| **PSI (Population Stability Index)** | Binary classifier inputs | PSI score (>0.2 = major drift) |
| **Wasserstein Distance** | Distribution similarity | Earth Mover's Distance |
| **Isolation Forest** | Prediction anomalies | Anomaly score (-1 to 1) |

---

## Roadmap

- [ ] LLM output monitoring (hallucination detection, toxicity scoring)
- [ ] Automated retraining trigger on drift threshold breach
- [ ] SHAP-based feature importance shift analysis
- [ ] A/B test monitoring: compare two model versions in production
- [ ] Grafana integration for unified observability
- [ ] SDK: `pip install sentinel-ml-client`
- [ ] Multi-model correlation analysis
- [ ] SLA breach prediction

---

## License

```
Copyright (c) 2026 Akhil Vase. All rights reserved.

This source code is the proprietary property of Akhil Vase.
Unauthorized copying, distribution, or modification is strictly prohibited.
```

---

<div align="center">

**Your models are deployed. Now make sure they stay right.**

*Sentinel-ML -- Monitor the model. Catch the drift. Fire the alert.*

</div>
