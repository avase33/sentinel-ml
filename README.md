# 🚀 Production ML Platform

[![CI](https://github.com/akhilvase/production-ml-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/akhilvase/production-ml-platform/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.yml)
[![MLflow](https://img.shields.io/badge/MLflow-integrated-orange.svg)](https://mlflow.org)

A **batteries-included, production-ready ML platform** combining real-time model serving, RAG-powered LLM gateway, online feature store, A/B testing, and full model observability — deployable in minutes via Docker Compose or Kubernetes.

> Built from patterns across fintech and enterprise ML systems handling millions of daily predictions.

---

## ✨ What's Inside

| Component | Description | Tech |
|-----------|-------------|------|
| **Model Gateway** | FastAPI serving layer with versioned endpoints | FastAPI, Pydantic |
| **LLM / RAG Pipeline** | Multi-provider LLM routing with vector-backed RAG | LangChain, ChromaDB, OpenAI, Anthropic |
| **Feature Store** | Online + offline feature store with sub-10ms reads | Redis, PostgreSQL |
| **A/B Testing Router** | Champion-challenger traffic splitting with metric tracking | Redis, Prometheus |
| **Drift Monitor** | Statistical data drift + prediction quality monitoring | Evidently AI, Prometheus |
| **Experiment Tracking** | Full ML lifecycle with model registry | MLflow |
| **Observability Stack** | Metrics, dashboards, alerting | Prometheus, Grafana |
| **CI/CD** | Automated test → build → deploy pipeline | GitHub Actions, ArgoCD |

---

## 🏗️ Architecture

```
                         ┌─────────────────────────────────────────┐
                         │            API Gateway (FastAPI)         │
                         │  /predict  /rag  /features  /ab-test    │
                         └────────┬──────────┬────────┬────────────┘
                                  │          │        │
               ┌──────────────────┘          │        └──────────────────┐
               ▼                             ▼                           ▼
   ┌───────────────────┐       ┌─────────────────────┐      ┌──────────────────────┐
   │  A/B Test Router  │       │    RAG Pipeline      │      │   Feature Store      │
   │  champion 90%     │       │  Embed → Search →    │      │  Redis (online)      │
   │  challenger 10%   │       │  Retrieve → Generate │      │  Postgres (offline)  │
   └────────┬──────────┘       └──────────┬──────────┘      └──────────┬───────────┘
            │                             │                             │
            ▼                             ▼                             ▼
   ┌────────────────┐        ┌────────────────────┐       ┌────────────────────────┐
   │ Model Registry │        │   Vector Store      │       │   Drift Monitor        │
   │ MLflow         │        │   ChromaDB          │       │   Evidently AI         │
   └────────────────┘        └────────────────────┘       └──────────┬─────────────┘
                                                                      │
                                                          ┌───────────▼──────────────┐
                                                          │  Prometheus + Grafana     │
                                                          └───────────────────────────┘
```

---

## ⚡ Quick Start

### Docker Compose (recommended)

```bash
git clone https://github.com/akhilvase/production-ml-platform.git
cd production-ml-platform

# Set your API keys
cp .env.example .env
# Edit .env with your OpenAI/Anthropic keys

# Launch the full stack
make up

# Verify everything is healthy
make health
```

Services available at:
- **API**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Kubernetes (Helm)

```bash
helm repo add mlplatform https://akhilvase.github.io/production-ml-platform
helm install mlplatform mlplatform/mlplatform \
  --set openai.apiKey=$OPENAI_API_KEY \
  --set anthropic.apiKey=$ANTHROPIC_API_KEY \
  -f helm/mlplatform/values.yaml
```

---

## 🔌 API Reference

### Model Inference

```bash
# Real-time prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "fraud_detector",
    "features": {"amount": 150.0, "merchant_category": "electronics", "hour": 23},
    "version": "champion"
  }'

# Response
{
  "prediction": 0.87,
  "model_version": "v2.1.0",
  "variant": "champion",
  "latency_ms": 12.4,
  "request_id": "req_abc123"
}
```

### RAG / LLM Gateway

```bash
# Query with RAG
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the chargeback policy for recurring transactions?",
    "collection": "compliance_docs",
    "top_k": 5,
    "provider": "openai"
  }'
```

### Feature Store

```bash
# Get online features (sub-10ms)
curl http://localhost:8000/api/v1/features/user_123?feature_set=fraud_v2

# Batch upsert features
curl -X POST http://localhost:8000/api/v1/features/batch \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "user_123", "features": {"tx_count_24h": 12, "avg_amount_7d": 85.5}}'
```

### A/B Testing

```bash
# Route request (auto-splits traffic)
curl -X POST http://localhost:8000/api/v1/ab/predict \
  -H "Content-Type: application/json" \
  -d '{"experiment": "fraud_model_v3_test", "entity_id": "user_123", "features": {...}}'
```

---

## 📊 Model Monitoring

The platform automatically tracks:

- **Data drift** — PSI and KL-divergence on input feature distributions
- **Prediction drift** — Score distribution shifts over time
- **Feature skew** — Training vs. serving distribution mismatch
- **Business metrics** — Precision@threshold, catch rate, false positive rate

Drift alerts trigger **automated retraining** via the pipeline webhook.

```python
from src.monitoring.drift import DriftMonitor

monitor = DriftMonitor(model_name="fraud_detector", reference_dataset="train_2024_q4")
report = monitor.run(current_data=serving_logs_df)

if report.drift_detected:
    monitor.trigger_retraining()  # Fires GitHub Actions workflow
```

---

## 🧪 A/B Testing

```python
from src.ab_testing.router import ABRouter

router = ABRouter(experiment_name="fraud_model_v3_test")

# Define split
router.configure(
    champion={"model": "fraud_v2", "traffic": 0.9},
    challenger={"model": "fraud_v3", "traffic": 0.1}
)

# Route and record
result = router.predict(entity_id="user_123", features=features)
router.record_outcome(request_id=result.request_id, label=actual_fraud_label)

# Get stats
stats = router.get_experiment_stats()
# {"champion_precision": 0.931, "challenger_precision": 0.948, "p_value": 0.031}
```

---

## 🔁 Training Pipeline

```bash
# Train a new model with experiment tracking
python -m src.training.trainer \
  --model-type xgboost \
  --dataset s3://your-bucket/features/fraud_v2/ \
  --experiment fraud_detection_q2_2025 \
  --register-if-better

# MLflow auto-logs: params, metrics, artifacts, model card
```

---

## 📦 Project Structure

```
production-ml-platform/
├── src/
│   ├── api/                    # FastAPI app + routers
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── models.py       # /predict endpoints
│   │   │   ├── features.py     # /features endpoints
│   │   │   ├── llm.py          # /rag endpoints
│   │   │   └── ab_test.py      # /ab endpoints
│   │   └── middleware/
│   │       └── auth.py
│   ├── feature_store/
│   │   ├── store.py            # Online (Redis) + offline (Postgres)
│   │   └── pipeline.py         # Kafka → feature computation
│   ├── training/
│   │   ├── trainer.py          # MLflow-tracked training
│   │   └── experiments.py      # Experiment management
│   ├── serving/
│   │   ├── registry.py         # MLflow model registry wrapper
│   │   └── predictor.py        # Model loading + inference
│   ├── monitoring/
│   │   ├── drift.py            # Evidently AI drift detection
│   │   └── metrics.py          # Prometheus instrumentation
│   ├── rag/
│   │   ├── pipeline.py         # End-to-end RAG orchestration
│   │   ├── embeddings.py       # Embedding generation + caching
│   │   └── retriever.py        # ChromaDB / Pinecone retrieval
│   └── ab_testing/
│       └── router.py           # Traffic splitting + stats
├── helm/mlplatform/            # Kubernetes Helm chart
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboards/     # Pre-built ML dashboards
├── .github/workflows/
│   ├── ci.yml                  # Test + lint on PR
│   └── cd.yml                  # Deploy on merge to main
├── tests/
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## 🧰 Tech Stack

**Core:** Python 3.10+, FastAPI, Pydantic v2  
**ML:** XGBoost, LightGBM, PyTorch, Scikit-learn, HuggingFace Transformers  
**LLM/RAG:** LangChain, OpenAI API, Anthropic API, ChromaDB, Sentence Transformers  
**MLOps:** MLflow, Evidently AI, DVC  
**Feature Store:** Redis, PostgreSQL  
**Streaming:** Apache Kafka (optional, for high-throughput feature pipelines)  
**Infra:** Docker, Kubernetes (EKS/AKS/GKE), Helm, Terraform  
**Observability:** Prometheus, Grafana, Datadog-compatible metrics  
**CI/CD:** GitHub Actions, ArgoCD  

---

## 🤝 Contributing

PRs welcome. Please open an issue first for major changes.

```bash
git clone https://github.com/akhilvase/production-ml-platform.git
cd production-ml-platform
pip install -e ".[dev]"
make test
```

---

## 📄 License

MIT © [Akhil Vase](https://linkedin.com/in/akhil-vase)
