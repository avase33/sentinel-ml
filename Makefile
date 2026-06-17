.PHONY: help up down logs health test lint train ingest-docs

DOCKER_COMPOSE = docker compose

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Stack management ──────────────────────────────────────────────────────────

up:  ## Start the full platform stack
	$(DOCKER_COMPOSE) up -d --build
	@echo ""
	@echo "  API     → http://localhost:8000/docs"
	@echo "  MLflow  → http://localhost:5000"
	@echo "  Grafana → http://localhost:3000  (admin/admin)"
	@echo "  Prometheus → http://localhost:9090"
	@echo ""

down:  ## Stop all services
	$(DOCKER_COMPOSE) down

restart:  ## Restart the API service only (fast reload)
	$(DOCKER_COMPOSE) restart api

logs:  ## Follow API logs
	$(DOCKER_COMPOSE) logs -f api

health:  ## Check health of all services
	@curl -sf http://localhost:8000/health | python3 -m json.tool
	@echo "API ✅"
	@curl -sf http://localhost:5000/health | python3 -m json.tool
	@echo "MLflow ✅"

# ── Development ───────────────────────────────────────────────────────────────

install:  ## Install Python dependencies
	pip install -e ".[dev]"

lint:  ## Run ruff + mypy
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

test:  ## Run test suite
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:  ## Run tests excluding slow integration tests
	pytest tests/ -v -m "not slow" --cov=src

# ── Training ──────────────────────────────────────────────────────────────────

train:  ## Train fraud detector (example) — set DATASET env var
	python -m src.training.trainer \
		--dataset $(DATASET) \
		--target label \
		--model-name fraud_detector \
		--model-type xgboost \
		--experiment fraud_detection \
		--register-if-better

# ── RAG ingestion ─────────────────────────────────────────────────────────────

ingest-docs:  ## Ingest documents into ChromaDB — set DOCS_DIR and COLLECTION env vars
	python -c "\
import asyncio, pathlib, sys; \
from langchain.document_loaders import DirectoryLoader; \
from langchain.text_splitter import RecursiveCharacterTextSplitter; \
from src.rag.pipeline import RAGPipeline; \
loader = DirectoryLoader('$(DOCS_DIR)', glob='**/*.{pdf,txt,md}'); \
docs = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64).split_documents(loader.load()); \
asyncio.run(RAGPipeline().ingest(docs, collection='$(COLLECTION)')); \
print(f'Ingested {len(docs)} chunks')"

# ── Monitoring ────────────────────────────────────────────────────────────────

drift-check:  ## Run drift check on current serving data — set MODEL and DATA env vars
	python -c "\
import pandas as pd; \
from src.monitoring.drift import DriftMonitor; \
ref = pd.read_parquet('$(REF_DATA)'); \
cur = pd.read_parquet('$(CUR_DATA)'); \
m = DriftMonitor('$(MODEL)', ref); \
r = m.run(cur); \
print('Drift detected:', r.drift_detected); \
print('Feature drift share:', r.feature_drift_share); \
print('Drifted features:', r.drifted_features)"

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:  ## Build the Docker image
	docker build -t production-ml-platform:local .

docker-push:  ## Push to GHCR (requires docker login)
	docker tag production-ml-platform:local ghcr.io/akhilvase/production-ml-platform:latest
	docker push ghcr.io/akhilvase/production-ml-platform:latest

# ── Kubernetes ────────────────────────────────────────────────────────────────

helm-install:  ## Install via Helm into current kubectl context
	helm upgrade --install mlplatform ./helm/mlplatform \
		--set env.OPENAI_API_KEY=$(OPENAI_API_KEY) \
		--set env.ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		--wait

helm-uninstall:  ## Uninstall from current kubectl context
	helm uninstall mlplatform
