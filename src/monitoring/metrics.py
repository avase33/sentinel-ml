"""
Prometheus metrics definitions — shared across the platform.

Import these in any module that needs to record metrics.
Prometheus client auto-registers them in the default registry,
which is exposed at /metrics by the FastAPI app.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary

# ---------------------------------------------------------------------------
# HTTP / API metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "mlplatform_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "mlplatform_http_request_duration_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ---------------------------------------------------------------------------
# Model inference metrics
# ---------------------------------------------------------------------------

PREDICTION_COUNT = Counter(
    "mlplatform_predictions_total",
    "Total predictions served",
    ["model_name", "variant", "status"],   # status: success | error
)

PREDICTION_LATENCY = Histogram(
    "mlplatform_prediction_latency_seconds",
    "Model inference latency",
    ["model_name", "variant"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

PREDICTION_SCORE = Histogram(
    "mlplatform_prediction_score",
    "Distribution of prediction scores (0–1)",
    ["model_name", "variant"],
    buckets=[i / 10 for i in range(11)],
)

# ---------------------------------------------------------------------------
# RAG / LLM metrics
# ---------------------------------------------------------------------------

RAG_REQUEST_COUNT = Counter(
    "mlplatform_rag_requests_total",
    "Total RAG/LLM requests",
    ["provider", "collection", "status"],
)

RAG_LATENCY = Histogram(
    "mlplatform_rag_latency_seconds",
    "RAG pipeline end-to-end latency",
    ["provider"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

RAG_TOKENS_USED = Counter(
    "mlplatform_rag_tokens_total",
    "Total tokens consumed by LLM calls",
    ["provider"],
)

RAG_DOCS_RETRIEVED = Histogram(
    "mlplatform_rag_docs_retrieved",
    "Number of documents retrieved per query",
    ["collection"],
    buckets=[1, 2, 3, 5, 10, 20],
)

# ---------------------------------------------------------------------------
# Feature store metrics
# ---------------------------------------------------------------------------

FEATURE_STORE_HITS = Counter(
    "mlplatform_feature_store_hits_total",
    "Online feature store cache hits",
    ["feature_set"],
)

FEATURE_STORE_MISSES = Counter(
    "mlplatform_feature_store_misses_total",
    "Online feature store cache misses (entity not found)",
    ["feature_set"],
)

FEATURE_STORE_LATENCY = Histogram(
    "mlplatform_feature_store_latency_seconds",
    "Online feature store read latency",
    ["feature_set"],
    buckets=[0.001, 0.003, 0.005, 0.01, 0.025, 0.05],
)

# ---------------------------------------------------------------------------
# A/B testing metrics
# ---------------------------------------------------------------------------

AB_PREDICTION_COUNT = Counter(
    "mlplatform_ab_predictions_total",
    "Predictions made per A/B experiment variant",
    ["experiment", "variant"],
)

AB_OUTCOME_COUNT = Counter(
    "mlplatform_ab_outcomes_total",
    "Ground-truth outcomes recorded per experiment variant",
    ["experiment", "variant", "outcome"],  # outcome: tp|fp|fn|tn
)

# ---------------------------------------------------------------------------
# Drift monitoring metrics
# ---------------------------------------------------------------------------

DRIFT_FEATURE_SHARE = Gauge(
    "mlplatform_drift_feature_share",
    "Fraction of features currently drifted",
    ["model_name"],
)

DRIFT_DETECTED = Gauge(
    "mlplatform_drift_detected",
    "1 if drift is currently detected for a model, 0 otherwise",
    ["model_name"],
)

RETRAINING_TRIGGERS = Counter(
    "mlplatform_retraining_triggers_total",
    "Number of automated retraining triggers fired",
    ["model_name", "trigger_reason"],
)

# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------

MODELS_LOADED = Gauge(
    "mlplatform_models_loaded_total",
    "Number of models currently loaded in memory",
)

MODEL_LOAD_TIME = Summary(
    "mlplatform_model_load_duration_seconds",
    "Time to load a model from the registry",
    ["model_name"],
)
