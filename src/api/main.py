"""
Production ML Platform — FastAPI Application Entry Point
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from src.api.routers import ab_test, features, llm, models
from src.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY
from src.serving.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Lifespan: warm up registry and feature store on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    await registry.load_champions()          # pre-load champion models into memory
    app.state.registry = registry
    yield
    await registry.unload_all()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Production ML Platform",
    description="Real-time model serving, RAG/LLM gateway, feature store, A/B testing, and drift monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose /metrics for Prometheus scraping
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ---------------------------------------------------------------------------
# Request ID + latency middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency * 1000:.2f}"
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(models.router,   prefix="/api/v1",         tags=["Model Inference"])
app.include_router(features.router, prefix="/api/v1",         tags=["Feature Store"])
app.include_router(llm.router,      prefix="/api/v1",         tags=["LLM / RAG"])
app.include_router(ab_test.router,  prefix="/api/v1",         tags=["A/B Testing"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": app.version}
