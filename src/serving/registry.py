"""
Model Registry — MLflow-backed wrapper for champion/challenger model management.
Handles model loading, versioning, and hot-swapping without service restarts.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://mlflow:5000"  # override via env var MLFLOW_TRACKING_URI


@dataclass
class LoadedModel:
    name: str
    version: str
    stage: str          # "champion" | "challenger" | "staging"
    model: Any          # mlflow.pyfunc.PyFuncModel
    metadata: dict = field(default_factory=dict)


class ModelRegistry:
    """
    Thin async wrapper around the MLflow model registry.

    - Loads champion (Production stage) and challenger (Staging stage) models on startup.
    - Supports hot-reloading: call reload_model() to swap a model without restart.
    - Thread-safe: model references are replaced atomically via asyncio.Lock.
    """

    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI):
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient()
        self._models: dict[str, dict[str, LoadedModel]] = {}   # name → {stage → LoadedModel}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def load_champions(self) -> None:
        """Load all Production-stage models from the registry on startup."""
        loop = asyncio.get_event_loop()
        registered = await loop.run_in_executor(None, self._client.search_registered_models)

        tasks = []
        for rm in registered:
            tasks.append(self._load_stage(rm.name, "Production", "champion"))
            tasks.append(self._load_stage(rm.name, "Staging", "challenger"))

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Registry loaded. Models: %s", list(self._models.keys()))

    async def _load_stage(self, model_name: str, mlflow_stage: str, alias: str) -> None:
        loop = asyncio.get_event_loop()
        try:
            versions = await loop.run_in_executor(
                None,
                lambda: self._client.get_latest_versions(model_name, stages=[mlflow_stage]),
            )
            if not versions:
                return

            mv = versions[0]
            model_uri = f"models:/{model_name}/{mv.version}"
            pyfunc_model = await loop.run_in_executor(
                None, mlflow.pyfunc.load_model, model_uri
            )

            loaded = LoadedModel(
                name=model_name,
                version=mv.version,
                stage=alias,
                model=pyfunc_model,
                metadata={"run_id": mv.run_id, "description": mv.description or ""},
            )

            async with self._lock:
                self._models.setdefault(model_name, {})[alias] = loaded

            logger.info("Loaded %s [%s] v%s", model_name, alias, mv.version)
        except Exception as exc:
            logger.warning("Could not load %s/%s: %s", model_name, mlflow_stage, exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, model_name: str, variant: str = "champion") -> LoadedModel:
        """
        Retrieve a loaded model by name and variant.
        Raises KeyError if not found — callers should handle and return 404.
        """
        try:
            return self._models[model_name][variant]
        except KeyError:
            raise KeyError(f"Model '{model_name}' variant '{variant}' is not loaded.")

    def list_models(self) -> list[dict]:
        return [
            {"name": name, "variant": variant, "version": lm.version, "stage": lm.stage}
            for name, variants in self._models.items()
            for variant, lm in variants.items()
        ]

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    async def reload_model(self, model_name: str, version: str, variant: str = "champion") -> None:
        """Hot-swap a model version without restarting the service."""
        loop = asyncio.get_event_loop()
        model_uri = f"models:/{model_name}/{version}"
        pyfunc_model = await loop.run_in_executor(
            None, mlflow.pyfunc.load_model, model_uri
        )
        mv = await loop.run_in_executor(
            None, lambda: self._client.get_model_version(model_name, version)
        )
        loaded = LoadedModel(
            name=model_name,
            version=version,
            stage=variant,
            model=pyfunc_model,
            metadata={"run_id": mv.run_id},
        )
        async with self._lock:
            self._models.setdefault(model_name, {})[variant] = loaded
        logger.info("Hot-reloaded %s [%s] → v%s", model_name, variant, version)

    async def unload_all(self) -> None:
        async with self._lock:
            self._models.clear()
        logger.info("All models unloaded.")
