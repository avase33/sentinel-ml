"""
Predictor — inference engine wrapping the ModelRegistry.
Handles feature validation, pre/post-processing, and async inference.
"""

import asyncio
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.serving.registry import ModelRegistry

logger = logging.getLogger(__name__)

# Singleton registry shared across the process
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


class Predictor:
    def __init__(self):
        self.registry = get_registry()
        self._loop = asyncio.get_event_loop()

    # ------------------------------------------------------------------
    # Single-row inference
    # ------------------------------------------------------------------

    async def predict(
        self,
        model_name: str,
        features: dict[str, Any],
        version: str = "champion",
    ) -> dict:
        """
        Async single-row prediction.
        Returns: {prediction, probabilities, version, variant}
        """
        loaded = self.registry.get(model_name, variant=version)

        df = pd.DataFrame([features])
        result = await self._loop.run_in_executor(None, loaded.model.predict, df)

        # Normalise output: classifiers return array of shape (n, k) or (n,)
        prediction, probabilities = self._parse_output(result)

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "version": loaded.version,
            "variant": loaded.stage,
        }

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    async def predict_batch(
        self,
        model_name: str,
        rows: list[dict[str, Any]],
        version: str = "champion",
    ) -> dict:
        loaded = self.registry.get(model_name, variant=version)
        df = pd.DataFrame(rows)
        result = await self._loop.run_in_executor(None, loaded.model.predict, df)

        if isinstance(result, np.ndarray) and result.ndim == 2:
            # Multi-class: take argmax or positive-class probability
            predictions = result[:, 1].tolist() if result.shape[1] == 2 else result.argmax(axis=1).tolist()
        else:
            predictions = np.asarray(result).flatten().tolist()

        return {
            "predictions": predictions,
            "version": loaded.version,
        }

    # ------------------------------------------------------------------
    # Registry introspection
    # ------------------------------------------------------------------

    async def list_loaded_models(self) -> list[dict]:
        return self.registry.list_models()

    async def get_model_versions(self, model_name: str) -> list[dict]:
        """Fetch all versions from MLflow registry for a model."""
        loop = asyncio.get_event_loop()
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        versions = await loop.run_in_executor(
            None,
            lambda: client.search_model_versions(f"name='{model_name}'"),
        )
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "description": v.description,
                "created_at": v.creation_timestamp,
            }
            for v in versions
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output(result) -> tuple[float, dict | None]:
        arr = np.asarray(result)
        if arr.ndim == 2 and arr.shape[1] == 2:
            # Binary classifier with [neg_prob, pos_prob] columns
            pos_prob = float(arr[0, 1])
            return pos_prob, {"negative": float(arr[0, 0]), "positive": pos_prob}
        elif arr.ndim == 2:
            # Multi-class
            idx = int(arr[0].argmax())
            probs = {str(i): float(p) for i, p in enumerate(arr[0])}
            return float(idx), probs
        else:
            return float(arr.flat[0]), None
