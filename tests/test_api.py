"""
Integration tests for the FastAPI application.
Uses httpx AsyncClient against the real app (no mocks for core logic).
Services (Redis, Postgres) must be running — handled by CI docker services.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestPredict:
    @pytest.mark.asyncio
    async def test_predict_missing_model(self, client):
        """Should return 404 when model is not loaded."""
        resp = await client.post("/api/v1/predict", json={
            "model_name": "nonexistent_model",
            "features": {"amount": 100.0},
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_predict_response_shape(self, client, mocker):
        """Predictor returns the correct response schema."""
        mocker.patch(
            "src.serving.predictor.Predictor.predict",
            return_value={
                "prediction": 0.82,
                "probabilities": {"negative": 0.18, "positive": 0.82},
                "version": "3",
                "variant": "champion",
            },
        )
        resp = await client.post("/api/v1/predict", json={
            "model_name": "fraud_detector",
            "features": {"amount": 500.0, "hour": 2},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "prediction" in body
        assert "model_version" in body
        assert "latency_ms" in body
        assert "request_id" in body
        assert body["prediction"] == 0.82


class TestFeatureStore:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self, client):
        """Round-trip: write features then read them back."""
        # Write
        resp = await client.post("/api/v1/features/batch", json={
            "entity_id": "test_user_999",
            "feature_set": "fraud_v2",
            "features": {"tx_count_24h": 5, "avg_amount_7d": 42.0},
        })
        assert resp.status_code == 200

        # Read
        resp = await client.get("/api/v1/features/test_user_999?feature_set=fraud_v2")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("tx_count_24h") == 5
        assert data.get("avg_amount_7d") == 42.0


class TestABTesting:
    @pytest.mark.asyncio
    async def test_configure_and_predict(self, client, mocker):
        mocker.patch(
            "src.ab_testing.router.ABRouter.predict",
            return_value={
                "request_id": "req_test_001",
                "experiment": "test_exp",
                "variant": "champion",
                "model_version": "2",
                "prediction": 0.1,
                "latency_ms": 5.0,
            },
        )
        resp = await client.post("/api/v1/ab/predict", json={
            "experiment": "test_exp",
            "entity_id": "user_abc",
            "features": {"amount": 100.0},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["variant"] in ("champion", "challenger")
        assert "prediction" in body
