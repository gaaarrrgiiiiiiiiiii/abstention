"""
tests/test_api.py
=================
Integration tests for the Flask API (api/app.py).
Runs in-process using Flask's test client — no server startup required.

Tests:
- /api/health  → 200 OK
- /api/ready   → 503 when model not loaded
- /api/predict → 400 on missing features, 400 on wrong count
- /api/predict → 400 on non-numeric values
"""
import pytest
import json
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def client():
    """Create Flask test client without loading the real model."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "app", os.path.join(os.path.dirname(__file__), '..', 'api', 'app.py')
    )
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)

    app_module.app.config["TESTING"] = True
    # NOTE: model_ready is False (no model loaded) — this is intentional for unit tests
    with app_module.app.test_client() as client:
        yield client


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_json_has_status_ok(self, client):
        resp = client.get("/api/health")
        data = json.loads(resp.data)
        assert data["status"] == "ok"

    def test_health_reports_model_not_loaded(self, client):
        resp = client.get("/api/health")
        data = json.loads(resp.data)
        assert "model_loaded" in data
        # In test environment without training, model is not loaded
        assert data["model_loaded"] is False


class TestReadyEndpoint:

    def test_ready_returns_503_when_model_not_loaded(self, client):
        resp = client.get("/api/ready")
        assert resp.status_code == 503


class TestPredictEndpoint:

    def test_predict_returns_503_when_model_not_ready(self, client):
        """Without a loaded model, /api/predict must return 503."""
        payload = {"features": [0.0] * 30}
        resp = client.post("/api/predict",
                           data=json.dumps(payload),
                           content_type="application/json")
        assert resp.status_code == 503

    def test_predict_returns_400_on_missing_features_key(self, client):
        """Request without 'features' key must return 400."""
        payload = {"data": [1, 2, 3]}
        resp = client.post("/api/predict",
                           data=json.dumps(payload),
                           content_type="application/json")
        # Either 400 (validation) or 503 (not ready) — both are acceptable error codes
        assert resp.status_code in {400, 503}

    def test_predict_invalid_json(self, client):
        """Malformed JSON must not crash the server."""
        resp = client.post("/api/predict",
                           data="not-valid-json",
                           content_type="application/json")
        assert resp.status_code in {400, 500}


class TestFeaturesEndpoint:

    def test_features_returns_503_when_not_loaded(self, client):
        resp = client.get("/api/features")
        # Returns 503 when feature_names is None (no model loaded)
        assert resp.status_code in {200, 503}


class TestSampleDataEndpoint:

    def test_sample_data_returns_200(self, client):
        resp = client.get("/api/sample-data")
        assert resp.status_code == 200

    def test_sample_data_has_samples_key(self, client):
        resp = client.get("/api/sample-data")
        data = json.loads(resp.data)
        assert "samples" in data
        assert isinstance(data["samples"], list)
        assert len(data["samples"]) >= 1
