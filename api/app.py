"""
Flask API for the Agentic Abstention Governance Framework.

Loads the trained PyTorch AbstentionModel and the fitted StandardScaler
from the data/model_metadata.json produced by src/export_scaler_features.py,
then serves predictions via the /predict endpoint.

FIXED: Was previously hardcoded to UCI Credit Card (input_dim=30, V1-V28).
       Now dynamically reads input_dim and scaler from the training pipeline's
       serialized metadata, correctly supporting the IEEE-CIS Fraud dataset
       (~432 features after identity merge and behavioral enrichment).
"""

import sys
import os
import json
import hashlib
import datetime
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from abstention_model import AbstentionModel

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:5000",
                   "http://localhost:5000"])          # restrict to known origins

# ── global state ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(PROJECT_ROOT, "abstention_model.pth")
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "model_metadata.json")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "predictions.jsonl")

model = None
scaler_mean = None
scaler_std = None
feature_names = None
input_dim = None
model_ready = False


# ── startup: load model + scaler from metadata ────────────────────────────────
def load_model_and_scaler():
    global model, scaler_mean, scaler_std, feature_names, input_dim, model_ready

    # 1. Load metadata JSON (produced by src/export_scaler_features.py)
    if not os.path.exists(METADATA_PATH):
        print(
            f"ERROR: {METADATA_PATH} not found.\n"
            "Run `cd src && python export_scaler_features.py` after training."
        )
        return

    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)

    input_dim = meta["input_dim"]
    feature_names = meta["feature_names"]
    scaler_mean = np.array(meta["scaler_mean"], dtype=np.float32)
    scaler_std = np.array(meta["scaler_std"], dtype=np.float32)

    print(f"Scaler loaded: input_dim={input_dim}, "
          f"train_size={meta.get('train_size', 'N/A')}, "
          f"fraud_rate={meta.get('fraud_rate_train', 0):.4%}")

    # 2. Load the model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}. Train first.")
        return

    model = AbstentionModel(input_dim=input_dim, dropout=0.0).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    model_ready = True
    print(f"Model loaded from {MODEL_PATH} on {DEVICE} (input_dim={input_dim})")

    # 3. Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def _scale_features(raw: list) -> np.ndarray:
    """Apply the training scaler (z-score) to a raw feature vector."""
    x = np.array(raw, dtype=np.float32)
    return (x - scaler_mean) / (scaler_std + 1e-8)


def _log_prediction(feature_hash: str, prediction: str, confidence: dict):
    """Append a structured log entry to predictions.jsonl (GDPR-safe: no raw values)."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "feature_hash": feature_hash,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None,
    }
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # logging must never crash the prediction path


def _validate_features(features):
    """
    Validate and coerce a raw features list.
    Returns (coerced_list, error_response_or_None).
    """
    if len(features) != input_dim:
        return None, ({"error": f"Expected {input_dim} features, received {len(features)}.",
                       "hint": "Check GET /api/features for the expected feature list."}, 400)
    try:
        features = [float(x) for x in features]
    except (ValueError, TypeError):
        return None, ({"error": "All feature values must be numeric."}, 400)
    if any(abs(v) > 1e7 for v in features):
        return None, ({"error": "One or more feature values are out of plausible range (|val| > 1e7)."}, 400)
    return features, None


def _gradient_x_input_attribution(scaled: np.ndarray, target_class: int, top_n: int = 15):
    """
    Compute gradient × input attribution scores for a single sample.

    This is a fast (CPU ~2ms), zero-dependency attribution method:
      attribution_i = ∂logit_k/∂x_i  ×  x_i

    where k = target_class. Positive values push toward the target class;
    negative values push away. We return the top_n features sorted by
    absolute magnitude.

    Args:
        scaled:       z-score-normalised feature vector, shape (D,).
        target_class: the predicted class index (0=Legit, 1=Fraud, 2=Abstain).
        top_n:        number of features to return.

    Returns:
        list of dicts with keys: feature, score, direction.
    """
    x = torch.tensor(scaled, dtype=torch.float32, requires_grad=True).unsqueeze(0).to(DEVICE)

    # Forward pass with grad enabled
    model.train(False)
    outputs = model(x)
    logit = outputs[0, target_class]
    logit.backward()

    grad = x.grad.detach().cpu().numpy()[0]        # (D,)
    attr = grad * scaled                            # gradient × input

    # Pick top_n by absolute value
    indices = np.argsort(np.abs(attr))[::-1][:top_n]
    results = []
    for idx in indices:
        fname = feature_names[idx] if feature_names else f"feature_{idx}"
        results.append({
            "feature":   fname,
            "score":     float(round(attr[idx], 6)),
            "direction": "positive" if attr[idx] >= 0 else "negative",
        })
    return results


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Basic liveness check."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler_mean is not None,
        "device": str(DEVICE),
    })


@app.route("/api/ready", methods=["GET"])
def ready():
    """Readiness probe: returns 200 only when model + scaler are fully loaded."""
    if model_ready:
        # Run a dummy forward pass to confirm inference works
        try:
            dummy = torch.zeros(1, input_dim, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                _ = model(dummy)
            return jsonify({"status": "ready", "input_dim": input_dim}), 200
        except Exception as e:
            return jsonify({"status": "error", "detail": str(e)}), 503
    return jsonify({"status": "not_ready",
                    "hint": "Run export_scaler_features.py and train the model."}), 503


@app.route("/api/features", methods=["GET"])
def get_features():
    """Return feature names and expected count."""
    if feature_names is None:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({
        "features": feature_names,
        "count": len(feature_names),
        "input_dim": input_dim,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run abstention-aware prediction on user-provided features.

    Request body:
        { "features": [val1, val2, ..., valN] }

    where N == input_dim from model_metadata.json.

    Response:
        {
            "prediction": "Legitimate" | "Fraud" | "Abstain",
            "prediction_code": 0 | 1 | 2,
            "confidence": { "legitimate": 0.xx, "fraud": 0.xx, "abstain": 0.xx },
            "should_decide": true | false,
            "recommendation": "..."
        }
    """
    if not model_ready:
        return jsonify({"error": "Model not ready. Check /api/ready."}), 503

    try:
        data = request.get_json(force=True)

        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key in JSON body."}), 400

        features, err = _validate_features(data["features"])
        if err:
            return jsonify(err[0]), err[1]

        # Scale features
        scaled = _scale_features(features)

        # Log request (hash raw, never log raw values)
        feature_hash = hashlib.sha256(
            json.dumps(features).encode()
        ).hexdigest()[:16]

        # Run model
        input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(outputs, dim=1).item()

        probs_np = probs.cpu().numpy()[0]

        LABELS = {0: "Legitimate", 1: "Fraud", 2: "Abstain"}
        prediction_label = LABELS[pred_class]
        should_decide = pred_class != 2

        if pred_class == 0:
            recommendation = (
                f"Legitimate with {probs_np[0] * 100:.2f}% confidence. "
                "Model has committed to this prediction."
            )
        elif pred_class == 1:
            recommendation = (
                f"FRAUD ALERT — {probs_np[1] * 100:.2f}% confidence. "
                "Immediate manual review is recommended."
            )
        else:
            recommendation = (
                f"Model uncertainty is too high (abstain score {probs_np[2] * 100:.2f}%). "
                "Deferred to human analyst queue."
            )

        confidence = {
            "legitimate": round(float(probs_np[0]), 6),
            "fraud":      round(float(probs_np[1]), 6),
            "abstain":    round(float(probs_np[2]), 6),
        }
        _log_prediction(feature_hash, prediction_label, confidence)

        return jsonify({
            "prediction":      prediction_label,
            "prediction_code": pred_class,
            "confidence":      confidence,
            "should_decide":   should_decide,
            "recommendation":  recommendation,
            "feature_count":   input_dim,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/explain", methods=["POST"])
def explain():
    """
    Return gradient×input attribution scores for a prediction.

    Uses gradient × input (integrated-gradients lite) — a fast, zero-dependency
    attribution method suitable for CPU inference on high-dimensional tabular data.

    Request body (same shape as /api/predict):
        { "features": [val1, ..., valN], "top_n": 15 }

    Response:
        {
            "prediction":   "Legitimate" | "Fraud" | "Abstain",
            "prediction_code": 0 | 1 | 2,
            "attributions": [
                { "feature": "TransactionAmt", "score": 0.043, "direction": "positive" },
                ...
            ],
            "method": "gradient_x_input",
            "note": "Positive score = pushes toward the predicted class."
        }
    """
    if not model_ready:
        return jsonify({"error": "Model not ready. Check /api/ready."}), 503

    try:
        data = request.get_json(force=True)

        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key in JSON body."}), 400

        features, err = _validate_features(data["features"])
        if err:
            return jsonify(err[0]), err[1]

        top_n = int(data.get("top_n", 15))
        top_n = max(1, min(top_n, input_dim))   # clamp to [1, input_dim]

        scaled = _scale_features(features)

        # Run predict first (no grad) to get the predicted class
        input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_class = torch.argmax(outputs, dim=1).item()

        LABELS = {0: "Legitimate", 1: "Fraud", 2: "Abstain"}
        prediction_label = LABELS[pred_class]

        # Compute gradient × input attribution for the predicted class
        attributions = _gradient_x_input_attribution(scaled, target_class=pred_class, top_n=top_n)

        return jsonify({
            "prediction":      prediction_label,
            "prediction_code": pred_class,
            "attributions":    attributions,
            "method":          "gradient_x_input",
            "note":            (
                "Positive score = feature pushed toward the predicted class. "
                "Negative score = feature pushed against the predicted class."
            ),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample-data", methods=["GET"])
def sample_data():
    """
    Return sample test vectors.
    NOTE: These are now zero-padded placeholders. To generate realistic samples,
    run src/evaluation.py which saves real test vectors.
    For demo purposes, realistic pre-scaled values from the IEEE-CIS dataset
    are loaded if available from data/sample_vectors.json.
    """
    sample_file = os.path.join(PROJECT_ROOT, "data", "sample_vectors.json")
    if os.path.exists(sample_file):
        with open(sample_file) as f:
            samples = json.load(f)
        return jsonify({"samples": samples, "feature_count": input_dim or 0})

    # Fallback: zeros (placeholder)
    dim = input_dim or 30
    samples = [
        {
            "name": "Zero vector (placeholder)",
            "description": (
                "Run export_scaler_features.py to generate real sample vectors. "
                "This is a placeholder with all features set to 0."
            ),
            "features": [0.0] * dim,
        }
    ]
    return jsonify({"samples": samples, "feature_count": dim})


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model_and_scaler()
    # Development server only — use waitress/gunicorn in production
    app.run(host="0.0.0.0", port=5000, debug=False)
