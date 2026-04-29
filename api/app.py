"""
Flask API for the Abstention Model.
Loads the trained PyTorch model and the fitted StandardScaler,
then serves predictions via /predict endpoint.
"""

import sys
import os
import json
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add the src directory so we can import model classes and data utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from abstention_model import AbstentionModel
from dataset import load_data

app = Flask(__name__)
CORS(app)

# ============================================================
# Model & Scaler Loading
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'abstention_model.pth')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')

# Feature names for the credit card dataset
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'scaler.joblib')

model = None
scaler = None


def load_model_and_scaler():
    """Load the trained abstention model and the fitted scaler."""
    global model, scaler

    # --- Load scaler (prefer persisted joblib, fallback to CSV) ---
    if os.path.exists(SCALER_PATH):
        import joblib
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded from {SCALER_PATH}")
    else:
        print("WARNING: scaler.joblib not found, falling back to CSV fitting...")
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        from sklearn.model_selection import train_test_split

        data = pd.read_csv(DATA_PATH)
        X = data.drop("Class", axis=1).values
        y = data["Class"].values
        X_train, _, _, _ = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

        scaler = StandardScaler()
        scaler.fit(X_train)
        print(f"Scaler fitted on {len(X_train)} training samples (CSV fallback).")

    # --- Load the model ---
    model = AbstentionModel(input_dim=30, dropout=0.0).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    print(f"Model loaded from {MODEL_PATH} on {DEVICE}")


# ============================================================
# Routes
# ============================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "device": str(DEVICE)
    })


@app.route('/api/features', methods=['GET'])
def get_features():
    """Return the list of feature names the model expects."""
    return jsonify({
        "features": FEATURE_NAMES,
        "count": len(FEATURE_NAMES)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Run prediction on user-provided feature values.
    
    Expects JSON body:
    {
        "features": [val1, val2, ..., val30]
    }
    
    Returns:
    {
        "prediction": "Legitimate" | "Fraud" | "Abstain",
        "prediction_code": 0 | 1 | 2,
        "confidence": { "legitimate": 0.xx, "fraud": 0.xx, "abstain": 0.xx },
        "should_decide": true | false,
        "recommendation": "..."
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' in request body."}), 400
        
        features = data['features']
        
        if len(features) != 30:
            return jsonify({
                "error": f"Expected 30 features, received {len(features)}."
            }), 400
        
        # Validate all values are numeric
        try:
            features = [float(x) for x in features]
        except (ValueError, TypeError):
            return jsonify({
                "error": "All feature values must be numeric."
            }), 400
        
        # Scale the features using the same scaler used during training
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Convert to tensor
        input_tensor = torch.tensor(
            features_scaled, dtype=torch.float32
        ).to(DEVICE)
        
        # Run prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(outputs, dim=1).item()
        
        probs_np = probs.cpu().numpy()[0]
        
        # Map prediction
        LABELS = {0: "Legitimate", 1: "Fraud", 2: "Abstain"}
        prediction_label = LABELS[pred_class]
        
        # Determine if the model should decide
        should_decide = pred_class != 2
        
        # Recommendation message
        if pred_class == 0:
            recommendation = (
                "The model predicts this transaction is legitimate with "
                f"{probs_np[0]*100:.2f}% confidence. "
                "The model has decided to commit to this prediction."
            )
        elif pred_class == 1:
            recommendation = (
                "The model predicts this transaction is fraudulent with "
                f"{probs_np[1]*100:.2f}% confidence. "
                "The model has decided to commit to this prediction. "
                "Immediate review is recommended."
            )
        else:
            recommendation = (
                "The model is uncertain about this transaction and has chosen to abstain. "
                f"Abstention confidence: {probs_np[2]*100:.2f}%. "
                "This case should be deferred to a human analyst for manual review."
            )
        
        return jsonify({
            "prediction": prediction_label,
            "prediction_code": pred_class,
            "confidence": {
                "legitimate": round(float(probs_np[0]), 6),
                "fraud": round(float(probs_np[1]), 6),
                "abstain": round(float(probs_np[2]), 6)
            },
            "should_decide": should_decide,
            "recommendation": recommendation
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/sample-data', methods=['GET'])
def sample_data():
    """
    Return sample test vectors for users to try.
    These are realistic values from the Credit Card dataset.
    """
    samples = [
        {
            "name": "Typical Legitimate Transaction",
            "description": "A standard transaction with normal PCA features and moderate amount.",
            "features": [
                43200.0,
                -1.36, -0.07, 2.54, 1.38, -0.34, -0.47, -0.42, 0.12,
                0.02, -0.23, -0.27, 0.50, -0.06, -0.22, 0.42, 0.89,
                -0.21, -0.03, -0.14, -0.12, 0.07, -0.01, -0.05, 0.83,
                -0.44, 0.43, 0.06, 0.01,
                65.32
            ]
        },
        {
            "name": "Suspicious Fraud-like Transaction",
            "description": "Features showing high deviation in PCA components, typical of anomalous activity.",
            "features": [
                406.0,
                -2.31, 1.95, -1.61, 3.99, -0.52, -1.43, -2.54, 1.39,
                -2.77, -2.77, 3.20, -2.90, -0.60, -4.29, 0.39, -1.14,
                -2.83, -0.02, 0.42, 0.13, -0.21, -0.26, -0.12, 0.01,
                -0.55, -0.24, -0.07, -0.06,
                0.00
            ]
        },
        {
            "name": "Edge Case (Uncertain)",
            "description": "Borderline features where the model may choose to abstain rather than commit.",
            "features": [
                75000.0,
                1.19, 0.27, 0.17, 0.45, 0.06, -0.08, -0.22, 0.08,
                -0.04, 0.58, -0.83, 0.26, 0.94, -0.47, -0.33, -0.27,
                0.10, -0.15, -0.43, -0.09, -0.06, -0.09, 0.00, -0.63,
                0.33, -0.12, 0.14, 0.07,
                231.48
            ]
        }
    ]
    
    return jsonify({
        "samples": samples,
        "feature_names": FEATURE_NAMES
    })


if __name__ == '__main__':
    load_model_and_scaler()
    app.run(host='0.0.0.0', port=5000, debug=True)
