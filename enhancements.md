# Future Enhancements — Production Readiness Roadmap

A comprehensive list of improvements to take the Abstention Classifier from a research prototype to a production-grade fraud detection system.

---

## 1. Model Serving & Infrastructure

### 1.1 Production WSGI/ASGI Server
Replace the Flask development server with a production-grade server.
- Use **Gunicorn** (Linux) or **Waitress** (Windows) behind **Nginx** as a reverse proxy.
- Implement worker-based concurrency to handle multiple simultaneous prediction requests.
- Configure proper timeout, keep-alive, and connection limits.

### 1.2 Model Versioning & Registry
- Integrate **MLflow** or **DVC** to version-control trained models alongside their training hyperparameters and metrics.
- Maintain a model registry with staging, production, and archived model states.
- Enable one-command rollback to a previous model version if a deployed model degrades.

### 1.3 Containerization
- Package the API and frontend into a single **Docker** image with a multi-stage build.
- Create a `docker-compose.yml` for local development with hot-reload.
- Publish images to a private container registry (e.g. GitHub Container Registry, AWS ECR).

### 1.4 Scaler Persistence
- Serialize the fitted `StandardScaler` to disk using `joblib` during training, instead of re-fitting it from the CSV on every API startup.
- This eliminates the dependency on the raw dataset being present at serving time and reduces startup latency from ~10 seconds to under 1 second.

---

## 2. API Robustness

### 2.1 Input Validation & Sanitization
- Add strict schema validation using **Pydantic** or **Marshmallow** for incoming JSON payloads.
- Enforce feature value range bounds based on training data statistics (e.g., Time between 0 and 172792, Amount between 0 and 25691.16).
- Return descriptive error messages identifying exactly which field failed validation.

### 2.2 Rate Limiting & Throttling
- Implement per-IP rate limiting using **Flask-Limiter** (e.g., 60 requests/minute per client).
- Add API key authentication for programmatic access.
- Protect against abuse from automated flood requests.

### 2.3 Request/Response Logging
- Log every prediction request and response with timestamps, input feature hashes, and model output.
- Store logs in a structured format (JSON lines) for downstream analysis.
- Mask or hash sensitive feature values in logs to comply with data privacy requirements.

### 2.4 Health Checks & Readiness Probes
- Expand the `/api/health` endpoint to verify model inference latency (run a dummy prediction).
- Add a `/api/ready` endpoint that confirms the scaler and model are fully loaded before accepting traffic.
- Integrate with orchestrator health checks (Kubernetes liveness/readiness probes).

---

## 3. Model Performance & Reliability

### 3.1 Ensemble Prediction
- Serve multiple experiment models (Exp 1, Exp 4) simultaneously and aggregate their predictions via majority vote or confidence averaging.
- Ensemble predictions reduce variance and improve reliability on edge cases where a single model might misclassify.

### 3.2 Confidence Calibration
- Apply **Platt Scaling** or **Temperature Scaling** post-training to calibrate softmax probabilities so that a 90% confidence actually corresponds to 90% accuracy.
- The current model has an ECE of 0.0033, which is already low, but post-hoc calibration further strengthens trust in the confidence scores.

### 3.3 Adaptive Abstention Threshold
- Instead of using `argmax` directly, allow the abstention threshold to be configurable via an API parameter (e.g., `?abstain_threshold=0.15`).
- This lets operators adjust the risk tolerance in real-time: lower thresholds mean more abstentions (safer), higher thresholds mean fewer (more coverage).

### 3.4 Feature Drift Detection
- Track the statistical distribution of incoming features over time using **Evidently AI** or similar tools.
- Alert when live data drifts significantly from training data distribution, which signals model degradation.
- Automate re-training triggers when drift exceeds configurable thresholds.

---

## 4. Frontend Enhancements

### 4.1 Batch Prediction Upload
- Add a CSV file upload component that accepts multiple transactions at once.
- Display results in a sortable, filterable table with color-coded risk indicators.
- Allow CSV download of batch results with all confidence scores included.

### 4.2 Feature Importance Visualization
- Integrate **SHAP** (SHapley Additive Explanations) into the backend to compute per-prediction feature contributions.
- Display a horizontal bar chart on the result panel showing which features pushed the prediction toward Fraud vs. Legitimate.
- This provides explainability, which is a regulatory requirement in many financial jurisdictions.

### 4.3 Real-Time Dashboard
- Add a live metrics panel showing:
  - Total predictions served today
  - Fraud detection rate
  - Abstention rate
  - Average inference latency
- Use **WebSockets** or **Server-Sent Events** for real-time updates.

### 4.4 Dark Mode
- Add a light/dark mode toggle that persists user preference in `localStorage`.
- Maintain the blue accent palette in both modes with adjusted background and text contrast.

### 4.5 Responsive Mobile Layout
- Optimize the feature input grid for single-column layout on mobile screens.
- Implement touch-friendly sample data buttons and swipeable result cards.

---

## 5. Security

### 5.1 API Authentication
- Implement **JWT-based authentication** with access and refresh tokens.
- Add role-based access control: `viewer` (can predict) vs. `admin` (can view history, adjust thresholds).
- Protect sample data and model metadata endpoints behind authentication.

### 5.2 HTTPS & CORS Hardening
- Enforce HTTPS-only communication using TLS certificates (Let's Encrypt or organizational CA).
- Restrict CORS origins to the specific frontend domain instead of wildcard `*`.
- Add `Content-Security-Policy`, `X-Frame-Options`, and `X-Content-Type-Options` headers.

### 5.3 Input Adversarial Protection
- Validate that input features are within physically plausible ranges.
- Detect and reject adversarial inputs designed to manipulate model predictions (e.g., gradient-based attacks that flip fraud to legitimate).
- Implement anomaly scoring on inputs before they reach the model.

---

## 6. Data Pipeline & Retraining

### 6.1 Automated Retraining Pipeline
- Build a scheduled pipeline (weekly or triggered by drift detection) that:
  1. Pulls new labeled transaction data
  2. Re-trains the baseline and abstention models
  3. Evaluates on the held-out test set
  4. Promotes to production only if metrics improve over the current model
- Use **Apache Airflow**, **Prefect**, or **GitHub Actions** for orchestration.

### 6.2 Data Versioning
- Version all training datasets alongside model weights using **DVC** or **LakeFS**.
- Ensure full reproducibility: any model checkpoint can be traced back to the exact dataset and hyperparameters that produced it.

### 6.3 Label Feedback Loop
- Build a human review interface where analysts label transactions the model abstained on.
- Feed confirmed labels back into the training pipeline to improve the model's understanding of edge cases over time.
- Track the distribution of analyst overrides to identify systematic model weaknesses.

---

## 7. Monitoring & Observability

### 7.1 Prediction Monitoring
- Track prediction distribution over time (% Legitimate, % Fraud, % Abstain).
- Set alerts when the abstention rate spikes (could indicate data drift) or fraud rate drops to zero (could indicate a model collapse like the one observed in Exp 2 and Exp 3).

### 7.2 Latency & Throughput Metrics
- Instrument the API with **Prometheus** metrics: request latency (p50, p95, p99), throughput (req/sec), error rate.
- Build **Grafana** dashboards for real-time monitoring.
- Set SLOs: e.g., p99 prediction latency under 100ms, availability above 99.9%.

### 7.3 Audit Trail
- Maintain an immutable log of every prediction with:
  - Input features (hashed for privacy)
  - Model version used
  - Prediction output and confidence scores
  - Timestamp
- This is required for regulatory compliance in financial fraud detection systems (PCI DSS, PSD2).

---

## 8. Testing & CI/CD

### 8.1 Automated Test Suite
- **Unit tests**: Model loading, scaler transformation, prediction output shape.
- **Integration tests**: End-to-end API call with known inputs and expected outputs.
- **Frontend tests**: Playwright/Selenium tests for form fill, prediction submission, and result rendering.
- **Regression tests**: Golden test vectors that must produce the same predictions across model updates.

### 8.2 CI/CD Pipeline
- On every push:
  1. Run linting (flake8, ESLint)
  2. Run unit and integration tests
  3. Build Docker image
  4. Deploy to staging environment
  5. Run smoke tests against staging
  6. Promote to production on manual approval
- Use **GitHub Actions** or **GitLab CI** as the orchestrator.

### 8.3 Load Testing
- Use **Locust** or **k6** to simulate concurrent prediction requests.
- Establish baseline throughput: e.g., single worker handles 200 predictions/second.
- Identify bottlenecks (model inference, scaler transform, JSON serialization).

---

## 9. Compliance & Documentation

### 9.1 Model Card
- Create a standardized model card documenting:
  - Training data description and limitations
  - Performance metrics across demographic subgroups (if applicable)
  - Known failure modes (e.g., Exp 2/3 abstention collapse under gradient accumulation)
  - Intended use cases and out-of-scope applications

### 9.2 API Documentation
- Generate OpenAPI/Swagger documentation from the Flask routes.
- Host interactive API docs at `/api/docs` for developers.
- Include request/response examples for each endpoint.

### 9.3 Regulatory Compliance
- Document compliance with relevant financial regulations:
  - **PSD2** (EU) — Strong Customer Authentication requirements
  - **PCI DSS** — Cardholder data protection standards
  - **GDPR** — Data subject rights for any personal data involved
- Maintain records of model decisions for auditability.

---

## Priority Matrix

| Priority | Enhancement | Impact | Effort |
|----------|-------------|--------|--------|
| High | Scaler persistence (1.4) | Eliminates CSV dependency at startup | Low |
| High | Input validation (2.1) | Prevents malformed requests | Low |
| High | HTTPS & CORS hardening (5.2) | Security baseline | Low |
| High | Automated test suite (8.1) | Catches regressions | Medium |
| Medium | Docker containerization (1.3) | Deployment portability | Medium |
| Medium | Feature importance / SHAP (4.2) | Explainability / compliance | Medium |
| Medium | Prediction monitoring (7.1) | Detects model degradation | Medium |
| Medium | Batch prediction (4.1) | Operational efficiency | Medium |
| Low | Ensemble prediction (3.1) | Marginal accuracy gain | High |
| Low | Adaptive threshold (3.3) | Operational flexibility | Low |
| Low | Real-time dashboard (4.3) | Visibility | High |
| Low | Retraining pipeline (6.1) | Long-term reliability | High |
