"""
Dockerfile — Agentic Abstention Governance API
================================================
Multi-stage build for the Flask prediction API.

Stage 1: Build — install Python dependencies
Stage 2: Runtime — copy only what's needed for serving

Usage:
    # Build
    docker build -t abstention-api:latest .

    # Run (requires trained model + data/model_metadata.json to be mounted)
    docker run -p 5000:5000 \
        -v $(pwd)/abstention_model.pth:/app/abstention_model.pth:ro \
        -v $(pwd)/data/model_metadata.json:/app/data/model_metadata.json:ro \
        abstention-api:latest
"""

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for numpy/torch compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch for minimal image size
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.3.0+cpu \
        torchvision==0.18.0+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY api/        ./api/
COPY src/        ./src/
COPY agents/     ./agents/

# Ensure data dir exists (model files are mounted at runtime)
RUN mkdir -p /app/data /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

EXPOSE 5000

# Use waitress for production serving
CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=5000", "api.app:app"]
