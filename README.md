<div align="center">

# 🛡️ Risk-Aware Deep Learning with Abstention Mechanism

### *"I Don't Know" is a Valid Answer*

A deep learning framework for credit card fraud detection that can **abstain from uncertain predictions**, improving reliability in safety-critical financial systems.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Overview

Traditional ML classifiers **must** produce a prediction for every input — even when highly uncertain. In fraud detection, this leads to:
- **False negatives**: Real fraud slipping through undetected
- **False positives**: Legitimate transactions being flagged unnecessarily

This project implements a **Deep Abstaining Classifier (DAC)** that adds a third option: **"I don't know."** When the model is unsure, it *abstains* and defers the decision to human analysts — dramatically improving the reliability of committed predictions.

### Key Results

| Model | Accuracy | Coverage | F1 Score | ECE |
|-------|----------|----------|----------|-----|
| Baseline (Standard MLP) | 99.93% | 100.00% | 0.789 | 0.0133 |
| **DAC Standard** | **99.95%** | **99.77%** | **0.855** | **0.0033** |
| DAC + GA + MP Combined | 99.96% | 99.75% | **0.861** | 0.0043 |

> **+8.3% F1 improvement** over baseline while maintaining 99.77% coverage (the model only abstains on 0.23% of transactions).

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    BASELINE MODEL (2-class)                   │
│                                                              │
│  Input(30) → Linear(128) → BN → ReLU → Dropout(0.3)        │
│           → Linear(64)  → BN → ReLU → Dropout(0.3)         │
│           → Linear(2)   → [Legitimate, Fraud]               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  ABSTENTION MODEL (3-class)                   │
│                                                              │
│  Input(30) → Linear(128) → BN → ReLU → Dropout(0.3)        │
│           → Linear(64)  → BN → ReLU → Dropout(0.3)         │
│           → Linear(3)   → [Legitimate, Fraud, ABSTAIN]      │
│                                          ↑                   │
│                                    "I don't know"            │
└──────────────────────────────────────────────────────────────┘
```

The abstention model uses the **DAC Loss Function**:

```
L_DAC = -log(p_true + p_abstain) + α · p_abstain
```

Where `α = 0.3` controls the cost of abstaining — lower values allow the model to abstain more freely.

---

## 📂 Project Structure

```
abstention/
├── data/
│   └── creditcard.csv          # Dataset (not included — see Setup)
├── src/
│   ├── dataset.py              # Data loading, splitting, scaling
│   ├── baseline_model.py       # 2-class MLP architecture
│   ├── abstention_model.py     # 3-class MLP with abstain neuron
│   ├── train_baseline.py       # Phase 1: Baseline training
│   ├── train_abstention.py     # Phase 2: DAC training + loss function
│   ├── train_experiments.py    # Phase 3: Hardware simulation experiments
│   ├── evaluation.py           # Phase 4: Comprehensive model evaluation
│   ├── metrics.py              # Coverage, selective risk, ECE calculations
│   ├── plots.py                # Phase 5: Matplotlib visualizations
│   └── run_all.py              # Master pipeline orchestrator
├── api/
│   └── app.py                  # Flask REST API for serving predictions
├── results/                    # Training metrics CSVs + plots (generated)
├── frontend/                   # Prediction UI (HTML/CSS/JS + Chart.js)
│   ├── index.html              # Dynamic prediction form & results
│   ├── css/style.css           # White & blue design system
│   ├── js/app.js               # API integration & rendering
│   ├── js/charts.js            # Dashboard chart logic
│   └── data/aggregate.py       # CSV → JSON converter
├── enhancements.md             # Production readiness roadmap
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/abstention.git
cd abstention
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision scikit-learn pandas matplotlib psutil
```

### 4. Download the Dataset

Download the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place `creditcard.csv` inside the `data/` folder:

```
data/
└── creditcard.csv
```

### 5. Run the Full Pipeline

```bash
cd src
python run_all.py
```

This executes all 5 phases sequentially:

| Phase | Description | Duration (approx.) |
|-------|------------|-------------------|
| 1 | Baseline Training | ~2 min |
| 2 | Abstention Training (DAC) | ~8 min |
| 3 | Hardware Simulation (4 experiments) | ~15 min |
| 4 | Comprehensive Evaluation | ~1 min |
| 5 | Visualization & Plotting | ~10 sec |

> ⏱️ **Total: ~30 minutes on CPU**

### 6. Install API Dependencies

```bash
pip install flask flask-cors
```

### 7. Launch the Prediction Frontend

You need **two terminals** — one for the API, one for the frontend:

**Terminal 1 — Flask API (port 5000):**
```bash
python api/app.py
```

**Terminal 2 — Frontend Server (port 8000):**
```bash
cd frontend
python -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🧪 Experimental Design

We run **4 hardware simulation experiments** to study how training optimizations affect abstention behavior:

| Experiment | Configuration | Purpose |
|-----------|--------------|---------|
| Exp 1 | Standard Training | Control |
| Exp 2 | Gradient Accumulation (batch=64, 4 steps) | Simulate distributed training |
| Exp 3 | Mixed Precision (FP16) | Simulate resource-constrained GPUs |
| Exp 4 | Combined (GA + MP) | Production-like scenario |

### 🔬 Key Finding: Abstention Collapse

We discovered that gradient accumulation and mixed precision, **when applied individually** under extreme class imbalance (1:578), cause the model to *collapse* — achieving 99.97% accuracy but **F1 = 0.0** for fraud detection. The model learns to route all fraud cases to the abstain class instead of predicting them.

Remarkably, when **combined** (Exp 4), performance recovers to F1 = 0.861, suggesting a mutual regularization effect.

> ⚠️ **Implication**: Hardware optimizations are **not neutral** — they must be validated with per-class metrics in safety-critical systems.

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|------------|
| **Accuracy** | Correctness on non-abstained predictions |
| **Coverage** | % of inputs the model commits to (not abstained) |
| **Selective Risk** | Error rate on committed predictions |
| **F1 Score** | Harmonic mean of precision & recall for fraud class |
| **ECE** | Expected Calibration Error — how well confidence matches accuracy |

---

## 📈 Results

### Full Comparison Table

| Model | Accuracy | Coverage | Sel. Risk | ECE | F1 Score |
|-------|----------|----------|-----------|-----|----------|
| Baseline | 99.93% | 100.00% | 0.075% | 0.0133 | 0.789 |
| Abstention (Phase 2) | 99.90% | 99.93% | 0.096% | 0.0021 | 0.752 |
| **Exp 1 (Standard)** | **99.95%** | **99.77%** | **0.047%** | 0.0033 | **0.855** |
| Exp 2 (Grad Accum) | 99.97% | 99.61% | 0.028% | 0.0025 | 0.000 |
| Exp 3 (Mixed Prec) | 99.97% | 99.63% | 0.028% | 0.0025 | 0.000 |
| **Exp 4 (Combined)** | **99.96%** | **99.75%** | **0.045%** | 0.0043 | **0.861** |

### Generated Visualizations

The pipeline automatically generates:
- `results/plot_training_curves.png` — Validation loss over epochs
- `results/plot_risk_coverage.png` — Risk vs Coverage tradeoff scatter
- `results/plot_hardware_throughput.png` — Throughput comparison
- `results/plot_hardware_memory.png` — Memory usage comparison

---

## 🌐 Prediction Frontend

A clean, professional prediction interface built with vanilla HTML/CSS/JS + Chart.js:

- **Dynamic Form** — All 30 feature inputs are generated dynamically from the API (no hardcoded values)
- **Sample Data Buttons** — Pre-built test vectors for Legitimate, Fraud, and Edge Case transactions
- **Confidence Visualization** — Animated progress bars + doughnut chart showing class probabilities
- **Prediction History** — Session-based table logging all predictions with timestamps
- **Testing Guide** — Built-in documentation explaining features, usage, and result interpretation

### Design

- **Color Palette**: White and blue (10-shade blue spectrum)
- **Typography**: Times New Roman (serif)
- **Layout**: Responsive grid, sticky header, smooth scroll navigation

### Prediction API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Model and scaler status check |
| `/api/features` | GET | Returns feature names dynamically |
| `/api/sample-data` | GET | Returns 3 realistic test vectors |
| `/api/predict` | POST | Accepts 30 features, returns prediction + confidence |

### Testing with Values

1. Click **"Typical Legitimate Transaction"** → Expect ~99.97% Legitimate
2. Click **"Suspicious Fraud-like Transaction"** → Expect ~99.67% Fraud
3. Click **"Edge Case (Uncertain)"** → Observe confidence distribution
4. Modify individual features (e.g., set V14 = -15.0) to explore model sensitivity

---

## 🔮 Future Enhancements

See [`enhancements.md`](enhancements.md) for a comprehensive production readiness roadmap covering:

- Model serving infrastructure (Docker, Gunicorn, model versioning)
- API robustness (input validation, rate limiting, logging)
- Model improvements (ensemble, calibration, drift detection)
- Frontend features (batch upload, SHAP explainability, dark mode)
- Security, monitoring, CI/CD, and compliance

---

## 📚 Dataset

**Credit Card Fraud Detection Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|----------|-------|
| Total Transactions | 284,807 |
| Fraud Cases | 492 (0.173%) |
| Features | 30 (V1–V28 PCA + Time + Amount) |
| Class Imbalance | ~1:578 |

> The dataset is **not included** in this repository due to its size (~150MB). Download it from Kaggle and place it in the `data/` folder.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Data Processing | Pandas, Scikit-learn |
| Visualization (Backend) | Matplotlib |
| Dashboard (Frontend) | HTML, CSS, JavaScript, Chart.js |
| Memory Profiling | psutil |

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{abstention2026,
  title={Risk-Aware Deep Learning with Abstention Mechanism for Fraud Detection},
  author={Your Name},
  year={2026},
  url={https://github.com/<your-username>/abstention}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using PyTorch**

</div>
