# Risk-Aware Deep Learning with Abstention Mechanism
## Implementation Guide and Experimental Workflow

---

# 1. Project Overview

Traditional deep learning models always produce a prediction, even when the model is uncertain. In safety-critical systems such as healthcare, finance, and autonomous systems, incorrect predictions can lead to serious consequences.

To address this problem, **abstention learning** (also called **selective classification**) allows a model to refuse to make a prediction when its confidence is low.

Instead of always predicting:

```
Input → Prediction
```

the model can choose:

```
Input → Prediction OR Abstain
```

This improves reliability because uncertain cases can be forwarded to human experts or additional systems.

---

# 2. Objective of the Project

The main goals of this project are:

1. Train a **baseline deep learning model** for credit card fraud detection.
2. Extend the model with an **abstention mechanism** using the Deep Abstaining Classifier (DAC) framework.
3. Simulate **hardware-level training effects** such as:
   - Distributed training (via gradient accumulation)
   - Mixed precision training (via `torch.amp`)
4. Study how these changes affect:
   - Model accuracy and F1 score
   - Confidence calibration (ECE)
   - Risk-coverage behavior
   - Abstention reliability
5. Validate all findings with **multi-seed statistical testing** (3 seeds with paired t-tests).

---

# 3. Dataset

Dataset used:

**Credit Card Fraud Detection Dataset** (Kaggle — Pozzolo et al., 2015)

Characteristics:

- Total samples: **284,807 transactions**
- Fraud cases: **492 (~0.17%)**
- Features: **30 numerical features**
- Class imbalance ratio: **~1:578**

Features include:

```
V1–V28 → PCA transformed features (anonymized)
Time   → Seconds elapsed since first transaction
Amount → Transaction value in Euros
Class  → Target label (0 = legitimate, 1 = fraud)
```

---

# 4. Data Preprocessing

## Step 1: Load Dataset

The dataset is loaded using pandas from `data/creditcard.csv`.

## Step 2: Train / Validation / Test Split

We perform a **stratified split** to preserve the fraud ratio across all partitions.

| Dataset | Percentage | Approximate Size | Fraud Samples |
|---------|-----------|-----------------|---------------|
| Training | 70% | ~199,364 | ~344 |
| Validation | 15% | ~42,721 | ~74 |
| Test | 15% | ~42,722 | ~74 |

The split uses `random_state=42` for deterministic partitioning.

## Step 3: Feature Scaling

StandardScaler is applied:

```
X_scaled = (X − mean) / std
```

**Critical rule**: Scaler is fitted ONLY on training data, then applied to validation and test sets. The fitted scaler is persisted via `joblib.dump()` for API serving consistency.

---

# 5. Reproducibility Infrastructure

All random seeds are controlled via a centralized `seed.py` module:

| Source | Function | Impact |
|--------|----------|--------|
| `random.seed(s)` | Python built-in | Data augmentation randomness |
| `np.random.seed(s)` | NumPy | sklearn operations |
| `torch.manual_seed(s)` | PyTorch CPU | Weight initialization, dropout |
| `torch.cuda.manual_seed_all(s)` | PyTorch GPU | CUDA kernel randomness |
| `cudnn.deterministic = True` | cuDNN | Deterministic convolution |
| `cudnn.benchmark = False` | cuDNN | Prevents non-deterministic optimization |

Every training function accepts a `seed` parameter (default: 42).

---

# 6. Baseline Deep Learning Model

Architecture: **Multilayer Perceptron (MLP)**

```
Input Layer (30 features)
↓
Linear Layer (128 neurons) + BatchNorm + ReLU + Dropout(0.3)
↓
Linear Layer (64 neurons) + BatchNorm + ReLU + Dropout(0.3)
↓
Output Layer (2 neurons: Legitimate, Fraud)
```

Loss function: **Weighted CrossEntropyLoss** with class weights [1.0, 100.0]

Optimizer: **Adam** (lr = 0.0001, weight_decay = 1e-5)

---

# 7. Baseline Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 256 |
| Epochs | 60 (max) |
| Learning rate | 0.0001 |
| Optimizer | Adam (weight_decay=1e-5) |
| LR Scheduler | ReduceLROnPlateau (patience=3) |
| Early Stopping | Patience=10 |
| Gradient Clipping | Max norm 1.0 |
| Class Weights | [1.0, 100.0] |

---

# 8. Abstention Model

The baseline model is extended to include an **abstain class**:

```
Input Layer (30 features)
↓
Linear Layer (128 neurons) + BatchNorm + ReLU + Dropout(0.3)
↓
Linear Layer (64 neurons) + BatchNorm + ReLU + Dropout(0.3)
↓
Output Layer (3 neurons: Legitimate, Fraud, Abstain)
```

| Class | Meaning |
|-------|---------|
| 0 | Legitimate |
| 1 | Fraud |
| 2 | Abstain |

**Transfer learning**: Shared layers are initialized from the trained baseline model. Only layers with matching dimensions are copied; the final output layer (2→3 neurons) is randomly initialized.

### DAC Loss Function

```
L_DAC = w_class[y] · [-log(p_true + p_abstain) + α · p_abstain]
```

| Parameter | Value | Justification |
|-----------|-------|---------------|
| α (abstention penalty) | 0.3 | Balances coverage (~99.77%) with selective accuracy. See `methodology_decisions.md` §1. |
| Class weight (fraud) | 50.0 | Capped for DAC loss stability under extreme imbalance. See `methodology_decisions.md` §2. |

---

# 9. Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | correct / total (non-abstained) | Selective prediction correctness |
| Coverage | non-abstained / total | Fraction of inputs the model commits to |
| Selective Risk | errors / predicted samples | Error rate on committed predictions |
| F1 Score | 2·P·R/(P+R) (binary, pos=fraud) | Fraud detection performance |
| ECE | Σ |acc_bin − conf_bin| · bin_weight | Expected Calibration Error |
| Precision (Fraud) | TP / (TP + FP) | Fraud prediction precision |
| Recall (Fraud) | TP / (TP + FN) | Fraud detection recall |

Additionally, **confusion matrices** (2×2 on non-abstained predictions) and **abstention breakdowns** (how many legit vs fraud were abstained on) are computed for each model.

---

# 10. Hardware Simulation Experiments

We simulate production training conditions using gradient accumulation and mixed precision:

| Experiment | Configuration | Purpose |
|-----------|--------------|---------|
| Exp 1 | Standard DAC Training | Control |
| Exp 2 | Gradient Accumulation (batch=64, 4 steps) | Simulate distributed training |
| Exp 3 | Mixed Precision (via `torch.amp`) | Simulate reduced-precision training |
| Exp 4 | Combined (GA + MP) | Production-like scenario |

**Note on CPU execution**: On CPU, `torch.amp.autocast` uses **bfloat16** (not fp16). This still reduces mantissa precision (7 bits vs 23 bits in float32) but does not cause gradient underflow in the same way as GPU fp16. See `methodology_decisions.md` §4 for full analysis.

---

# 11. Multi-Seed Statistical Validation

All experiments are repeated across **3 independent random seeds** (42, 123, 256):

| Validation Step | Tool |
|----------------|------|
| Mean ± standard deviation | NumPy |
| Paired t-tests (5 key comparisons) | SciPy |

Key comparisons tested:
1. DAC vs Baseline (is abstention significantly better?)
2. Grad Accum vs Standard (is the collapse significant?)
3. Mixed Prec vs Standard (is the collapse significant?)
4. Combined vs Standard (is the recovery significant?)
5. Combined vs Grad Accum (is the recovery vs collapse significant?)

Results are saved to `results/multi_seed_summary.csv` and `results/statistical_tests.csv`.

---

# 12. Key Findings

### Finding 1: Abstention Improves Fraud Detection

The DAC mechanism improves F1 score by ~8.3% over the baseline while maintaining 99.77% coverage. The model abstains on only 0.23% of transactions.

### Finding 2: Abstention Collapse

Gradient accumulation and mixed precision, applied individually under 1:578 class imbalance, cause the model to achieve high accuracy but **F1 = 0.0**. The model routes all fraud to the abstain class — a degenerate solution.

### Finding 3: Combined Recovery

When GA and MP are combined, performance recovers to the best F1 score (~0.861). This counter-intuitive result suggests a mutual regularization effect.

### Finding 4: Calibration Improvement

The DAC model reduces ECE by ~6x compared to the baseline, producing more trustworthy confidence scores.

---

# 13. Complete Pipeline

```
Dataset (creditcard.csv)
↓
Preprocessing (StandardScaler, stratified split)
↓
Phase 1: Baseline Model Training (2-class MLP)
↓
Phase 2: Abstention Model Training (3-class DAC, transfer learning)
↓
Phase 3: Hardware Simulation Experiments (4 configurations)
↓
Phase 4: Comprehensive Evaluation (metrics + confusion matrices)
↓
Phase 5: Visualization (training curves, risk-coverage, hardware plots)
↓
Multi-Seed Validation (3 seeds, paired t-tests)
↓
Results Analysis & Statistical Reporting
```

---

# 14. Project Files

| File | Purpose |
|------|---------|
| `src/seed.py` | Centralized reproducibility seeds |
| `src/dataset.py` | Data loading, splitting, scaling |
| `src/baseline_model.py` | 2-class MLP architecture |
| `src/abstention_model.py` | 3-class MLP with abstain neuron |
| `src/train_baseline.py` | Phase 1: Baseline training |
| `src/train_abstention.py` | Phase 2: DAC training + loss function |
| `src/train_experiments.py` | Phase 3: Hardware simulation |
| `src/evaluation.py` | Phase 4: Metrics + confusion matrices |
| `src/plots.py` | Phase 5: Visualizations |
| `src/run_all.py` | Master pipeline orchestrator |
| `src/run_multi_seed.py` | Multi-seed validation runner |
| `api/app.py` | Flask REST API |
| `methodology_decisions.md` | Research backing document |
| `requirements.txt` | Pinned dependencies |

---

# 15. Final Outcome

This project demonstrates:

1. Abstention **improves prediction reliability** in safety-critical fraud detection.
2. Hardware optimizations are **not neutral** — they can silently degrade model performance.
3. The "abstention collapse" phenomenon is a **reproducible, statistically validated** finding.
4. Combined GA + MP provides a **regularization effect** that prevents collapse.
5. Proper validation requires **per-class metrics** (F1, precision, recall), not just aggregate accuracy.
