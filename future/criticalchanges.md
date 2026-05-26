# Future Work & Critical Roadmap: Agentic Abstention Governance

This document lists the recommended future research and critical changes designed to elevate the **Risk-Aware Deep Learning with Abstention Mechanism** framework to the standards of high-impact peer-reviewed venues (such as IEEE Transactions on Neural Networks and Learning Systems, IEEE Access, or top-tier AI conferences like AAAI and NeurIPS).

---

## 1. Methodological and Theoretical Enhancements

### A. Non-Stationary Temporal (Time-Based) Splitting
* **Current Approach**: The dataset uses a random stratified split to evaluate baseline and Deep Abstaining Classifier (DAC) performance.
* **Critical Change**: Tabular fraud datasets (such as IEEE-CIS) exhibit extreme temporal non-stationarity and distribution drift. Reviewers will demand a temporal validation strategy:
  - **Proposed Implementation**: Split the data chronologically based on the `TransactionDT` column (e.g., training on the first 4 months, validating on the 5th month, and testing on the final month). This simulates a realistic "train on the past, test on the future" setting and prevents any lookahead bias in model evaluation.

### B. Mathematical Conformal Gating inside RL Policies
* **Current Approach**: Conformal prediction is applied post-hoc during evaluation in `evaluation.py` to calculate set sizes and coverage guarantees.
* **Critical Change**: Integrate conformal guarantees directly into the active decision pipeline:
  - **Proposed Implementation**: Feed the conformal set prediction size or conformity scores as active features into the Reinforcement Learning `DecisionAgent`'s state representation. This allows the gating policy to learn an optimal action strategy that mathematically respects preset confidence thresholds (e.g., maintaining selective risk under 5%).

### C. Comprehensive Pareto Frontier Ablations on $\alpha$
* **Current Approach**: The cost-penalty parameter $\alpha$ is fixed to a static value of `0.3` across baseline training and hardware experiments.
* **Critical Change**: Map the explicit Risk-Coverage trade-off:
  - **Proposed Implementation**: Conduct a sweep over $\alpha \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$ to construct the Risk-Coverage Pareto frontier. This will quantify the precise economic exchange rate between false negatives (leaked fraud) and cost of manual reviews (abstention queue size).

---

## 2. Model & Agentic Governance Enhancements

### A. Upgrading from Heuristic Mocks to Local SLM Inference
* **Current Approach**: The Reflection Agent and the verbalized context confidence score are mocked or use static placeholder indicators to avoid model loading delays.
* **Critical Change**: Integrate a real, local Small Language Model (SLM) to perform explainable human-in-the-loop reasoning:
  - **Proposed Implementation**: Standardize on a quantized local model (e.g., `Qwen-1.5B-Instruct` or `Llama-3-8B-Instruct` via Ollama or Hugging Face Transformers). Let the agent process tabular feature summaries (e.g., "Transaction amount is $5,000, user transaction velocity is 10x higher than their 30-day average") and output a natural language justification for the abstention decision.

### B. Advanced Selective Classification Baselines
* **Current Approach**: The framework evaluates against standard Neural Network output and Maximum Softmax Probability (MSP) confidence thresholding.
* **Critical Change**: Reviewers in selective classification will expect comparison with state-of-the-art baselines:
  - **Proposed Implementation**: Implement and compare against:
    1. **SelectiveNet** (Geifman & El-Yaniv, 2019) which optimizes coverage and selective risk end-to-end.
    2. **Monte Carlo Dropout Entropy-based thresholding** (applying thresholding on the predictive entropy of 10 forward passes instead of a single softmax probability).

### C. Scaling Statistical Validation (Multi-Seed Paired t-Tests)
* **Current Approach**: Statistical validation is executed across 3 seeds (`42`, `123`, `256`).
* **Critical Change**: Increase the sample size of random seeds to improve statistical confidence:
  - **Proposed Implementation**: Expand the multi-seed pipeline to **10 independent seeds**. Report paired t-tests and Cohen's $d$ effect sizes for baseline vs. DAC, and DAC vs. RL Gating, to guarantee that the results are not statistical anomalies.

---

## 3. Engineering & Production Enhancements

### A. Dynamic Sliding Window Behavioral Profiles
* **Current Approach**: The `PerceptionAgent` computes causal cumulative features (`cumcount` and cumulative sum/mean) over the entire historical sequence.
* **Critical Change**: Replace cumulative sum with sliding time windows to model short-term behavior:
  - **Proposed Implementation**: Implement rolling time-window aggregations (e.g., 24-hour transaction frequency, 7-day average transaction amount, 30-day billing country variance) utilizing pandas rolling intervals or DuckDB SQL queries to optimize processing.

### B. Production Serving Infrastructure & Safeguards
* **Current Approach**: A basic Flask API (`api/app.py`) serves predictions locally on Port 5000.
* **Critical Change**: Harden the API for production deployment:
  - **Proposed Implementation**:
    - **Containerization**: Provide a multi-stage `Dockerfile` and a `docker-compose.yml` to orchestrate the Flask API and the JS/Chart.js frontend.
    - **API Gateway**: Set up rate limiting (e.g., using `Flask-Limiter`) and request validation using Pydantic schemas.
    - **Explainability**: Integrate SHAP (SHapley Additive exPlanations) or LIME into the frontend prediction dashboard, providing visual feature importance bar charts next to the model's confidence scores.
