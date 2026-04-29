# Project Presentation Speech: Risk-Aware Deep Learning with Abstention

## Opening (Hook & Problem Statement)

*"Good morning, esteemed faculty members.*

*Imagine a bank deploying an advanced AI to catch credit card fraud. Every millisecond counts. However, traditional machine learning classifiers operate under a dangerous constraint: they are conceptually 'forced' to make a prediction on every single transaction, even when they are highly uncertain. This 'forced prediction' paradigm creates two massive issues in safety-critical domains:*

1. *False Positives: A legitimate purchase is declined, infuriating the customer.*
2. *False Negatives: Fraud slips through, costing the bank money and reputation.*

*My project challenges this constraint. What if the model could simply say 'I don't know'?"*

---

## 2. Our Solution: The Deep Abstaining Classifier (DAC)

*"To solve this, we implemented a **Risk-Aware Deep Learning Framework** using a **Deep Abstaining Classifier (DAC)**.*

*Rather than a binary 'Legitimate' or 'Fraud' output, we added a third, explicit class: **Abstain**. When the model encounters a highly ambiguous transaction, it defers the decision to a human analyst rather than risking a blind guess.*

*This is controlled mathematically via a specialized 'DAC Loss Function':*

```
L_DAC = w · [-log(p_true + p_abstain) + α · p_abstain]
```

*Where α = 0.3 balances coverage against abstention cost, and class weights (w = 50.0 for fraud) ensure the model cannot ignore the rare fraud class. We trained this on a highly imbalanced dataset of nearly 285,000 transactions, where only 0.17% were actual fraud cases.*

*Crucially, we ensured all findings are reproducible — every training run uses controlled random seeds, and all results are validated across 3 independent seeds with paired t-tests."*

---

## 3. The Key Results

*"The results were outstanding. Let's look at the numbers:*

* *Our standard baseline model achieved an F1 score of 0.789.*
* *By bringing in the abstention mechanism, the **F1 score improved to 0.855** — an 8.3% improvement in fraud detection capability.*
* *More importantly, the model achieved a **Selective Accuracy of 99.95%** while retaining **99.77% Coverage**. This means it only needed to defer 0.23% of transactions to human reviewers.*
* *The model's confidence scores became significantly more trustworthy, reducing the Expected Calibration Error (ECE) by over 6 times compared to the baseline.*
* *We validated these results across 3 random seeds, confirming the improvements are statistically significant, not artifacts of random initialization."*

---

## 4. The Novel Discovery: "Abstention Collapse"

*"However, the most fascinating takeaway from this research emerged when we simulated real-world hardware constraints.* 

*We experimented with **Gradient Accumulation** (used in distributed training) and **Mixed Precision** (used to reduce computation cost).* 

*When we applied either of these optimizations **individually**, the model catastrophically failed. It achieved 99.97% accuracy but an **F1 score of 0.0**. It learned to 'game' the system by classifying all clear cases as legitimate and routing all potential fraud to the 'Abstain' class. It became completely blind to fraud.*

*We validated this collapse across all 3 seeds — it was consistent and reproducible, not a random fluke.*

*But, quite surprisingly, when we **combined** both Gradient Accumulation and Mixed Precision, the performance fully recovered, achieving our best F1 score of 0.861. They acted as mutual regularizers.*

*This tells us a crucial cautionary tale for MLOps: **hardware-level optimizations are not neutral**. Deploying standard optimizations without validating per-class metrics in safety-critical systems can lead to catastrophic, silent failures."*

---

## 5. Methodology Rigor

*"A few words on how we ensured scientific validity:*

* *Every training run uses **deterministic random seeds** — torch, numpy, random, and cuDNN are all controlled.*
* *All DAC training phases use **identical class weights** (50.0 for fraud), making cross-experiment comparisons fair.*
* *Results are validated across **3 independent seeds** (42, 123, 256) with **paired t-tests** confirming statistical significance.*
* *We include **confusion matrices** and **per-class metrics** (precision, recall) for each model, not just aggregate accuracy.*
* *The full methodology is documented in our `methodology_decisions.md` covering every hyperparameter choice with citations."*

---

## 6. Live Demonstration

*(Transition to the running Frontend Dashboard)*

*"To demonstrate this in action, we built a fully dynamic, real-time prediction dashboard served by a Flask API.*

1. **[Click 'Typical Legitimate Transaction']** *Here is a standard transaction. The model rapidly processes 30 features and confidently commits to 'Legitimate'.*
2. **[Click 'Suspicious Fraud-like Transaction']** *Here, we introduce extreme deviations in the PCA features typical of anomalies. The model commits firmly to 'Fraud'.*
3. **[Click 'Edge Case (Uncertain)']** *But watch what happens with an ambiguous transaction. Instead of forcing a bad guess, the model's confidence distributes across categories, and it outputs **'Abstain'**. It intentionally defers the decision, effectively saying: 'Human review recommended.'*

*"This proves that 'I don't know' is not a failure of the AI; in safety-critical systems, it is the safest and most reliable valid answer."*

*(Pause for Questions)*
