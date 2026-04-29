# Methodology Decisions — Research Backing Document

This document provides rigorous justification for every key methodological decision in the project. It serves as supplementary material for the research paper and can be referenced by reviewers.

---

## 1. Abstention Penalty α = 0.3

### Decision
The DAC loss function uses α = 0.3 as the abstention penalty coefficient.

### Justification
The DAC loss is defined as:

```
L_DAC = -log(p_true + p_abstain) + α · p_abstain
```

The parameter α controls the trade-off between prediction accuracy and abstention frequency:
- **α → 0**: Model abstains freely (high safety, low coverage)
- **α → ∞**: Model never abstains (behaves like standard classifier)

We selected α = 0.3 based on:

1. **Thulasidasan et al. (2019)** recommend α values in the range [0.1, 1.0] for most classification tasks. Their experiments on CIFAR-10 and SVHN use α values between 0.1 and 0.5.

2. **Empirical observation**: At α = 0.3, our model achieves a coverage of ~99.77% (abstaining on only 0.23% of transactions), which is operationally practical — a bank reviewing 0.23% of transactions manually is feasible.

3. **Sensitivity**: Values below 0.2 cause excessive abstention (>5% of transactions deferred), which is impractical for high-throughput payment processing. Values above 0.5 suppress abstention almost entirely, defeating the purpose.

### Limitation
We did not perform an exhaustive hyperparameter sweep over α. Future work should include an ablation study varying α ∈ {0.1, 0.2, 0.3, 0.4, 0.5} and reporting the risk-coverage Pareto frontier.

---

## 2. Class Weight Cap = 50.0

### Decision
All DAC training phases (Phase 2 abstention + Experiments 1–4) use a fraud class weight of `min(N_legit / N_fraud, 50.0) = 50.0`.

The baseline model (Phase 1) uses `min(N_legit / N_fraud, 100.0) = 100.0` with standard CrossEntropyLoss.

### Justification

1. **Why not use the raw ratio (~578)?**
   
   The raw class imbalance ratio is approximately 578:1 (284,315 legitimate vs 492 fraud). Using this directly as a class weight in the DAC loss causes **training instability**:
   - The combined effect of the DAC loss penalty term (α · p_abstain) and an extreme class weight (578×) on fraud samples creates contradictory gradient signals: the weight pushes the model to predict fraud aggressively, while the DAC penalty encourages conservative abstention.
   - This manifests as oscillating validation loss and failure to converge within 60 epochs.

2. **Why 50.0 specifically?**
   
   - A cap of 50.0 provides ~28× the gradient signal for fraud samples compared to legitimate ones — sufficient to prevent the model from ignoring the rare class.
   - At 50.0, the DAC loss gradient from a single fraud sample is 50× its natural weight, which is strong enough to counterbalance the abstention penalty but not so strong as to destabilize training.
   - Empirically, values between 30 and 100 produce comparable results; 50 was selected as a round midpoint.

3. **Why is the baseline different (100.0)?**
   
   The baseline uses standard CrossEntropyLoss without the DAC abstention penalty. Without the competing gradient from p_abstain, higher class weights are stable. The cap of 100.0 is conservative (vs. the raw 578) to prevent gradient explosion, following best practices for weighted cross-entropy on imbalanced data.

4. **Why unified weights across DAC phases?**
   
   Using different class weights between Phase 2 (abstention training) and Phase 3 (experiments) would make cross-phase comparisons scientifically invalid. Experiment 1 is intended as a **control** — identical to Phase 2 but independently trained. If the class weights differ, any performance difference could be attributed to the weight change rather than the experimental variable (gradient accumulation, mixed precision).

### References
- Cui, Y., et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples." CVPR.
- Lin, T.-Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.

---

## 3. Multi-Seed Validation (3 Seeds)

### Decision
All experiments are repeated with 3 independent random seeds (42, 123, 256) and results are reported as mean ± standard deviation with paired t-tests.

### Justification

1. **Why multiple seeds?**
   
   Deep learning training is non-deterministic in several ways: weight initialization, data shuffling order, and dropout masks all depend on random seeds. A single-run result may represent an outlier rather than the expected behavior. Our most critical finding — the "abstention collapse" under gradient accumulation and mixed precision — must be validated as a consistent phenomenon, not a random artifact.

2. **Why 3 seeds (not 5 or 10)?**
   
   - The full pipeline takes approximately 30 minutes per seed on CPU, totaling ~90 minutes for 3 seeds. This is a practical compromise between statistical power and computational budget.
   - For the abstention collapse finding (F1 = 0.0 vs F1 ≈ 0.85), the effect size is extremely large (Cohen's d >> 2.0), meaning even 3 samples provide high statistical power (>0.99) to detect the difference via paired t-test.
   - For more subtle effects (e.g., comparing Exp 1 F1 = 0.855 vs Exp 4 F1 = 0.861), 3 seeds may not provide sufficient power. We acknowledge this limitation and note that a full study would use 10+ seeds.

3. **Statistical test choice: Paired t-test**
   
   We use the paired (dependent) t-test rather than the independent t-test because:
   - Each seed produces a matched pair of observations (same data split → same test set)
   - Pairing controls for seed-specific variance in data shuffling
   - With 3 pairs, we have 2 degrees of freedom — marginal but acceptable for large effect sizes

### Limitation
Three seeds provide limited statistical power for subtle differences. We report effect sizes alongside p-values so readers can assess practical significance independently.

---

## 4. CPU vs GPU Mixed Precision

### Decision
All experiments were conducted on CPU. The mixed precision experiments (Exp 3, Exp 4) use `torch.amp.autocast` on CPU.

### What Actually Happens on CPU

When `torch.amp.autocast(device_type='cpu')` is invoked:
- PyTorch uses **bfloat16** (Brain Floating Point), NOT fp16 (IEEE half-precision)
- bfloat16 has the same exponent range as float32 (8 bits) but reduced mantissa precision (7 bits vs 23 bits)
- The `GradScaler` is technically a no-op on CPU because bfloat16 does not suffer from the same underflow issues as fp16

### Implications for Our Findings

1. **The abstention collapse phenomenon is still valid** — reduced numerical precision (7-bit mantissa vs 23-bit) demonstrably affects gradient computation, particularly for extremely rare events (0.17% fraud rate).

2. **However, the mechanism differs from GPU fp16**:
   - On GPU with fp16: gradients can literally underflow to zero (fp16 minimum positive value is ~6×10⁻⁸)
   - On CPU with bfloat16: gradients lose precision but do not underflow (bfloat16 minimum is ~1.2×10⁻³⁸, same as float32)

3. **The correct characterization** is: "reduced precision training alters gradient dynamics for minority classes under extreme imbalance, causing the model to converge to degenerate solutions." The specific mechanism (underflow vs precision loss) depends on the hardware.

### Recommendation
Future work should replicate these experiments on GPU with fp16 to confirm whether the collapse is more severe (due to actual gradient underflow) or is a general property of reduced-precision arithmetic.

---

## 5. Transfer Learning Strategy

### Decision
The abstention model (3-class) initializes its shared layers from the pre-trained baseline model (2-class). Only layers with matching shapes are transferred; the final output layer (2→3 neurons) is randomly initialized.

### Justification

1. **Why transfer from baseline?**
   - The baseline model has already learned useful feature representations for distinguishing legitimate from fraudulent transactions
   - Starting from these weights gives the abstention model a warm start, reducing training time and improving convergence
   - The shared layers (Linear 30→128, BatchNorm, Linear 128→64, BatchNorm) are identical in both architectures

2. **What is transferred?**
   - `net.0.weight`, `net.0.bias` (Linear 30→128)
   - `net.1.weight`, `net.1.bias`, `net.1.running_mean`, `net.1.running_var`, `net.1.num_batches_tracked` (BatchNorm1d)
   - `net.4.weight`, `net.4.bias` (Linear 128→64)
   - `net.5.weight`, `net.5.bias`, `net.5.running_mean`, `net.5.running_var`, `net.5.num_batches_tracked` (BatchNorm1d)
   - Total: 10 parameter tensors transferred

3. **What is NOT transferred?**
   - `net.8.weight` (64→2 in baseline, 64→3 in abstention — shape mismatch)
   - `net.8.bias` (size 2 vs size 3 — shape mismatch)
   - These are initialized randomly using PyTorch's default Kaiming uniform initialization

---

## 6. Data Split Rationale (70/15/15)

### Decision
The dataset is split into 70% train, 15% validation, 15% test using stratified sampling.

### Justification

1. **Stratified split**: Critical for extremely imbalanced data. With only 492 fraud samples, a non-stratified split could result in a test set with zero fraud cases. Stratification guarantees proportional representation (≈0.17% fraud) in each split.

2. **70/15/15 ratio**: Provides ~344 fraud samples for training, ~74 for validation, and ~74 for test. While 74 fraud test samples is small, it enables meaningful F1 computation. Common alternatives:
   - 80/10/10 would give only ~49 fraud test samples — too few for reliable metrics
   - 60/20/20 would give ~98 fraud test samples but reduce training data

3. **Fixed random_state=42**: Ensures the same data split across all experiments, enabling fair comparison. Combined with our multi-seed approach, this means the test set is constant; only training dynamics vary.

---

## 7. DAC Loss Formulation

### Decision
We use the DAC loss from Thulasidasan et al. (2019) with class-weighted extension:

```
L_DAC = w_class[y] · [-log(p_y + p_abstain) + α · p_abstain]
```

### Justification

1. **Why this formulation (not confidence thresholding)?**
   - Post-hoc confidence thresholding (reject if max_prob < threshold) requires a separate calibration step and doesn't optimize for selective classification during training
   - The DAC loss directly trains the model to learn when to abstain, producing better-calibrated uncertainty estimates
   - The abstention probability is a learned output, not an ad-hoc threshold

2. **Why class-weighted DAC loss?**
   - Standard (unweighted) DAC loss treats all misclassifications equally
   - Under 578:1 imbalance, the gradient contribution from fraud samples is negligible
   - Without weighting, the model learns a trivial solution: predict legitimate for everything, abstain on nothing → high accuracy, zero fraud detection
   - Weighting multiplies each fraud sample's loss by 50×, ensuring the model cannot ignore them

3. **Validation loss consistency**
   - We apply the same class weights during validation loss computation as during training
   - This ensures the loss metric being monitored (for early stopping and LR scheduling) reflects the same objective the model is optimizing
   - Previous versions had an inconsistency where validation loss omitted class weights, causing early stopping to trigger at suboptimal points

---

## 8. Reproducibility Guarantees

### Seeds Set
Our `set_seed()` function controls randomness at all levels:

| Source | Function | Impact |
|--------|----------|--------|
| `random.seed(s)` | Python built-in | Data augmentation, any Python randomness |
| `np.random.seed(s)` | NumPy | sklearn operations, any array randomness |
| `torch.manual_seed(s)` | PyTorch CPU | Weight initialization, dropout masks |
| `torch.cuda.manual_seed_all(s)` | PyTorch GPU | CUDA kernel randomness |
| `cudnn.deterministic = True` | cuDNN | Convolution algorithm selection |
| `cudnn.benchmark = False` | cuDNN | Prevents non-deterministic optimization |

### Data Split
- `sklearn.model_selection.train_test_split` with `random_state=42` (constant)
- Guarantees identical train/val/test partitions across all seeds
- Only training dynamics (weight init, batch order) vary between seeds

### What Is NOT Controlled
- PyTorch DataLoader with `shuffle=True` uses the torch generator (controlled by `torch.manual_seed`)
- Floating-point non-associativity in parallel reduction (minimal impact on CPU)
