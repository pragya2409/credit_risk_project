## Week 1: Environment, data and EDA

**Goal.** Get a reproducible local pipeline running and understand the data before modelling.

**Setup.** PyCharm with a Python 3.14 virtualenv on a MacBook Air M2 (no GPU, so every component is sized to run locally). Git with a GitHub remote over SSH; `data/`, `.venv/` and `.idea/` are gitignored. The Kaggle API is used only to download the data.

**Data.** Home Credit Default Risk. The main table has ~307K labeled applications and 122 columns. Auxiliary tables (`bureau_balance`, `installments_payments`, `POS_CASH_balance`) are held back for the Week 11 transition-probability work, because they contain real monthly repayment-status histories.

**Findings.**
- Severe class imbalance: 91.9% non-default vs 8.1% default. Accuracy is meaningless here, so the primary metrics are ROC-AUC (threshold-independent, rank-based) and PR-AUC (more sensitive to the minority class).
- Heavy missingness in the building/apartment feature block (up to ~70%). Treatment differs by model: LightGBM and XGBoost handle NaNs natively; Logistic Regression needs explicit imputation.
- Every standard public credit dataset contains only approved applicants. The rejected population is unobserved, which motivates the reject-inference work in Week 6 and is stated as a limitation throughout.

**Design decision.** The professor requires 2-3 datasets. Plan: Home Credit for the core pipeline, LendingClub as a cross-domain base learner and external check, and the UPI transactions dataset to parameterise the simulator. Row-level merging was rejected in favour of methodologically justified roles for each dataset.

---

## Weeks 2-3: Individual baselines

**Goal.** Establish honest single-model numbers that the stacked ensemble must beat.

**Preprocessing.**
- Logistic Regression: drop columns above a missingness threshold (~50-60%) - Logistic Regression: drop columns with 50% or more missing values, add a binary `_MISSING` indicator flag for each remaining column that still has missing values, median imputation, `StandardScaler`.
- LightGBM / XGBoost: raw features, NaNs left as they are.
- Categoricals one-hot encoded.

**Results (test AUC).**
| Model | AUC | Note |
|---|---|---|
| Logistic Regression | 0.7469 | |
| LightGBM | 0.7520 | default params |
| XGBoost | 0.7236 | default learning_rate=0.3, n_estimators=500 |
| XGBoost | 0.7562 | learning_rate=0.05 |

**Interpretation.** XGBoost's first number was an artefact of an untuned learning rate (overfitting at 0.3), not evidence the model is weaker. After tuning it is the best single model. Random Forest was dropped from the design to keep the final stack to three diverse base learners (linear, two boosted-tree libraries).

---

## Week 4: Stacked ensemble

**Goal.** Build the proposed architecture and show whether combining models beats any single one.

**Method.**
- Level 0: LightGBM, XGBoost, Logistic Regression, each trained with 5-fold CV on the training set, producing out-of-fold (OOF) predictions for every training row.
- Level 1: a Logistic Regression meta-learner fitted on the three OOF columns, then evaluated on the true test set.
- *Why OOF:* fitting the meta-learner on in-sample base predictions leaks training labels and inflates it. The design follows the lesson from a top Kaggle solution to trust OOF CV over leaderboard chasing.
- *Why an LR meta-learner:* it only learns three weights, so it can't overfit and stays interpretable. A neural meta-learner (as in the baseline literature) was rejected for the opposite reasons.

**Results.**
- OOF AUC: LightGBM 0.7424, XGBoost 0.7495, LR 0.7435.
- Stack test AUC: **0.7579**, above all three standalone baselines.
- Meta-weights: LightGBM 1.290, XGBoost 3.481, LR 3.950. The meta-learner leans on LR and XGBoost, which suggests LightGBM's information is largely redundant given the other two.

**Caveat.** The weights are on raw OOF probabilities, so treat them as a rough indication of reliance, not a precise ranking of importance.

---

## Week 5: Model comparison

| Model | AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.7469 | 0.2297 |
| LightGBM | 0.7520 | 0.2388 |
| XGBoost | 0.7562 | 0.2476 |
| Stacked ensemble | 0.7579 | 0.2485 |

**Interpretation.** The stack is best on both metrics, but its margin over XGBoost is small (+0.0017 AUC, +0.0009 PR-AUC). We claim the ordering, not a large improvement, and test whether the gap is statistically meaningful in Week 12 (repeated runs and DeLong's test).

**Outputs.** `results_summary.csv`, `roc_comparison.png` (ROC overlay), `model_comparison.png` (AUC and PR-AUC bars).

---

## Week 6: Fuzzy augmentation (reject inference)

**Goal.** Test whether adding simulated rejected applicants improves scoring. Each unlabeled applicant is duplicated into a "good" copy weighted P(good) and a "bad" copy weighted P(bad), then the model is retrained with `sample_weight`.

**Experiment A: unlabeled pool (`application_test.csv`).**
- 80/20 labeled split of `application_train` (train_sub / holdout_eval). The pool is 48,744 rows, ~16.5% of total training weight, mean P(default) 0.0788.
- Single run: AUC 0.7594 to 0.7603. Over 5 seeds: mean delta AUC +0.0006 (std 0.0010), PR-AUC -0.0004 (std 0.0010), sign flipping across seeds.
- **Conclusion:** no measurable effect. This is expected, since the pool comes from the same approved population, so there is no selection bias to correct.

**Experiment B: tail-based rejects (genuine selection bias).**
- The bottom 20% of applicants by mean EXT_SOURCE_1/2/3 are held out as "hidden rejects" (true labels kept aside); the rest are "accepts". Reject default rate 18.68% vs 5.42% for accepts, which is real selection bias.
- LightGBM, 5 seeds: mean delta AUC **+0.0060** (std 0.0007) and PR-AUC +0.0064 (std 0.0005) on the hidden rejects, positive in every seed (range +0.0054 to +0.0072).
- XGBoost (seed 42): +0.0061 AUC on rejects, +0.0041 on holdout. This independently confirms the LightGBM effect size.
- **Logistic Regression: delta is exactly 0.0000.** This is a provable no-op, not a bug. A pair duplicated at the model's own predicted p contributes p(1-p) - (1-p)p = 0 to the MLE score equation, so the optimum is unchanged. Tree models are not fit through a linear score equation, so they do shift.

**Stack ablation (in-fold augmentation to avoid leakage).** Baseline vs augmented stack AUC: 0.6646 vs 0.6651 on rejects, 0.7583 vs 0.7584 on holdout. Small but positive on all four metrics. Meta-weights moved: LightGBM 2.012 to 2.707, XGBoost 4.502 to 4.698, LR 6.676 to 6.067.

**Conclusion.** No effect without selection bias; a real ~0.006 AUC gain for tree models when it exists; a structural zero for MLE-linear models; a small positive effect at stack level.

**Outputs.** `fuzzy_augmentation.py`, `fuzzy_augmentation_tail.py`, `fuzzy_augmentation_results.csv`, `fuzzy_augmentation_tail_results.csv`.

---

## Week 7: LendingClub cross-domain transfer

**Goal.** Train a model on LendingClub, use it to score Home Credit applicants, and add that score as a 4th meta-feature. Also report LendingClub standalone as external validation.

**Method.**
1. Column inventory of both datasets and a concept-mapping table (concept | Home Credit column | LendingClub column | notes), giving **5 shared features**: log_income, loan_to_income, payment_to_income, emp_years, owns_property.
2. LendingClub preprocessed on its own terms with its own train/test split.
3. LightGBM trained on the shared features only; LC model scored on Home Credit; meta-learner retrained with 4 inputs.
4. Ablation of the 3-model vs 4-model stack using a paired bootstrap on the test set, reusing the cached 5-fold OOF predictions.

**Results.**
- LC standalone AUC: 0.6713 with raw ratios, 0.6765 after within-domain percentile ranking (medians aligned at ~0.5).
- **Cross-domain AUC on Home Credit: 0.4950 (raw), 0.4987 (percentile), which is chance.**
- Diagnostic: Home Credit's own in-domain AUC with the same 5 features is only 0.6242, so the shared features are weak even in-domain.
- Univariate AUC (LC / HC): log_income 0.451 / 0.481, loan_to_income 0.598 / 0.498, payment_to_income 0.588 / 0.520, emp_years 0.490 / 0.421, owns_property 0.460 / 0.495. Several features carry signal in LendingClub that doesn't transfer, and some flip direction.
- Ablation: 3-model stack AUC 0.7579 / PR-AUC 0.2485 vs 4-model 0.7583 / 0.2488. Delta +0.00038, 95% bootstrap CI [-0.00031, +0.00103], which includes zero. LC meta-weight -0.222; LC score alone on the HC test set AUC 0.4973; correlation with the base OOFs ~0.

**Interpretation.** The LendingClub score adds nothing. The reason is likely that the 5 mappable features are weak and the two populations (US consumer lending vs Home Credit's population) relate to default differently. This is reported as a **negative result with diagnostics**, not hidden.

---

## Week 8: Fairness check

**Goal.** Measure group disparities in the model's decisions. Mitigation is out of scope.

**Method.** Each model approves its lowest-risk 80% using one threshold for all groups. Metrics per group: approval rate, disparate-impact (DI) ratio as a proxy, TPR and FPR gaps, per-group AUC. Slices: `CODE_GENDER`, then `DAYS_BIRTH` age bands.

**Results: gender (all models, baseline vs augmented stack essentially identical).**
- Approval rate: F ~0.85 vs M ~0.70-0.71. Default rates: F 0.070, M 0.102. DI ratio 0.82-0.84.
- TPR gap ~0.19-0.20, FPR gap ~0.12-0.135. Per-group AUC: F ~0.750, M ~0.758, so ranking quality is similar across groups even though approval differs.

**Results: age bands.** DI ratio ~0.66-0.68, TPR gap 0.43-0.49, FPR gap 0.27-0.29. Larger gaps than gender.

**Proxy check (LightGBM, 80% approval).**
| | AUC | Approval F | Approval M | DI |
|---|---|---|---|---|
| With CODE_GENDER_M | 0.7520 | 0.845 | 0.714 | 0.845 |
| Without | 0.7489 | 0.825 | 0.752 | 0.912 |

Dropping gender costs 0.003 AUC and closes about half the DI gap. `CODE_GENDER_M` and `DAYS_BIRTH` are direct model inputs (via `get_dummies`), so this is the model using protected attributes, not just correlated proxies.

**Interpretation.** Part of the approval gap reflects a real default-rate difference between groups (M 0.102 vs F 0.070), not model bias alone. The gaps are still large enough to need disclosure, and age is the bigger issue. Week 9 then asks how much these attributes show up in the model's own explanations.

**Outputs.** `fairness.py`, `fairness_results.csv`, `fairness_summary.csv`.
## Week 9: SHAP adverse-action reasons

### Setup
- Model: LightGBM, holdout AUC 0.759.
- SHAP: TreeExplainer; global plots on an 8K holdout sample, reasons on all rejected applicants. Reconstruction error ~1e-14.
- Rejection rule: riskiest 20% of the holdout (80% approval), 12,301 applicants.

### Global findings
- EXT_SOURCE_1/2/3 dominate, with directions matching expectations.
- AMT_CREDIT and AMT_GOODS_PRICE push in opposite directions (collinearity), so they are merged into one loan-to-goods reason.
- Age effect is non-monotonic (lowest under 30 and over 55, peak in the mid-30s), probably through interactions with the external scores.
- OWN_CAR_AGE is 65.8% missing and has placeholder values at 64-65 years. It is used by the model but not reported.

### Reason generation
- SHAP values are summed within reason groups, groups are ranked, and a group is reported only if it contributes more than 0.02 log-odds toward default; at most 4 reasons per applicant.
- Not reported: protected attributes (gender, age, marital status); education type; social-circle counts; building-statistics columns; application hour; OWN_CAR_AGE; document and work-phone flags.
- 92.6% of rejects receive 4 reasons, 5.6% receive 3, and 1.9% (228) receive fewer than 3.

### Reason frequency (share of rejected applicants)
![img.png](img.png)

### Fairness
- A protected attribute appears in the top three unfiltered SHAP drivers toward default for 31.5% of rejected applicants (gender 21.0%, age 11.2%, marital status 0.9%).
- These features are excluded from the reported reasons, so for those applicants the reported reasons omit part of the model's actual explanation. Removing protected attributes from the explanation does not remove their influence on the decision.
- Waterfall plots show the full model explanation; reported reasons are the filtered subset.

### Limitations
- "Low external credit score" appears for 97.6% of rejects, so the model is essentially external-score-driven.
- "Residential region risk rating" (20.4%) is a possible geographic proxy.
- The 0.02 threshold and the 4-reason cap are design choices, not tuned.

### Figures
shap_beeswarm.png, shap_bar.png, shap_dep_DAYS_BIRTH.png, shap_dep_OWN_CAR_AGE.png, waterfall_{clear_cut,borderline,typical}.png