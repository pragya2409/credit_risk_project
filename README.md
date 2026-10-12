# Beyond Default Prediction: Reject Inference, Explainable Adverse-Action Reasoning, and Counterfactual Guidance for Credit Risk

Semester ML research project on credit-risk scoring (Home Credit Default Risk). The running write-up with all results, caveats and figures is in [`report.md`](report.md).

## Setup
- Python 3.14 virtualenv; install with `pip install -r requirements.txt`.
- Runs on a laptop CPU (developed on a MacBook Air M2, no GPU). Each script takes seconds to a few minutes unless noted.
- `data/` and `*.pkl` caches are not in the repository.

## Data (download yourself; not redistributed)
Accept each competition's rules on Kaggle first.

| Dataset | Source | Files | Put in |
|---|---|---|---|
| Home Credit Default Risk | Kaggle competition `home-credit-default-risk` | `application_train.csv`, `application_test.csv`, `bureau.csv`, `bureau_balance.csv`, `installments_payments.csv` | `data/` |
| LendingClub | Kaggle `wordsforthewise/lending-club` (check the license on the dataset page) | `accepted_2007_to_2018Q4.csv.gz` | `data/lc/` |
| UPI transactions 2024 | Kaggle `skullagos5246/upi-transactions-2024-dataset` (CC0) | `upi_transactions_2024.csv` | `data/upi/` |

```bash
kaggle competitions download -c home-credit-default-risk -p data/
kaggle datasets download -d skullagos5246/upi-transactions-2024-dataset -p data/upi --unzip
```

## Run order (approximate; check each script's header)
| Weeks | Scripts | Output |
|---|---|---|
| 2-5 baselines, OOF stack, comparison | `main.py` | `results_summary.csv`, `roc_comparison.png`, `model_comparison.png`, `stack_cache.pkl`, `standalone_cache.pkl` |
| 6 reject inference | `fuzzy_augmentation.py`, `fuzzy_augmentation_tail.py` | `fuzzy_augmentation*_results.csv`, `fuzzy_tail_cache.pkl` |
| 7 LendingClub transfer | `lc_inventory.py`, `make_mapping.py`, `lc_transfer.py`, `lc_ablation.py` | `feature_mapping.csv`, `lc_*_results.csv` |
| 8 fairness | `fairness.py` | `fairness_results.csv`, `fairness_summary.csv` |
| 9 SHAP reasons | `shap_global.py`, `generate_reason.py` (uses `common.py`, `reasons_config.py`) | `shap_*.png`, `reasons.csv`, `waterfall_*.png` |
| 10 counterfactuals | `dice_smoke.py`, `dice_constrained.py`, `dice_counterfactuals.py`, `manual_perturbation.py`, `cf_summary.py`, `cf_evaluate.py`, `cf_figure.py` | `counterfactuals*.csv`, `manual_perturbation.csv`, `counterfactual_before_after.png` |
| 11 simulator | `inspect_tables.py`, `explore_aux.py`, `inspect_upi.py`, `upi_params.py`, `transition_matrices.py`, `transition_matrices_fit.py`, `simulator.py <seed> _fit`, `empirical_check.py`, `simulator_stress.py`, `week11_figures.py` | `transition_*.csv`, `simulation_results*.csv`, `stress_sweep_seed42.csv`, `*.png` |
| 12 statistical rigor | `delong.py`, `repeated_runs.py`, `repeated_runs_analysis.py`, `fairness_bootstrap.py` | `delong_results.csv`, `repeated_runs_results.csv`, `fairness_bootstrap*.csv` |

## Notes
- Weeks 2-5 and Week 12 use the preprocessing in `main.py` (imputation, missingness flags, one-hot encoding). Weeks 9-10 use `common.py`, which reads the raw CSV and keeps NaNs. They are different pipelines.
- `bureau_balance.csv` and `installments_payments.csv` are large (hundreds of MB); Week 11 scripts need a few GB of free memory.
- Random seeds are fixed throughout; `repeated_runs.py` runs 10 seeds.
- Limitations and caveats for each component are in `report.md`.