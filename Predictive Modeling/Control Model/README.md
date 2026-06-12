# Control Model

Baseline **control XGBoost model** trained on **real patient data** using **all
biomarkers** (no causal feature selection). This is the "Control" + "XGB"
configuration from `../DAG Parameterization.ipynb` (Experiment 1:
`remove_drugs=False`, `remove_interventions=False`), broken out into a
standalone, reproducible artifact. It is the baseline against which the
causal-DAG-selected feature subsets are compared.

## Reproduce

```bash
# conda env with xgboost 3.0.5 / scikit-learn 1.7.2 / pandas 2.3.2
/opt/homebrew/Caskroom/miniconda/base/envs/causal_inf/bin/python train_control_model.py
```

## Configuration

- **Data:** real patients (`X_train/X_test_c12_w48_imp.csv`), NOT the synthetic
  patients used in `DAG Parameterization Synth.ipynb`.
- **Features:** all 45 biomarkers → 811 catch22 time-window features remaining
  after the ≥0.01 variance filter. XGBoost trains on the raw (un-imputed) matrix
  and handles missing values natively.
- **Model:** `XGBClassifier(objective="binary:logistic", random_state=42,
  eval_metric="aucpr")` — default trees/depth, identical to the notebook.
- **Labels:** `Outcome`, collapsed to one row per patient (`uid`) via `max`.

## Test-set performance (real patient held-out set, n=3895, 11.07% positive)

| Metric | Value |
|---|---|
| AUROC | 0.9304 |
| Average Precision (AUPRC) | 0.8082 |
| Brier Score | 0.0418 |
| Balanced Accuracy | 0.8018 |
| Accuracy | 0.9510 |
| Precision | 0.9196 |
| Recall | 0.6102 |
| F1 | 0.7336 |

## Files

| File | Description |
|---|---|
| `train_control_model.py` | Reproducible training script |
| `control_xgb_model.json` | Trained XGBoost model (load with `XGBClassifier().load_model(...)`) |
| `X_train_control.csv` | Training features used (13054 × 811) |
| `X_test_control.csv` | Testing features used (3895 × 811) |
| `y_train_control.csv` | Training labels (`Outcome`, per uid) |
| `y_test_control.csv` | Testing labels (`Outcome`, per uid) |
| `feature_list.txt` | Ordered list of the 811 features |
| `control_model_metrics.csv` | Test-set metrics above |
