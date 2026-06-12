# Ensemble Model — per-component stacking

A **stacked ensemble** that predicts the composite neuro-morbidity `Outcome` by
first predicting its components, then combining them. Compared against the
monolithic all-biomarker control model in `../Control Model/`.

## ✅ Selected model

**Tuned per-component heads (EEG, CT, MRI) → tuned shallow-XGBoost meta-learner.**
Load and predict via `predict_ensemble.py`:

```python
from predict_ensemble import load_ensemble, predict_outcome
heads, meta = load_ensemble()
p = predict_outcome(X, heads, meta)   # X: DataFrame with the 811 biomarker columns
```

**Test-set performance (n=3895, predicting Outcome):**

| Metric | Selected ensemble | Control baseline |
|---|---|---|
| AUROC | 0.9229 | 0.9304 |
| AUPRC | 0.7913 | 0.8082 |
| Brier | 0.0433 | 0.0418 |
| Balanced Acc | 0.8093 | 0.8018 |
| F1 @ 0.5 | **0.7354** | 0.7336 |

The selected model is the artifacts in `tuned/` (`head_{EEG,CT,MRI}_tuned.json` +
`meta_xgb_tuned.json`); final test predictions in `ensemble_final_test_predictions.csv`.

## Architecture

- **Level-0 (per-component XGBoost heads):** EEG, CT, MRI — the three components
  that compose `Outcome` (BHMeds=0 in this cohort ⇒ `Outcome == OR(EEG, CT, MRI)`).
  Each head uses all 811 biomarkers and predicts its own component. **Hyperparameters
  tuned per head with Optuna (50 trials, 3-fold CV) to maximize AUPRC** — see `tuned/`.
- **Meta-features:** leakage-safe **5-fold out-of-fold (cross-fitted)** predictions
  on train; heads refit on full train to score the test set.
- **Level-1 (pure stacking):** shallow XGBoost trained ONLY on the 3 component
  probabilities → `Outcome`. (LogReg and Noisy-OR were evaluated as alternatives.)

## How tuning changed things

Tuning each head for AUPRC improved every head and lifted the ensemble, shrinking
the AUPRC gap to control from ~0.050 to ~0.017:

| | Default heads | **Tuned heads (selected)** | Control |
|---|---|---|---|
| Ensemble AUROC (XGB meta) | 0.9163 | **0.9229** | 0.9304 |
| Ensemble AUPRC (XGB meta) | 0.7586 | **0.7913** | 0.8082 |
| Per-head test AUPRC (EEG/CT/MRI) | 0.277 / 0.276 / 0.218 | **0.327 / 0.310 / 0.254** | — |

Notable tuning findings: rare heads (EEG, MRI) favored heavy regularization (few
trees, high `gamma`), and aggressive `scale_pos_weight` did **not** help AUPRC.

## Takeaway

After tuning, the ensemble is **competitive with the control** — its F1 slightly
exceeds control and Brier nearly matches — but the monolithic control still edges
ahead on the ranking metrics (AUROC/AUPRC). The per-component decomposition's main
value is **interpretability and competing-risks structure** (you see each
component's contribution), now at minimal accuracy cost.

## Reproduce

```bash
# default ensemble
python train_ensemble.py
# per-head AUPRC tuning + rebuild (writes tuned/)
python tune_heads.py
# score the selected model
python predict_ensemble.py
```

(env: `/opt/homebrew/Caskroom/miniconda/base/envs/causal_inf/bin/python`)

## Files

| File | Description |
|---|---|
| **`predict_ensemble.py`** | **Selected-model loader + inference entry point** |
| `ensemble_final_test_predictions.csv` | Selected model's test predictions |
| `tuned/head_{EEG,CT,MRI}_tuned.json` | **Selected** tuned level-0 heads |
| `tuned/meta_xgb_tuned.json` | **Selected** tuned XGBoost meta-learner |
| `tuned/best_params_*.json` | Tuned hyperparameters + CV-AUPRC per head |
| `tuned/tuned_metrics.csv`, `tuned/tuning_summary.json` | Tuned results |
| `tune_heads.py` | Optuna AUPRC tuning pipeline |
| `train_ensemble.py` | Default (untuned) ensemble pipeline |
| `head_*.json`, `meta_logreg.joblib`, `meta_xgb.json` | Default-ensemble artifacts (ablations) |
| `oof_train_components*.csv`, `test_components*.csv` | OOF/test component-probability matrices |
| `ensemble_metrics.csv`, `ensemble_summary.json` | Default-ensemble metrics |
