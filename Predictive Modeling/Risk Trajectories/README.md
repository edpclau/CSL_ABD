# Risk & SHAP Trajectories via Moving Window

Turns the tuned ensemble's single static test-set prediction into a **time
trajectory**: for every test patient we slide the 48-hour observation window
backward hour-by-hour through their PICU course, and at each window position we
faithfully recompute the model's inputs, its predicted risk, and its
per-biomarker SHAP attributions. This shows how risk and feature attributions
evolve as the window approaches the point the model was actually tested on, and
quantifies how the model's discrimination changes with lead time.

Spec: `../../docs/superpowers/specs/2026-06-11-risk-shap-trajectories-design.md`
Plan: `../../docs/superpowers/plans/2026-06-11-risk-shap-trajectories.md`

## What the model is

Tuned stacked ensemble in `../Ensemble Model/tuned/`: three XGBoost component
heads (EEG, CT, MRI) over the 811 catch22 biomarker features → a shallow XGBoost
meta-learner → `P(Outcome)`. Published held-out metrics (n=3895): **AUROC 0.9229,
AUPRC 0.7913**.

## What this computes

For each test patient:
- **Anchor** = the censor timestamp (identical to `TemporalDataSubset`): for a
  case, the first `Outcome==1` minus the 12 h censor; for a control, the last
  timestamp. The window at offset **k=0** (`[anchor−48h, anchor)`) is exactly the
  window the model was tested on.
- **Sliding windows**: `window_end = anchor − k·1h` for every hour `k` back that
  the 48 h window still fits the record. Each window is featurized through the
  **exact training path** — slice → pad to 48 rows → reuse the saved SAITS
  imputer (`20251023_T144845`) → sktime catch22 → 811 features — then scored to
  get the ensemble risk and the three component probabilities.
- **SHAP** (biomarker level, all heads, saved in full): exact TreeSHAP via
  XGBoost `pred_contribs` for each head vs the 811 features, plus the small
  meta-learner SHAP over the 3 components.

Windows with **zero observed timepoints** (a 48 h slice falling entirely inside a
data gap) are skipped — a gap should be a gap in the trajectory, not a fabricated
risk. This never affects k=0, so the test-set reproduction is unchanged.

## Faithfulness (validation gate)

The k=0 window reproduces the published test set **bit-exactly**, on all 3,895
patients:

| check (k=0, full test set) | result |
|---|---|
| AUPRC | **0.7913** (published 0.7913) |
| AUROC | **0.9229** (published 0.9229) |
| risk vs `ensemble_final_test_predictions.csv` | max\|Δ\| = 3.0e-8 |
| features vs `X_test_control.csv` | max\|Δ\| ≈ 1e-12, identical NaN structure |

This is checked by `validate_k0.py` (a sample) and by the k=0 row of
`window_metrics.csv` (the full set).

## Results

- **3,895 patients, 423,183 (patient, window) rows**; mean 108.6 windows/patient
  (median 56, max 854).
- **Average AUPRC over all windows: 0.137 pooled / 0.175 macro** (vs 0.791 at
  k=0). Discrimination is highest at the censor point and degrades with lead
  time — see `figures/discrimination_vs_lead.pdf`.
- Cases carry higher predicted risk than controls at every lead time, with the
  clearest separation near k=0 — see `figures/aggregate_risk_vs_lead.pdf`.

### Interpreting the lead-time curves (caveats)

- **Length-selection bias.** Only long-stay patients reach far-back offsets, so
  per-offset metrics at large k are computed on a shrinking, sicker subpopulation
  (the `n_patients` column / gray band quantify this). Far-back metrics are noisy
  and not comparable to k=0.
- **Short-lead cases.** `n_pos` falls from 431 at k=0 to 192 at k=1: ~239 cases
  have an outcome early enough that only the k=0 window fits. The AUPRC drop from
  k=0→k=1 partly reflects losing these cases plus the 12 h censor offset.

## Artifacts (`artifacts/`)

The large arrays (`shap/`, `*.parquet`, `*.log`) are git-ignored; the two small
metrics files are committed.

| file | contents |
|---|---|
| `risk_trajectories.parquet` | one row per (uid, k): `k` (hours before anchor), `n_observed`, `EEG_p`, `CT_p`, `MRI_p`, `ensemble_risk`, `y_true`. 423,183 rows, 5.5 MB. *(git-ignored)* |
| `shap/<uid>.npz` | per-patient SHAP, full: `eeg`/`ct`/`mri` each `(n_windows, 812)` (811 features + bias), `meta` `(n_windows, 4)` (3 components + bias), `k` `(n_windows,)`. 3,895 files, ~990 MB. *(git-ignored)* |
| `shap/feature_names.json` | the 811 feature names (column order of the SHAP arrays). |
| `window_metrics.csv` | per offset `k`: `n_patients`, `n_pos`, `AUPRC`, `AUROC`. *(committed)* |
| `window_metrics_summary.json` | the headline averages: `auprc_pooled`/`auroc_pooled`, `auprc_macro`/`auroc_macro`, `n_window_rows`. *(committed)* |
| `test_raw.parquet` | compact long table of the test patients' raw biomarkers (build cache, rebuilt each run). *(git-ignored)* |

Loading one patient (memory-safe — never load the whole SHAP set at once):

```python
import json, numpy as np, pandas as pd
risk = pd.read_parquet("artifacts/risk_trajectories.parquet")
names = json.load(open("artifacts/shap/feature_names.json"))   # 811
z = np.load("artifacts/shap/BA1466936712.npz")
# z["eeg"]: (n_windows, 812); column j is SHAP of names[j] on the EEG head; col 811 = bias
# z["k"]: hours-before-anchor for each row (0 = tested window)
```

`figures/` (committed): `aggregate_risk_vs_lead.pdf`, `discrimination_vs_lead.pdf`,
and `example_{TP,FN,FP,TN}_<uid>.pdf` (risk trajectory + top-k biomarker SHAP over
time for two example patients per class).

## Reproduce

Env: `ts_ml` conda env (`/opt/homebrew/Caskroom/miniconda/base/envs/ts_ml/bin/python`) —
pycatch22, sktime 0.38.5, pypots 1.0, xgboost 3.2.0, torch, pyarrow.

```bash
export PY="/opt/homebrew/Caskroom/miniconda/base/envs/ts_ml/bin/python"

# full run (parallel; ~50 min on a 10-core / 32 GB Mac)
$PY run_full.py --nshards 6
# if a worker crashes partway, finish only the missing patients:
$PY run_full.py --resume --nshards 8

# metrics + figures
$PY window_metrics.py     # writes window_metrics.csv + window_metrics_summary.json
$PY make_figures.py       # writes figures/*.pdf

# tests (includes the k=0 gate)
$PY -m pytest tests/ -v
```

### Why parallel (and why not pycatch22)

sktime's catch22 is single-threaded and dominates ~94% of the runtime
(~15 h single-process). It is **not** swappable for raw `pycatch22`: sktime's
implementation produces different values for the scale-dependent features, and
the model was trained on sktime-flavored features — so we parallelize the *exact*
sktime path across processes instead. `run_full.py` builds the compact table
once, runs N workers over round-robin uid shards (each writing per-patient SHAP +
a shard parquet), then merges. `--resume` processes only uids without a SHAP npz
and folds them into the existing merged parquet, so a partial run can be finished
without redoing work.

## Files

| file | role |
|---|---|
| `config.py` | paths + constants (window 48 h, censor 12 h, hourly stride; the 45 biomarkers; the 22 catch22 features) |
| `windowing.py` | anchor + hourly window enumeration (mirrors `TemporalDataSubset`) |
| `featurize.py` | biomarker prep, training-faithful padding, SAITS imputer, catch22 → 811 |
| `data_io.py` | streamed compact test-patient table |
| `model.py` | load tuned ensemble; ensemble risk + exact TreeSHAP |
| `pipeline.py` | per-patient engine: windows → 811 features (drops empty-gap windows) |
| `compute_trajectories.py` | streaming, chunked driver (risk parquet + per-patient SHAP npz) |
| `run_full.py` | parallel orchestrator (full run + `--resume`) |
| `window_metrics.py` | per-offset AUPRC/AUROC + headline averages |
| `make_figures.py` | trajectory figures |
| `validate_k0.py` | k=0 reproduction gate |
| `tests/` | unit + gate tests (pytest) |
