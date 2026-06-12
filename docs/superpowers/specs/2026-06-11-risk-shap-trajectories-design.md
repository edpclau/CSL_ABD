# Risk & SHAP Trajectories via Moving Window — Design

**Date:** 2026-06-11
**Status:** Approved for planning
**Author:** Eddie Pérez (with Claude)

## 1. Goal

For every patient in the held-out test set, turn the model's single static
prediction into a **time trajectory** by sliding the 48-hour observation window
backward through the patient's PICU course. At each window position we recompute,
faithfully, the model's inputs, its predicted risk, and its per-biomarker SHAP
attributions. We then chart how risk and attributions evolve as the window
approaches the exact point the model was tested on, and we quantify how the
model's discrimination (AUPRC/AUROC) changes with lead time.

This produces, per test patient:
- a **risk trajectory** — ensemble P(Outcome) as a function of hours-before-anchor;
- a **SHAP trajectory** — per-biomarker attribution over time, for each of the
  three component heads (EEG, CT, MRI), saved in full;

and, across patients:
- **per-window AUPRC/AUROC** as a function of lead time, plus a single headline
  average AUPRC over all (patient, window) pairs.

## 2. Background: the model and the existing pipeline

**Model — tuned stacked ensemble** (`Predictive Modeling/Ensemble Model/`):
- Level-0: three XGBoost **heads** (EEG, CT, MRI), each consuming the **811
  catch22 biomarker features** and predicting its own outcome component.
  Artifacts: `tuned/head_{EEG,CT,MRI}_tuned.json`.
- Level-1: a shallow XGBoost **meta-learner** on the 3 component probabilities →
  P(Outcome). Artifact: `tuned/meta_xgb_tuned.json`.
- Inference helper: `Ensemble Model/predict_ensemble.py`
  (`load_ensemble()`, `predict_outcome(X, heads, meta)`).
- Published test metrics (n=3895): AUROC 0.9229, **AUPRC 0.7913**, F1@0.5 0.7354.
  Final test predictions: `Ensemble Model/ensemble_final_test_predictions.csv`.

**Feature pipeline** that produced the model's inputs (from
`Data Pre-processing/Preprocessing/Timeseries Summarization copy.ipynb` and
`HelperFuncsTimeseries.py`):
1. **Raw data:** `Deidentified Staged Data/nmb.csv` — hourly long-format time
   series, indexed by `(uid, timestamp)`, ~45 biomarkers + outcome columns,
   28,594 patients. `timestamp` and `elapsed_time` are timedeltas.
2. **Temporal subset** (`TemporalDataSubset(df, censor=12, window=48)`): per
   patient compute the **anchor (censor timestamp)**:
   - case (outcome ever 1): timestamp of first `Outcome==1` minus `censor` (12h);
   - control: last observed timestamp.
   Keep rows with `Window_timestamp <= timestamp < Censor_timestamp`, i.e. the
   48h window `[anchor − 48h, anchor)`. Patients with
   `elapsed_time < window + censor` (60h) are excluded. Outcome columns are
   propagated to a single per-patient label (max).
3. **Pad** each patient's window to a fixed length (right-aligned hourly grid).
4. **SAITS impute** the continuous biomarkers (PyPOTS), model fit on TRAIN and
   applied to TEST. Saved checkpoint under
   `Data Pre-processing/Preprocessing/saits_model/` — candidate `20251023_T144845`
   (dated 2025-10-24, matching the `X_*_c12_w48_imp.csv` file dates).
5. **catch22** featurization (sktime `Catch22`, `catch24=True`, the 22-feature
   subset in notebook cell 21, `col_names="str_feat"`) → 45×22 columns, filtered
   to the **811** features listed in
   `Predictive Modeling/Control Model/feature_list.txt` (exact order matters).
6. Saved as `Data Pre-processing/X_{train,test}_c12_w48_imp.csv`; the test matrix
   is mirrored as `Predictive Modeling/Control Model/X_test_control.csv`.

The naming `c12_w48` = 12h censor + 48h window. (Note: the "copy" notebook on
disk was last run with `censor=0, window=72`; the **canonical config for the
saved model is `censor=12, window=48`**, which the validation gate in §6 confirms.)

## 3. Moving-window definition

Per patient, the **anchor** is identical to the test pipeline's censor timestamp
(§2 step 2). The **final window** (offset k=0) is `[anchor − 48h, anchor)` — the
exact window the model was tested on.

Sliding windows step the window-end backward by **1 hour** (hourly stride):

```
window_end(k) = anchor − k·1h        k = 0, 1, 2, ...
window(k)     = [window_end(k) − 48h, window_end(k))
```

valid while the window's start stays within the patient's observed record
(`window_end(k) − 48h >= patient's first timestamp`). k=0 is always present
(guaranteed by the 60h inclusion rule); deeper k exist only for longer stays.

**x-axis for charts:** hours-before-anchor = `k` (0 = the tested window;
increasing = earlier in the stay / longer lead time).

## 4. Faithful per-window feature computation

For each window we reproduce steps 3–5 of §2 **exactly**, so the model sees
inputs drawn from the same distribution it was trained/tested on:

1. Slice the patient's `nmb.csv` rows to `[window_end−48h, window_end)`.
2. Pad to the fixed SAITS length (same `pad_length` / `n_steps` as the saved
   checkpoint; right-aligned hourly grid, matching `padder`).
3. SAITS-impute using the **saved** checkpoint (no refitting).
4. catch22 transform with the **same** sktime config; subset/reorder to the 811
   `feature_list.txt` columns.

Categorical/ordinal handling (e.g. Pupillary Reaction ordinal map, the
confounder/bias/outcome column drops) replicates the notebook so the 811-column
feature frame is identical in schema.

**Environment:** `ts_ml` conda env
(`/opt/homebrew/Caskroom/miniconda/base/envs/ts_ml/bin/python`) — has pycatch22,
sktime 0.38.5, pypots 1.0, xgboost 3.2.0, torch 2.6.0. (`shap` is **not**
needed; see §5.)

## 5. Scoring and SHAP

At each window:
- **Risk:** run the three tuned heads on the 811-feature frame to get
  `[EEG_p, CT_p, MRI_p]`, then the meta-learner → **ensemble P(Outcome)**.
- **SHAP (biomarker level, all heads, saved in full):** use XGBoost's built-in
  **exact TreeSHAP** via `booster.predict(DMatrix(X, feature_names=...),
  pred_contribs=True)`. This returns an `(n, 812)` array per head (811 feature
  contributions + 1 bias), additive to the logit (verified: `sigmoid(sum) ==
  predict_proba`, atol 1e-5). No `shap` package required.
- **Component-level SHAP (small, also saved):** TreeSHAP on the meta-learner →
  3 component contributions, so biomarker-level (heads) and component-level
  (meta) views connect.

`pred_contribs` bypasses XGBoost feature-name validation by attaching the saved
`booster.feature_names` to the DMatrix.

## 6. Validation gate (built and passed FIRST)

Before any full run, prove the moving-window pipeline reproduces the published
artifacts at k=0:

- **Features:** for a sample of test patients, k=0 features ≈
  `X_test_control.csv` rows (per-feature max-abs-diff under a small tolerance,
  high correlation). Confirms SAITS checkpoint, pad length, catch22 config, and
  the 811-column selection/order.
- **Predictions:** k=0 ensemble risk ≈ `ensemble_final_test_predictions.csv`
  (`p_outcome`); overall k=0 AUPRC == 0.7913 / AUROC == 0.9229 within tolerance.

If the gate fails, diagnose before proceeding (most likely SAITS checkpoint or
`pad_length` / `n_steps` mismatch — try other `saits_model/*` checkpoints; verify
catch22 column naming). **No full trajectory run until k=0 reproduces.**

## 7. Memory strategy (bounded; machine-constrained)

Hourly stride over 3,895 patients yields **~250k (patient, window) rows**; full
per-head SHAP is **~2.3 GB**. This must never be held in RAM at once.

- A **single streaming script** processes patients in **small chunks
  (default 50–100 patients)**. Per chunk: build that chunk's windows → batch
  through SAITS → catch22 → score → SHAP → **append to disk** → free memory →
  next chunk. Nothing global is accumulated in memory.
- **Not** a parallel fan-out: multiple SAITS/XGBoost processes would multiply
  memory, the opposite of the requirement.
- SAITS and catch22 are batched **within** a chunk (a chunk's windows form one
  `(N_windows, n_steps, n_features)` tensor — tens of MB).

## 8. Outputs / artifacts

All under `Predictive Modeling/Risk Trajectories/`:

| Artifact | Contents |
|---|---|
| `risk_trajectories.parquet` | one row per (uid, k): `k` (hrs-before-anchor), `window_end`, `n_observed` (non-padded timepoints in window), `EEG_p`, `CT_p`, `MRI_p`, `ensemble_risk`, `y_true`. Appended per chunk. |
| `shap/<uid>.npz` | per-patient SHAP, **saved in full**: arrays `eeg`, `ct`, `mri` each shape `(n_windows, 812)` (811 features + bias), plus `meta` `(n_windows, 4)` (3 components + bias), `k` index, and `feature_names`. One file per patient → charting loads one patient at a time. |
| `window_metrics.csv` | per offset `k`: `n_patients`, `AUPRC`, `AUROC`. Plus a header/summary line with the **overall average AUPRC** across all (patient, window) pairs and the macro-mean over offsets. k=0 row must equal published 0.7913 / 0.9229. |
| `README.md` | what was computed, config (stride=1h, window=48h, censor=12h, SAITS checkpoint id), how to reproduce, how to load artifacts. |

Per-window AUPRC/AUROC at large `k` is computed over a shrinking, **length-biased**
subset (only long-stay patients reach far-back windows). `n_patients` is reported
at every offset and this bias is stated in the README and on the metric chart.

## 9. Charts

Charting script reads artifacts (never the full SHAP at once) and writes PDFs/PNGs
to `Risk Trajectories/figures/`:

1. **Aggregate risk vs. lead time** — mean ensemble risk (± IQR/CI band) over `k`,
   split by true outcome (case vs. control). Secondary axis or twin panel: `n`.
2. **Per-window discrimination** — AUPRC and AUROC vs. `k`, with `n` annotated;
   horizontal reference at the k=0 / published value.
3. **Illustrative individuals** — a grid of example patients (true positive,
   false negative, false positive, true negative): each shows the risk
   trajectory and, beneath it, the **top-k biomarker SHAP-over-time** panels for
   the relevant head(s). Top-k chosen by peak |SHAP| over the trajectory.
4. **(Optional) risk heatmap** — patients (rows, sorted by outcome/peak risk) ×
   lead time (cols), colored by risk.

All raw SHAP remains on disk regardless of what is plotted; charts surface a
top-k slice for legibility.

## 10. Code structure

`Predictive Modeling/Risk Trajectories/`:
- `windowing.py` — anchor computation + hourly window enumeration per patient
  (mirrors `TemporalDataSubset`); pure, unit-testable.
- `featurize.py` — per-window pad → SAITS → catch22 → 811-column frame; loads the
  saved SAITS checkpoint and catch22 config once.
- `score_shap.py` — load tuned heads + meta; ensemble risk + per-head TreeSHAP +
  meta TreeSHAP.
- `compute_trajectories.py` — the streaming driver: chunked patient loop, calls
  the above, appends artifacts. CLI flags for stride, window, censor, chunk size,
  patient subset (for the validation gate).
- `validate_k0.py` — the §6 gate: reproduce X_test/predictions at k=0.
- `make_figures.py` — the §9 charts from artifacts.
- `README.md`.

Tests (pytest, `ts_ml`): windowing edge cases (exact-60h stay → only k=0;
case vs. control anchor; window-start clipping), feature schema = 811 ordered
columns, SHAP additivity (`sigmoid(sum)==proba`).

## 11. Non-goals

- No retraining or re-tuning of any model; artifacts are loaded as-is.
- No re-fitting of SAITS; the saved checkpoint is reused.
- No change to the outcome definition, the train/test split, or the 811-feature
  set.
- Not a real-time/online system; this is a retrospective trajectory analysis.

## 12. Open risks

- **SAITS checkpoint identity.** If `20251023_T144845` doesn't reproduce k=0,
  iterate over the other `saits_model/*` checkpoints / check `n_steps`. Gated by §6.
- **catch22 version drift.** sktime 0.38.5 must yield the same 811 values as
  whatever produced the saved features; the §6 per-feature diff catches this. If
  it drifts, pin the column naming and re-validate.
- **Runtime.** ~250k windows × (SAITS + catch22 + TreeSHAP). catch22 (pycatch22,
  C) dominates but is ~µs/feature; expected to be minutes–low hours, chunked.
  Acceptable for a one-time retrospective run; progress logged per chunk.
