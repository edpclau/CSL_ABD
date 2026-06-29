# XGBoost Hyperparameter Optimization for DAG Parameterization Runs

**Date:** 2026-06-29
**Status:** Approved (design)

## Problem

In both modeling notebooks, the XGBoost classifier is trained with fixed defaults and
no tuning:

```python
models['XGB'] = xgboost.XGBClassifier(objective='binary:logistic',
                                      random_state=42, eval_metric='aucpr')
```

We want automated hyperparameter optimization (HPO) per training run, tuned once and
reused, without leaking the test set and without making the Year Sensitivity notebook
prohibitively expensive.

## Scope

- **In scope:** XGBoost only, in `DAG Parameterization.ipynb` (`train_models`) and
  `DAG Parameterization - Year Sensitivity.ipynb` (`train_models_year_sensitivity`).
- **Out of scope:** The LGBN model (its "structure" is the DAG; no real
  hyperparameters). The pgmpy/LGBN code path is untouched.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Library | Optuna (TPE sampler, MedianPruner) |
| Budget | 50 trials, 5-fold CV, per (DAG, config) |
| Objective | `average_precision` (AUPRC) — matches reporting + class imbalance |
| Strategy | Tune once per (DAG, config), cache to JSON, reuse across notebooks |
| Cache miss | Self-healing: tune on full training data, then cache |
| Control | `tune=True` default; `force_retune` flag to ignore cache; `tune=False` reproduces fixed defaults |
| Integration | Standalone `hpo.py` module imported by both notebooks (Approach A) |
| `scale_pos_weight` | Excluded from search space, to keep probabilities calibratable |

## Architecture

A standalone module `Predictive Modeling/hpo.py` with three public functions. Both
notebooks import it; no tuning logic is duplicated.

### `config_label(remove_drugs, remove_interventions) -> str`

Canonical config string so the **same cache key is shared across both notebooks**:

| `remove_drugs` | `remove_interventions` | label |
|---|---|---|
| False | False | `"full"` |
| True | False | `"no_drugs"` |
| True | True | `"vitals_labs"` |

Year Sensitivity uses only `full` and `vitals_labs`, both of which are produced by the
DAG Parameterization run — so its runs hit the cache.

### `tune_xgb(X, y, n_trials=50, cv=5, seed=42) -> dict`

- Optuna study, `direction="maximize"`, `TPESampler(seed=seed)`, `MedianPruner()`.
- Objective runs a manual `StratifiedKFold(cv, shuffle=True, random_state=seed)` loop on
  `(X, y)`, fits an `XGBClassifier` per fold, scores `average_precision_score` on the
  held-out fold, calls `trial.report(score, fold_idx)` and honors
  `trial.should_prune()`. Returns the mean fold AP.
- Rows are already aggregated per patient (`groupby('uid').max()`), so no patient spans
  folds; plain `StratifiedKFold` is sufficient (no `GroupKFold`).
- Returns `study.best_params` (the tunable hyperparameters only).

**Search space** (XGB only):

| Param | Range / scale |
|---|---|
| `n_estimators` | 100–800 |
| `max_depth` | 3–10 |
| `learning_rate` | 1e-3–0.3 (log) |
| `subsample` | 0.5–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `min_child_weight` | 1–10 |
| `reg_lambda` | 1e-3–10 (log) |
| `reg_alpha` | 1e-3–10 (log) |
| `gamma` | 0–5 |

### `get_xgb_params(dag_name, cfg_label, X, y, cache_path, tune=True, force_retune=False) -> dict`

Cache-aware wrapper:

1. If `tune is False` → return `{}` (empty dict), so `XGBClassifier(**base, **{})`
   reproduces the current fixed-default behavior exactly. No cache read/write.
2. Load cache JSON (if file exists). Key = `f"{dag_name}||{cfg_label}"`.
3. On hit and not `force_retune` → return cached params.
4. On miss (or `force_retune`) → run `tune_xgb(X, y)`, write the entry back, return.

Each cache entry stores: `params`, `n_features`, `n_trials`, `cv`, `objective`,
and an ISO `timestamp` for traceability.

## Cache file

- JSON at a path held in a notebook variable. For the corrected re-run this is
  `biomarker_counts_fixed/xgb_best_params.json`; the original notebooks default to
  `xgb_best_params.json` in the notebook folder.
- Structure:

```json
{
  "Simplified Clinician Consensus||full": {
    "params": {"max_depth": 5, "learning_rate": 0.04, ...},
    "n_features": 278, "n_trials": 50, "cv": 5,
    "objective": "average_precision", "timestamp": "2026-06-29T12:00:00"
  }
}
```

## Integration points

In both notebooks, the fixed XGB instantiation is replaced. Tuning happens on the
**full training data** for that DAG/config feature set (`X_train.filter(xgb_features)`),
never the test set:

```python
base = dict(objective='binary:logistic', random_state=42, eval_metric='aucpr')
best = get_xgb_params(dag_name, config_label(remove_drugs, remove_interventions),
                      X_train_filtered, y_train.Outcome, cache_path=CACHE_PATH,
                      tune=TUNE, force_retune=FORCE_RETUNE)
models['XGB'] = xgboost.XGBClassifier(**base, **best)
```

`CACHE_PATH`, `TUNE`, `FORCE_RETUNE` are module-level constants set near the top of each
notebook.

## No-leakage guarantee

Tuning and CV use only the training matrix. `X_test` / `X_test_imp` are never passed to
`tune_xgb` or the CV loop. Final metrics (bootstrapping, DeLong, permutation) remain
computed on the untouched test set exactly as today.

## Dependency

Add `optuna` to `pixi.toml` `[dependencies]`.

## Cost

- DAG Parameterization: ~11 DAGs × 3 configs = 33 tune calls, each up to 50 × 5 = 250
  fits (reduced by pruning). This is the heavy, one-time cost.
- Year Sensitivity: both its configs (`full`, `vitals_labs`) are already cached → no
  tuning; per-year fits reuse cached params.

## Testing

- Unit-test `config_label` mapping (3 cases).
- Unit-test cache round-trip: miss → tune (monkeypatched/tiny search) → write → hit.
- Unit-test `tune=False` returns fixed defaults and does not read/write cache.
- Smoke-test `tune_xgb` on a small synthetic imbalanced dataset (n_trials small) returns
  a params dict and runs without error.

## Reproducibility

- `TPESampler(seed=42)`, `StratifiedKFold(random_state=42)`, per-fold
  `XGBClassifier(random_state=42)`. Re-running with the same cache reproduces results;
  deleting the cache (or `force_retune=True`) re-tunes deterministically.
```
