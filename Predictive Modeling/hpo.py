"""XGBoost hyperparameter optimization with a per-(DAG, config) JSON cache.

Tuning uses Optuna over a StratifiedKFold loop scoring average_precision on
TRAINING data only. See docs/superpowers/specs/2026-06-29-xgb-hyperparameter-optimization-design.md
"""

FIXED_DEFAULTS = dict(objective="binary:logistic", random_state=42, eval_metric="aucpr")


def config_label(remove_drugs, remove_interventions):
    """Canonical config string shared as a cache key across both notebooks."""
    if remove_drugs and remove_interventions:
        return "vitals_labs"
    if remove_drugs:
        return "no_drugs"
    return "full"


import json
import os
from datetime import datetime


def _load_cache(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(path, cache):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def tune_xgb(X, y, n_trials=50, cv=5, seed=42):
    """Placeholder — full implementation in Task 4."""
    raise NotImplementedError("tune_xgb not yet implemented")


def get_xgb_params(dag_name, cfg_label, X, y, cache_path,
                   tune=True, force_retune=False, n_trials=50, cv=5, seed=42):
    """Return tuned XGB params for (dag_name, cfg_label), using a JSON cache.

    tune=False -> {} (reproduces fixed defaults, no cache I/O).
    Cache hit  -> stored params. Miss/force_retune -> tune, cache, return.
    """
    if not tune:
        return {}
    key = f"{dag_name}||{cfg_label}"
    cache = _load_cache(cache_path)
    if key in cache and not force_retune:
        return cache[key]["params"]
    params = tune_xgb(X, y, n_trials=n_trials, cv=cv, seed=seed)
    cache[key] = {
        "params": params,
        "n_features": int(X.shape[1]),
        "n_trials": n_trials,
        "cv": cv,
        "objective": "average_precision",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    _save_cache(cache_path, cache)
    return params
