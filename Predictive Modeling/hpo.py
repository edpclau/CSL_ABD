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


import numpy as np
import optuna
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score


def tune_xgb(X, y, n_trials=50, cv=5, seed=42):
    """Optuna TPE search (MedianPruner) maximizing CV average_precision.

    Uses TRAINING data only. Returns study.best_params.
    """
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    yv = np.asarray(y)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = []
        for fold, (tr, va) in enumerate(skf.split(Xv, yv)):
            model = xgboost.XGBClassifier(**FIXED_DEFAULTS, **params)
            model.fit(Xv[tr], yv[tr])
            prob = model.predict_proba(Xv[va])[:, 1]
            scores.append(average_precision_score(yv[va], prob))
            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


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
