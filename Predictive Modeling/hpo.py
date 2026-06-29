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
