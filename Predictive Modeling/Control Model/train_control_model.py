#!/usr/bin/env python
"""
Train the CONTROL XGBoost model on real patient data using ALL biomarkers.

This reproduces the 'Control' + 'XGB' configuration from
`DAG Parameterization.ipynb` (Experiment 1: remove_drugs=False,
remove_interventions=False). The Control configuration connects every
biomarker directly to the Outcome -- i.e. it performs NO causal feature
selection and uses the full set of biomarker features. It is the baseline
against which the causal-DAG-selected feature subsets are compared.

Data: real patient training/testing data (NOT the synthetic patients used in
`DAG Parameterization Synth.ipynb`).

Outputs (written next to this script, in 'Control Model/'):
  - control_xgb_model.json     trained XGBoost model
  - X_train_control.csv        training features actually used by the model
  - X_test_control.csv         testing features actually used by the model
  - y_train_control.csv        training labels (Outcome, one row per uid)
  - y_test_control.csv         testing labels (Outcome, one row per uid)
  - feature_list.txt           ordered list of features used
  - control_model_metrics.csv  test-set performance metrics
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
DATA_DIR = SCRIPT_DIR.parents[2] / "Data Pre-processing"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load real patient data (identical to DAG Parameterization.ipynb, cell 3)
# ---------------------------------------------------------------------------
X_train = pd.read_csv(DATA_DIR / "X_train_c12_w48_imp.csv", index_col=0)
X_test = pd.read_csv(DATA_DIR / "X_test_c12_w48_imp.csv", index_col=0)
print(f"Loaded raw:            X_train {X_train.shape}, X_test {X_test.shape}")

# Remove (near) zero-variance features; mirror surviving columns onto the test set.
X_train = X_train.loc[:, X_train.var() >= 0.01]
X_test = X_test.filter(items=X_train.columns)
print(f"After variance filter: X_train {X_train.shape}, X_test {X_test.shape}")

# The MICE-imputed matrix is used ONLY to derive the Control node/feature list,
# exactly as the notebook does.
X_train_imp = pd.read_csv(DATA_DIR / "X_train_c12_w48_mice.csv", index_col=0)

# Labels: collapse the (uid, window) multi-index to one row per patient (max).
y_train = pd.read_csv(DATA_DIR / "y_train_c12_w48_imp.csv", index_col=[0, 1]).groupby("uid").max()
y_test = pd.read_csv(DATA_DIR / "y_test_c12_w48_imp.csv", index_col=[0, 1]).groupby("uid").max()

# ---------------------------------------------------------------------------
# 2. Build the CONTROL feature set = ALL biomarkers -> Outcome
#    (mirrors dags['Control'] construction + feature mapping in the notebook)
# ---------------------------------------------------------------------------
# Control nodes: every unique biomarker (strip the _<window/stat> suffix), no Outcome.
control_nodes = [
    n
    for n in X_train_imp.columns.str.replace(r"(_.+)?$", "", regex=True).unique().tolist()
    if n != "Outcome"
]
# Features mapped to those nodes = every biomarker feature column.
features_in_dag = [
    col
    for col in X_train_imp.columns
    if any(re.search(rf"^{node}(_.+)?$", col) for node in control_nodes)
]
print(f"Control DAG:           {len(control_nodes)} biomarkers -> {len(features_in_dag)} features")

# The XGB model in the notebook trains on raw X_train filtered to these features
# (XGBoost handles missing values natively, so the un-imputed matrix is used).
X_train_model = X_train.filter(features_in_dag)
X_test_model = X_test.filter(features_in_dag)
print(f"Model matrices:        X_train {X_train_model.shape}, X_test {X_test_model.shape}")

# ---------------------------------------------------------------------------
# 3. Align labels to feature rows by patient id (uid) and sanity-check.
# ---------------------------------------------------------------------------
print(
    "Index already aligned (train/test): "
    f"{X_train_model.index.equals(y_train.index)} / {X_test_model.index.equals(y_test.index)}"
)
assert set(X_train_model.index) == set(y_train.index), "train uid mismatch between X and y"
assert set(X_test_model.index) == set(y_test.index), "test uid mismatch between X and y"
y_train = y_train.reindex(X_train_model.index)
y_test = y_test.reindex(X_test_model.index)
assert y_train["Outcome"].notna().all() and y_test["Outcome"].notna().all(), "NaN labels after align"

# ---------------------------------------------------------------------------
# 4. Train the Control XGBoost model (identical hyperparameters to notebook).
# ---------------------------------------------------------------------------
model = xgboost.XGBClassifier(
    objective="binary:logistic", random_state=RANDOM_STATE, eval_metric="aucpr"
)
model.fit(X_train_model, y_train["Outcome"])

# ---------------------------------------------------------------------------
# 5. Evaluate on the held-out real patient test set.
# ---------------------------------------------------------------------------
y_prob = model.predict_proba(X_test_model)[:, 1]
y_pred = model.predict(X_test_model)
y_true = y_test["Outcome"].astype(int).values

metrics = {
    "AUROC": roc_auc_score(y_true, y_prob),
    "Average Precision (AUPRC)": average_precision_score(y_true, y_prob),
    "Brier Score": brier_score_loss(y_true, y_prob),
    "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1": f1_score(y_true, y_pred, zero_division=0),
}
print("\n=== Control XGBoost -- real patient test set ===")
for k, v in metrics.items():
    print(f"  {k:28s}: {v:.4f}")
print(
    f"  {'Test positives':28s}: {int(y_true.sum())}/{len(y_true)} "
    f"({y_true.mean():.3%} prevalence)"
)

# ---------------------------------------------------------------------------
# 6. Persist model, data, features and metrics.
# ---------------------------------------------------------------------------
model.save_model(str(OUT_DIR / "control_xgb_model.json"))
X_train_model.to_csv(OUT_DIR / "X_train_control.csv")
X_test_model.to_csv(OUT_DIR / "X_test_control.csv")
y_train[["Outcome"]].to_csv(OUT_DIR / "y_train_control.csv")
y_test[["Outcome"]].to_csv(OUT_DIR / "y_test_control.csv")
(OUT_DIR / "feature_list.txt").write_text("\n".join(X_train_model.columns) + "\n")
pd.DataFrame([{"Metric": k, "Value": round(float(v), 4)} for k, v in metrics.items()]).to_csv(
    OUT_DIR / "control_model_metrics.csv", index=False
)

print(f"\nSaved model + train/test data + metrics to:\n  {OUT_DIR}")
