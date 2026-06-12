#!/usr/bin/env python
"""
F1 at each model's optimal probability cutoff (vs the default 0.5).

Reports two thresholds per model:
  - test-optimal : threshold that maximizes F1 ON THE TEST SET (oracle/upper
                   bound -- optimistic, since the cutoff is tuned on the same
                   data it is scored on).
  - train-chosen : threshold that maximizes F1 on TRAIN predictions, then applied
                   to the test set (deployment-realistic). For the control model
                   train preds are in-sample; for the meta-models they are the
                   cross-fitted OOF features the meta was fit on -- both mildly
                   optimistic, but neither peeks at the test labels.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import f1_score, precision_recall_curve

SCRIPT_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCRIPT_DIR.parent / "Control Model"

# Labels
comp_tr = pd.read_csv(CONTROL_DIR / "outcome_components_train.csv", index_col=0)
comp_te = pd.read_csv(CONTROL_DIR / "outcome_components_test.csv", index_col=0)
y_tr = comp_tr["Outcome"].astype(int).values
y_te = comp_te["Outcome"].astype(int).values

# Meta features (cross-fitted OOF for train, refit-head preds for test)
oof = pd.read_csv(SCRIPT_DIR / "oof_train_components.csv", index_col=0)[["EEG_pred", "CT_pred", "MRI_pred"]]
tef = pd.read_csv(SCRIPT_DIR / "test_components.csv", index_col=0)[["EEG_pred", "CT_pred", "MRI_pred"]]

# Reconstruct each model's (train_prob, test_prob)
probs = {}
Xtr = pd.read_csv(CONTROL_DIR / "X_train_control.csv", index_col=0)
Xte = pd.read_csv(CONTROL_DIR / "X_test_control.csv", index_col=0)
ctrl = xgboost.XGBClassifier(); ctrl.load_model(str(CONTROL_DIR / "control_xgb_model.json"))
probs["Control (all biomarkers)"] = (ctrl.predict_proba(Xtr)[:, 1], ctrl.predict_proba(Xte)[:, 1])

lr = joblib.load(SCRIPT_DIR / "meta_logreg.joblib")
probs["Ensemble -- LogReg meta"] = (lr.predict_proba(oof.values)[:, 1], lr.predict_proba(tef.values)[:, 1])

xm = xgboost.XGBClassifier(); xm.load_model(str(SCRIPT_DIR / "meta_xgb.json"))
probs["Ensemble -- XGB meta"] = (xm.predict_proba(oof.values)[:, 1], xm.predict_proba(tef.values)[:, 1])

probs["Ensemble -- Noisy-OR"] = (
    1 - np.prod(1 - oof.values, axis=1), 1 - np.prod(1 - tef.values, axis=1)
)


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


order = [
    "Ensemble -- LogReg meta",
    "Ensemble -- Noisy-OR",
    "Ensemble -- XGB meta",
    "Control (all biomarkers)",
]
rows = []
for name in order:
    p_tr, p_te = probs[name]
    thr_te, f1_te_opt = best_f1_threshold(y_te, p_te)          # test-optimal (oracle)
    thr_tr, _ = best_f1_threshold(y_tr, p_tr)                  # chosen on train
    f1_train_thr = f1_score(y_te, (p_te >= thr_tr).astype(int), zero_division=0)
    f1_05 = f1_score(y_te, (p_te >= 0.5).astype(int), zero_division=0)
    rows.append({
        "Model": name,
        "F1@0.5": round(f1_05, 4),
        "F1@test-opt": round(f1_te_opt, 4),
        "thr_test": round(thr_te, 4),
        "F1@train-thr": round(f1_train_thr, 4),
        "thr_train": round(thr_tr, 4),
    })

res = pd.DataFrame(rows)
res.to_csv(SCRIPT_DIR / "f1_optimal_threshold.csv", index=False)
print(res.to_string(index=False))
