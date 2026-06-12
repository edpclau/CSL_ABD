#!/usr/bin/env python
"""
SELECTED ensemble model for the neuro-morbidity Outcome.

Architecture (final, chosen configuration):
    biomarkers (811)
      -> tuned per-component XGBoost heads: EEG, CT, MRI   (tuned/head_*_tuned.json)
      -> [EEG_pred, CT_pred, MRI_pred]
      -> tuned shallow XGBoost meta-learner               (tuned/meta_xgb_tuned.json)
      -> P(Outcome)

Usage:
    from predict_ensemble import load_ensemble, predict_outcome
    heads, meta = load_ensemble()
    p = predict_outcome(X, heads, meta)   # X: DataFrame with the 811 biomarker columns

Run directly to score the held-out test set and verify the saved model reproduces
the selected metrics (AUROC 0.9229, AUPRC 0.7913).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

SCRIPT_DIR = Path(__file__).resolve().parent
TUNED = SCRIPT_DIR / "tuned"
CONTROL_DIR = SCRIPT_DIR.parent / "Control Model"
COMPONENTS = ["EEG", "CT", "MRI"]  # column order the meta-learner was trained on


def load_ensemble():
    heads = {}
    for c in COMPONENTS:
        m = xgboost.XGBClassifier()
        m.load_model(str(TUNED / f"head_{c}_tuned.json"))
        heads[c] = m
    meta = xgboost.XGBClassifier()
    meta.load_model(str(TUNED / "meta_xgb_tuned.json"))
    return heads, meta


def predict_outcome(X, heads, meta):
    """P(Outcome=1) for each row of X (DataFrame of the 811 biomarker features)."""
    feats = np.column_stack([heads[c].predict_proba(X)[:, 1] for c in COMPONENTS])
    return meta.predict_proba(feats)[:, 1]


if __name__ == "__main__":
    from sklearn.metrics import (
        average_precision_score, balanced_accuracy_score, brier_score_loss,
        f1_score, roc_auc_score,
    )

    X_test = pd.read_csv(CONTROL_DIR / "X_test_control.csv", index_col=0)
    y_test = pd.read_csv(CONTROL_DIR / "outcome_components_test.csv", index_col=0)["Outcome"].astype(int)

    heads, meta = load_ensemble()
    p = predict_outcome(X_test, heads, meta)

    pd.DataFrame({"Outcome": y_test.values, "p_outcome": p}, index=X_test.index).to_csv(
        SCRIPT_DIR / "ensemble_final_test_predictions.csv"
    )
    yhat = (p >= 0.5).astype(int)
    print("Selected ensemble (tuned heads + tuned XGB meta) -- test set (n=%d):" % len(y_test))
    print(f"  AUROC        {roc_auc_score(y_test, p):.4f}   (expect 0.9229)")
    print(f"  AUPRC        {average_precision_score(y_test, p):.4f}   (expect 0.7913)")
    print(f"  Brier        {brier_score_loss(y_test, p):.4f}")
    print(f"  Balanced Acc {balanced_accuracy_score(y_test, yhat):.4f}")
    print(f"  F1 @ 0.5     {f1_score(y_test, yhat, zero_division=0):.4f}")
    print(f"\nSaved final test predictions to: {SCRIPT_DIR / 'ensemble_final_test_predictions.csv'}")
