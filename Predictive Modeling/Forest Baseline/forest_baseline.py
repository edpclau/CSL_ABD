#!/usr/bin/env python
"""
Predictive forest baseline for the neuro-morbidity Outcome.

EconML's `grf.RegressionForest` is a *predictive* (honest) regression forest --
NOT a causal method. It is not installed in this env, and pinning it risks a
scikit-learn downgrade, so this uses the equivalent sklearn forests:
  - RandomForestRegressor on the 0/1 Outcome  (mirrors RegressionForest exactly:
    leaf means of binary labels are P(Outcome) in [0,1])
  - RandomForestClassifier (predict_proba, class_weight='balanced')

Forests can't ingest the NaNs XGBoost tolerated, so we use the dense MICE matrix
restricted to the 811 control features. An XGBoost-on-MICE row isolates the model
effect (forest vs XGB) from the imputation effect (MICE vs the raw-NaN control).

Evaluated on the same test individuals as the control model and tuned ensemble.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, roc_auc_score,
)

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
CM = SCRIPT_DIR.parent / "Control Model"
ENS = SCRIPT_DIR.parent / "Ensemble Model"
DP = SCRIPT_DIR.parents[2] / "Data Pre-processing"

feats = open(CM / "feature_list.txt").read().splitlines()
ytr = pd.read_csv(CM / "outcome_components_train.csv", index_col=0)["Outcome"].astype(int)
yte = pd.read_csv(CM / "outcome_components_test.csv", index_col=0)["Outcome"].astype(int)


def load_mice(fn, idx):
    df = pd.read_csv(fn, index_col=0)[feats].reindex(idx)
    assert not df.isna().any().any(), f"NaNs remain in {fn.name}"
    return df


Xtr = load_mice(DP / "X_train_c12_w48_mice.csv", ytr.index)
Xte = load_mice(DP / "X_test_c12_w48_mice.csv", yte.index)
print(f"MICE (dense) shapes -- train {Xtr.shape}, test {Xte.shape}")


def ev(name, p):
    yhat = (p >= 0.5).astype(int)
    return {"Model": name,
            "AUROC": round(roc_auc_score(yte, p), 4),
            "AUPRC": round(average_precision_score(yte, p), 4),
            "Brier": round(brier_score_loss(yte, p), 4),
            "BalAcc": round(balanced_accuracy_score(yte, yhat), 4),
            "F1": round(f1_score(yte, yhat, zero_division=0), 4)}


rows = []

# 1. RandomForestRegressor on 0/1  -- the RegressionForest analog
rfr = RandomForestRegressor(
    n_estimators=400, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED
).fit(Xtr, ytr.values)
rows.append(ev("RF Regressor on 0/1  (RegressionForest analog) [MICE]", np.clip(rfr.predict(Xte), 0, 1)))

# 2. RandomForestClassifier
rfc = RandomForestClassifier(
    n_estimators=400, min_samples_leaf=5, max_features="sqrt",
    class_weight="balanced", n_jobs=-1, random_state=SEED
).fit(Xtr, ytr.values)
rows.append(ev("RF Classifier (balanced) [MICE]", rfc.predict_proba(Xte)[:, 1]))

# 3. XGBoost on MICE -- controlled comparator (isolates model from imputation)
xgbm = xgboost.XGBClassifier(
    objective="binary:logistic", eval_metric="aucpr", random_state=SEED
).fit(Xtr, ytr.values)
rows.append(ev("XGBoost [MICE]  (controlled comparator)", xgbm.predict_proba(Xte)[:, 1]))

# 4. Reference: control XGBoost on raw (NaN) data
ctrl = xgboost.XGBClassifier(); ctrl.load_model(str(CM / "control_xgb_model.json"))
Xte_raw = pd.read_csv(CM / "X_test_control.csv", index_col=0)
rows.append(ev("Control XGBoost [raw NaN]  (reference)", ctrl.predict_proba(Xte_raw)[:, 1]))

# 5. Reference: tuned ensemble (saved predictions)
ens = pd.read_csv(ENS / "ensemble_final_test_predictions.csv", index_col=0).reindex(yte.index)
rows.append(ev("Tuned ensemble  (reference)", ens["p_outcome"].values))

res = pd.DataFrame(rows)
res.to_csv(SCRIPT_DIR / "forest_baseline_metrics.csv", index=False)
print("\n" + res.to_string(index=False))
print(f"\nSaved metrics to: {SCRIPT_DIR / 'forest_baseline_metrics.csv'}")
