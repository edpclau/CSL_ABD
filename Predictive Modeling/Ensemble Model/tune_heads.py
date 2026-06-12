#!/usr/bin/env python
"""
Hyperparameter-tune each per-component XGBoost head (EEG, CT, MRI) to maximize
AUPRC (average precision), then rebuild the stacking ensemble with the tuned
heads and compare against the default heads and the control model.

Protocol (no test peeking):
  - Optuna TPE, 50 trials/head. Objective = mean 3-fold CV AUPRC on TRAIN.
    Early stopping uses a nested split inside each fold; the scored fold is never
    used for early stopping, so the objective has no within-fold leakage.
  - Best params -> one full-train fit with an early-stopping split fixes the final
    n_estimators (best_iteration).
  - Tuned heads deployed with (best_params, fixed n_estimators) to regenerate the
    5-fold out-of-fold meta-features and the refit-on-full-train test predictions.
  - Meta-learners (LogReg / Noisy-OR / shallow XGB) re-fit on the tuned features.
  - Everything evaluated on the same untouched test individuals.

Outputs -> Ensemble Model/tuned/
"""

import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
N_TRIALS = 50
N_JOBS = 10
COMPONENTS = ["EEG", "CT", "MRI"]

SCRIPT_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCRIPT_DIR.parent / "Control Model"
OUT = SCRIPT_DIR / "tuned"
OUT.mkdir(exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- Data (aligned by uid) ----
X_train = pd.read_csv(CONTROL_DIR / "X_train_control.csv", index_col=0)
X_test = pd.read_csv(CONTROL_DIR / "X_test_control.csv", index_col=0)
comp_tr = pd.read_csv(CONTROL_DIR / "outcome_components_train.csv", index_col=0)
comp_te = pd.read_csv(CONTROL_DIR / "outcome_components_test.csv", index_col=0)
y_train_out = comp_tr["Outcome"].astype(int)
y_test_out = comp_te["Outcome"].astype(int)
print(f"X_train {X_train.shape}, X_test {X_test.shape}", flush=True)


def tune_component(comp):
    y = comp_tr[comp].astype(int).values
    spw_max = max(2.0, (y == 0).sum() / max(1, (y == 1).sum()))  # neg/pos ratio
    splits = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X_train, y))

    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 12),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, spw_max, log=True),
        )
        aps = []
        for tr_idx, va_idx in splits:
            Xtr, ytr = X_train.iloc[tr_idx], y[tr_idx]
            Xf, Xe, yf, ye = train_test_split(
                Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED
            )
            m = xgboost.XGBClassifier(
                n_estimators=2000, eval_metric="aucpr", early_stopping_rounds=40,
                random_state=SEED, n_jobs=N_JOBS, **params,
            )
            m.fit(Xf, yf, eval_set=[(Xe, ye)], verbose=False)
            p = m.predict_proba(X_train.iloc[va_idx])[:, 1]
            aps.append(average_precision_score(y[va_idx], p))
        return float(np.mean(aps))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))

    def log_cb(st, tr):
        if (tr.number + 1) % 10 == 0:
            print(f"  [{comp}] trial {tr.number + 1}/{N_TRIALS}  best CV-AUPRC={st.best_value:.4f}", flush=True)

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_cb])

    # Fix final n_estimators via one early-stopping fit on full train.
    Xf, Xe, yf, ye = train_test_split(X_train, y, test_size=0.15, stratify=y, random_state=SEED)
    probe = xgboost.XGBClassifier(
        n_estimators=2000, eval_metric="aucpr", early_stopping_rounds=40,
        random_state=SEED, n_jobs=N_JOBS, **study.best_params,
    )
    probe.fit(Xf, yf, eval_set=[(Xe, ye)], verbose=False)
    n_est = int(probe.best_iteration) + 1

    best = {"component": comp, "cv_auprc": round(study.best_value, 4),
            "n_estimators": n_est, "scale_pos_weight_max": round(spw_max, 2),
            **{k: round(v, 5) if isinstance(v, float) else v for k, v in study.best_params.items()}}
    (OUT / f"best_params_{comp}.json").write_text(json.dumps(best, indent=2))
    print(f"[{comp}] DONE  CV-AUPRC={study.best_value:.4f}  n_estimators={n_est}", flush=True)
    return best


def tuned_clf(best):
    p = {k: v for k, v in best.items()
         if k not in ("component", "cv_auprc", "scale_pos_weight_max")}
    return xgboost.XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", random_state=SEED, n_jobs=N_JOBS, **p
    )


# ---- Tune all three heads ----
best_params = {c: tune_component(c) for c in COMPONENTS}

# ---- Deploy tuned heads: 5-fold OOF + refit-on-full-train test preds ----
skf5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
folds5 = list(skf5.split(X_train, y_train_out))
oof = pd.DataFrame(index=X_train.index, columns=[f"{c}_pred" for c in COMPONENTS], dtype=float)
test_feats = pd.DataFrame(index=X_test.index, columns=[f"{c}_pred" for c in COMPONENTS], dtype=float)
head_perf = {}

print("\nDeploying tuned heads (5-fold OOF + full-train refit):", flush=True)
for comp in COMPONENTS:
    yc_tr = comp_tr[comp].astype(int).values
    col = oof.columns.get_loc(f"{comp}_pred")
    for tr_idx, va_idx in folds5:
        m = tuned_clf(best_params[comp])
        m.fit(X_train.iloc[tr_idx], yc_tr[tr_idx])
        oof.iloc[va_idx, col] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
    m = tuned_clf(best_params[comp])
    m.fit(X_train, yc_tr)
    m.save_model(str(OUT / f"head_{comp}_tuned.json"))
    p_te = m.predict_proba(X_test)[:, 1]
    test_feats[f"{comp}_pred"] = p_te
    yc_te = comp_te[comp].astype(int).values
    head_perf[comp] = {
        "tuned_test_AUPRC": round(average_precision_score(yc_te, p_te), 4),
        "tuned_test_AUROC": round(roc_auc_score(yc_te, p_te), 4),
        "tuned_oof_AUPRC": round(average_precision_score(yc_tr, oof[f"{comp}_pred"].values), 4),
        "cv_auprc": best_params[comp]["cv_auprc"],
    }
    print(f"  {comp}: tuned test AUPRC {head_perf[comp]['tuned_test_AUPRC']:.4f} "
          f"AUROC {head_perf[comp]['tuned_test_AUROC']:.4f}", flush=True)

assert not oof.isna().any().any()
oof.to_csv(OUT / "oof_train_components_tuned.csv")
test_feats.assign(Outcome=y_test_out.values).to_csv(OUT / "test_components_tuned.csv")

# ---- Meta-learners on tuned features ----
Xm_tr, Xm_te = oof.values, test_feats.values
logreg = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xm_tr, y_train_out.values)
joblib.dump(logreg, OUT / "meta_logreg_tuned.joblib")
p_lr = logreg.predict_proba(Xm_te)[:, 1]
p_nor = 1.0 - np.prod(1.0 - Xm_te, axis=1)
xgb_meta = xgboost.XGBClassifier(
    objective="binary:logistic", random_state=SEED, eval_metric="aucpr",
    max_depth=2, n_estimators=60, learning_rate=0.1,
).fit(Xm_tr, y_train_out.values)
xgb_meta.save_model(str(OUT / "meta_xgb_tuned.json"))
p_xgb = xgb_meta.predict_proba(Xm_te)[:, 1]

ctrl = xgboost.XGBClassifier()
ctrl.load_model(str(CONTROL_DIR / "control_xgb_model.json"))
p_ctrl = ctrl.predict_proba(X_test)[:, 1]


def ev(name, p):
    yhat = (p >= 0.5).astype(int)
    return {"Model": name,
            "AUROC": round(roc_auc_score(y_test_out, p), 4),
            "AUPRC": round(average_precision_score(y_test_out, p), 4),
            "Brier": round(brier_score_loss(y_test_out, p), 4),
            "BalancedAcc": round(balanced_accuracy_score(y_test_out, yhat), 4),
            "F1": round(f1_score(y_test_out, yhat, zero_division=0), 4)}


metrics = pd.DataFrame([
    ev("Ensemble (tuned) -- LogReg meta", p_lr),
    ev("Ensemble (tuned) -- Noisy-OR", p_nor),
    ev("Ensemble (tuned) -- XGB meta", p_xgb),
    ev("Control (all biomarkers)", p_ctrl),
])
metrics.to_csv(OUT / "tuned_metrics.csv", index=False)

(OUT / "tuning_summary.json").write_text(json.dumps(
    {"components": COMPONENTS, "n_trials": N_TRIALS, "best_params": best_params,
     "head_perf": head_perf, "metrics": metrics.to_dict(orient="records")}, indent=2))

print("\n=== Per-head AUPRC: default vs tuned (test) ===", flush=True)
default_head_auprc = {"EEG": 0.2769, "CT": 0.2759, "MRI": 0.2182}  # from the default ensemble run
for c in COMPONENTS:
    print(f"  {c}: default {default_head_auprc[c]:.4f} -> tuned {head_perf[c]['tuned_test_AUPRC']:.4f} "
          f"(CV {head_perf[c]['cv_auprc']:.4f})", flush=True)
print("\n=== Ensemble test metrics (tuned heads) ===", flush=True)
print(metrics.to_string(index=False), flush=True)
print(f"\nSaved tuned artifacts to: {OUT}", flush=True)
