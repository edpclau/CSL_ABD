# validate_k0.py — prove the moving-window pipeline reproduces the saved test set at k=0.
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import data_io
from config import X_TEST_CONTROL, ENSEMBLE_PRED, FEATURE_LIST_811
from pipeline import Engine
from model import load_models, predict_components, predict_ensemble


def run(n_patients=60, feat_atol=1e-3, risk_atol=5e-3):
    uids = data_io.test_uids()[:n_patients]
    raw = data_io.build_test_raw(uids)                       # compact table for the sample
    Xref = pd.read_csv(X_TEST_CONTROL, index_col=0)
    pref = pd.read_csv(ENSEMBLE_PRED, index_col=0)

    eng = Engine()
    bundle = load_models()

    feat_rows, risk_rows, ord_uids = [], [], []
    for uid, pdf in data_io.iter_patient_frames(raw, chunk_uids=uids):
        windows, bios = eng.windows_for_patient(pdf)
        w0 = [w for w in windows if w.k == 0]
        X = eng.features_for_windows(bios, w0)               # (1, 811)
        comps = predict_components(X, bundle)
        risk = predict_ensemble(comps, bundle)
        feat_rows.append(X.iloc[0].to_numpy()); risk_rows.append(risk[0]); ord_uids.append(uid)

    Xk0 = pd.DataFrame(feat_rows, index=ord_uids, columns=FEATURE_LIST_811)
    risk = pd.Series(risk_rows, index=ord_uids)

    # feature agreement
    common = [u for u in ord_uids if u in Xref.index]
    assert len(common) == len(ord_uids), \
        f"only {len(common)}/{len(ord_uids)} sampled uids found in X_test_control reference"
    diff = (Xk0.loc[common] - Xref.loc[common, FEATURE_LIST_811]).abs()
    feat_max = np.nanmax(diff.to_numpy())
    _v1 = Xk0.loc[common].to_numpy().ravel()
    _v2 = Xref.loc[common, FEATURE_LIST_811].to_numpy().ravel()
    _mask = ~np.isnan(_v1) & ~np.isnan(_v2)   # NaN from degenerate series — skip before corrcoef
    feat_corr = np.corrcoef(_v1[_mask], _v2[_mask])[0, 1]
    # risk agreement
    risk_max = (risk.loc[common] - pref.loc[common, "p_outcome"]).abs().max()
    print(f"feat max|Δ|={feat_max:.5f}  feat corr={feat_corr:.6f}  risk max|Δ|={risk_max:.5f}")
    print(f"k0 AUPRC={average_precision_score(pref.loc[common,'Outcome'], risk.loc[common]):.4f} "
          f"AUROC={roc_auc_score(pref.loc[common,'Outcome'], risk.loc[common]):.4f}")

    assert feat_corr > 0.999, f"feature correlation too low: {feat_corr}"
    assert feat_max < feat_atol, f"feature max abs diff too high: {feat_max}"
    assert risk_max < risk_atol, f"risk max abs diff too high: {risk_max}"
    return feat_max, feat_corr, risk_max


if __name__ == "__main__":
    run()
