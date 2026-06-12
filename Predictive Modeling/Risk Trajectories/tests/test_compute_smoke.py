# tests/test_compute_smoke.py
import json
import numpy as np
import pandas as pd
import compute_trajectories as C
import data_io
from config import FEATURE_LIST_811

def test_compute_small_subset(tmp_path):
    uids = data_io.test_uids()[:5]
    out = C.run(uids=uids, chunk_size=2, out_dir=tmp_path)
    risk = pd.read_parquet(out["risk_parquet"])
    assert {"uid", "k", "ensemble_risk", "EEG_p", "CT_p", "MRI_p", "y_true", "n_observed"} <= set(risk.columns)
    assert (risk["k"] == 0).sum() == 5                      # every patient has k=0
    assert ((risk["ensemble_risk"] >= 0) & (risk["ensemble_risk"] <= 1)).all()
    # one shap file per patient, shape (n_windows, 812) per head
    u0 = uids[0]
    z = np.load(out["shap_dir"] / f"{u0}.npz")
    n_w = (risk["uid"] == u0).sum()
    assert z["eeg"].shape == (n_w, 812)
    assert z["meta"].shape == (n_w, 4)
    assert z["k"].shape == (n_w,)
    names = json.loads((out["shap_dir"] / "feature_names.json").read_text())
    assert names == FEATURE_LIST_811
