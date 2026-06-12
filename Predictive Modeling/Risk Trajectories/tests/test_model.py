# tests/test_model.py
import numpy as np
import pandas as pd
from scipy.special import expit
import model as M
from config import FEATURE_LIST_811, COMPONENTS

def test_shap_additivity_per_head():
    bundle = M.load_models()
    X = pd.DataFrame(np.random.rand(5, 811).astype(np.float32), columns=FEATURE_LIST_811)
    comps = M.predict_components(X, bundle)           # (5,3)
    shap = M.head_shap(X, bundle)                     # dict head -> (5, 812)
    for j, h in enumerate(COMPONENTS):
        recon = expit(shap[h].sum(axis=1))
        assert np.allclose(recon, comps[:, j], atol=1e-5)

def test_ensemble_risk_in_unit_interval():
    bundle = M.load_models()
    X = pd.DataFrame(np.random.rand(5, 811).astype(np.float32), columns=FEATURE_LIST_811)
    comps = M.predict_components(X, bundle)
    risk = M.predict_ensemble(comps, bundle)
    assert risk.shape == (5,)
    assert ((risk >= 0) & (risk <= 1)).all()

def test_meta_shap_additivity():
    bundle = M.load_models()
    comps = np.random.rand(4, 3).astype(np.float32)
    mshap = M.meta_shap(comps, bundle)                # (4,4)
    risk = M.predict_ensemble(comps, bundle)
    assert np.allclose(expit(mshap.sum(axis=1)), risk, atol=1e-5)
