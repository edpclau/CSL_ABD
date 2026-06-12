# model.py — load the tuned ensemble; ensemble risk + exact TreeSHAP.
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb
from config import ENS_TUNED, COMPONENTS


@dataclass(frozen=True)
class Bundle:
    heads: dict          # component -> XGBClassifier
    meta: xgb.XGBClassifier
    feat_names: list     # 811 head feature names (from booster)


def load_models() -> Bundle:
    heads = {}
    for c in COMPONENTS:
        m = xgb.XGBClassifier()
        m.load_model(str(ENS_TUNED / f"head_{c}_tuned.json"))
        heads[c] = m
    meta = xgb.XGBClassifier()
    meta.load_model(str(ENS_TUNED / "meta_xgb_tuned.json"))
    feat_names = heads[COMPONENTS[0]].get_booster().feature_names
    assert all(heads[c].get_booster().feature_names == feat_names for c in COMPONENTS), \
        "head feature names diverge across components"
    return Bundle(heads=heads, meta=meta, feat_names=feat_names)


def predict_components(X811: pd.DataFrame, b: Bundle) -> np.ndarray:
    """(n,3) component probabilities in COMPONENTS order."""
    return np.column_stack([b.heads[c].predict_proba(X811)[:, 1] for c in COMPONENTS])


def predict_ensemble(components: np.ndarray, b: Bundle) -> np.ndarray:
    """(n,) ensemble P(Outcome) from the 3 component probabilities."""
    return b.meta.predict_proba(np.asarray(components, dtype=np.float32))[:, 1]


def head_shap(X811: pd.DataFrame, b: Bundle) -> dict:
    """Exact TreeSHAP per head vs the 811 biomarker features: head -> (n, 812)."""
    out = {}
    for c in COMPONENTS:
        booster = b.heads[c].get_booster()
        d = xgb.DMatrix(X811.to_numpy(dtype=np.float32), feature_names=booster.feature_names)
        out[c] = booster.predict(d, pred_contribs=True)     # (n, 812)
    return out


def meta_shap(components: np.ndarray, b: Bundle) -> np.ndarray:
    """Exact TreeSHAP of the meta-learner vs the 3 components: (n, 4)."""
    booster = b.meta.get_booster()
    d = xgb.DMatrix(np.asarray(components, dtype=np.float32), feature_names=booster.feature_names)
    return booster.predict(d, pred_contribs=True)           # (n, 4) = 3 comps + bias
