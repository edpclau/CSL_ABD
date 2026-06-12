"""Faithful biomarker -> final-outcome SHAP for the stacked ensemble, via the
full predict_outcome pipeline (interventional, background-based).

Two explainers (both estimate the same interventional Shapley of the composed
model; see shap_attribution_compare.py for why these beat composing head/meta SHAP):

  method="permutation": shap.PermutationExplainer  -- robust, no l1_reg footgun,
      preferred default for the fused outcome attribution.
  method="partition":   shap.PartitionExplainer    -- Owen values over a hierarchy
      that groups each biomarker's catch22 features together (fast, and credit is
      coherent within a biomarker; aggregates cleanly to per-biomarker importance).

Usage:
    from outcome_shap import OutcomeAttributor
    att = OutcomeAttributor(background_X)            # background = DataFrame of 811 features
    phi = att.attribute(X, method="permutation")     # (n, 811) outcome-logit SHAP
    by_bio = att.by_biomarker(phi)                   # (n, 45) summed within biomarker
"""
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import xgboost as xgb
import shap

from config import FEATURE_LIST_811, BIOMARKER_COLS
from model import load_models, predict_components

_PREFIX = np.array([f.rsplit("__", 1)[0] for f in FEATURE_LIST_811])


def _biomarker_clustering():
    """Hierarchy that merges same-biomarker catch22 features first (distance 0),
    then biomarkers -> Owen values respect the biomarker grouping."""
    D = (_PREFIX[:, None] != _PREFIX[None, :]).astype(float)
    return sch.linkage(squareform(D, checks=False), method="average")


class OutcomeAttributor:
    def __init__(self, background_X):
        self.b = load_models()
        self.bg = np.asarray(background_X, np.float32)
        self._meta_booster = self.b.meta.get_booster()
        self._part_masker = shap.maskers.Partition(self.bg, clustering=_biomarker_clustering())

    def _f(self, Z):
        comps = predict_components(pd.DataFrame(Z, columns=FEATURE_LIST_811), self.b).astype(np.float32)
        d = xgb.DMatrix(comps, feature_names=self._meta_booster.feature_names)
        return self._meta_booster.predict(d, output_margin=True)   # final outcome logit

    def attribute(self, X, method="permutation", max_evals=8000):
        X = np.asarray(X, np.float32)
        if method == "permutation":
            expl = shap.PermutationExplainer(self._f, shap.maskers.Independent(self.bg))
        elif method == "partition":
            expl = shap.PartitionExplainer(self._f, self._part_masker)
        else:
            raise ValueError(f"unknown method {method!r} (use 'permutation' or 'partition')")
        return np.asarray(expl(X, max_evals=max_evals).values)

    @staticmethod
    def by_biomarker(phi):
        """Sum (n,811) feature attributions within each of the 45 biomarkers."""
        out = np.zeros((phi.shape[0], len(BIOMARKER_COLS)))
        for bi, name in enumerate(BIOMARKER_COLS):
            out[:, bi] = phi[:, _PREFIX == name].sum(1)
        return pd.DataFrame(out, columns=BIOMARKER_COLS)


# ---- validation / comparison entry point ----
if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from config import X_TEST_CONTROL
    rng = np.random.default_rng(0)                       # identical split to shap_attribution_compare
    X = pd.read_csv(X_TEST_CONTROL, index_col=0)[FEATURE_LIST_811]
    sel = rng.choice(len(X), size=24 + 50, replace=False)
    Xn = X.iloc[sel[:24]].to_numpy(np.float32)
    bg = X.iloc[sel[24:].tolist()].to_numpy(np.float32)
    ref = np.load("artifacts/shap_compare.npz")["ks_ref"]   # converged 16k KernelSHAP

    att = OutcomeAttributor(bg)
    perm = att.attribute(Xn, "permutation", max_evals=8000)
    part = att.attribute(Xn, "partition", max_evals=8000)

    def cmp(M, R):
        pe = sg = ja = 0.0
        for i in range(M.shape[0]):
            top = np.argsort(np.abs(R[i]))[::-1][:50]
            t = set(np.argsort(np.abs(R[i]))[::-1][:20]); tm = set(np.argsort(np.abs(M[i]))[::-1][:20])
            pe += np.corrcoef(R[i, top], M[i, top])[0, 1]
            sg += np.mean(np.sign(R[i, list(t)]) == np.sign(M[i, list(t)]))
            ja += len(t & tm) / len(t | tm)
        n = M.shape[0]; return pe/n, sg/n, ja/n

    print("vs converged 16k-KernelSHAP reference (n=24):")
    for name, M in [("permutation", perm), ("partition", part)]:
        p, s, j = cmp(M, ref)
        print(f"   {name:<12} pearson_top50={p:.3f}  sign_top20={s:.3f}  top20_jac={j:.3f}")
    p, s, j = cmp(part, perm)
    print(f"   partition vs permutation: pearson_top50={p:.3f}  sign_top20={s:.3f}  top20_jac={j:.3f}")
    # efficiency: both sum to f(x) - E_bg[f]
    fx = att._f(Xn); base = att._f(bg).mean()
    print(f"   efficiency  permutation max|sum-Δ|={np.max(np.abs(perm.sum(1)-(fx-base))):.2e}  "
          f"partition={np.max(np.abs(part.sum(1)-(fx-base))):.2e}")
    np.savez_compressed("artifacts/outcome_shap_variants.npz",
                        permutation=perm, partition=part, instance_index=sel[:24],
                        feature_names=np.array(FEATURE_LIST_811))
    print("   saved -> artifacts/outcome_shap_variants.npz")
