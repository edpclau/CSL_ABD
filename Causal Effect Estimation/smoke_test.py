#!/usr/bin/env python
"""
Smoke test for the EconML env: build a CausalForest, then pull SHAP values from it.

Runs end-to-end on synthetic data with a KNOWN heterogeneous treatment effect, so
it both verifies the environment and demonstrates the workflow:
  1. fit a CausalForestDML (CATE estimator with confounding adjustment),
  2. check it recovers the true effect heterogeneity,
  3. compute est.shap_values(X) -- TreeSHAP on the CATE model -- and confirm it
     highlights the true effect modifiers (X0, X1).

This is the SHAP-of-a-CATE-model case: the SHAP values explain *who the treatment
helps and why*, not a risk prediction.
"""
import numpy as np
import econml
import shap
import sklearn
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

print(f"econml {econml.__version__} | shap {shap.__version__} | sklearn {sklearn.__version__}")

# --- Synthetic data with a known CATE that depends only on X0, X1 ---
rng = np.random.default_rng(0)
n, d = 3000, 5
X = rng.normal(size=(n, d))                      # candidate effect modifiers
W = rng.normal(size=(n, 3))                      # confounders / controls
true_tau = 1.0 + 1.5 * (X[:, 0] > 0) + 0.8 * X[:, 1]
propensity = 1.0 / (1.0 + np.exp(-(0.6 * X[:, 0] + 0.5 * W[:, 0])))  # confounded assignment
T = rng.binomial(1, propensity)
Y = true_tau * T + 0.7 * X[:, 2] + 0.5 * W[:, 1] + rng.normal(scale=0.5, size=n)

# --- CausalForest (DML) ---
est = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=10, random_state=0),
    model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=10, random_state=0),
    discrete_treatment=True,
    n_estimators=600,
    min_samples_leaf=10,
    random_state=0,
)
est.fit(Y, T, X=X, W=W)

tau_hat = est.effect(X)
corr = float(np.corrcoef(tau_hat, true_tau)[0, 1])
print(f"CATE recovery: mean est {tau_hat.mean():.3f} (true {true_tau.mean():.3f}) | "
      f"corr(est, true) = {corr:.3f}")
assert corr > 0.3, "CausalForest failed to recover the known heterogeneity"

# --- SHAP on the CATE model ---
shap_dict = est.shap_values(X)
print(f"shap_values() returned: {type(shap_dict).__name__}")
try:
    out_key = list(shap_dict.keys())[0]
    trt_key = list(shap_dict[out_key].keys())[0]
    expl = shap_dict[out_key][trt_key]
    mean_abs = np.abs(expl.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    print(f"SHAP Explanation shape {expl.values.shape}  (outcome '{out_key}', treatment '{trt_key}')")
    print("Mean|SHAP| of CATE per feature (descending):")
    for i in order:
        print(f"   X{i}: {mean_abs[i]:.4f}")
    top2 = set(order[:2].tolist())
    print(f"\nTop-2 effect modifiers: {sorted(top2)}  (expected {{0, 1}})")
    assert top2 == {0, 1}, "SHAP did not flag the true effect modifiers X0, X1"
except (AttributeError, IndexError, KeyError) as e:
    print(f"Could not parse nested SHAP structure ({e}); raw keys: {list(shap_dict)}")
    raise

print("\nSMOKE TEST PASSED")
