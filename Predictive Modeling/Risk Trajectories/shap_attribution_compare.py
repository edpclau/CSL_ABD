"""Compare two biomarker-level attributions of the FINAL ensemble outcome logit:

  (A) chain-rule: redistribute each component's meta-SHAP (phi_EEG/CT/MRI) across
      biomarkers by that head's SHAP shares -> additive by construction.
  (B) KernelSHAP on the full predict_outcome pipeline (biomarkers -> final logit).

To compare apples-to-apples, EVERYTHING uses the same Shapley formulation:
INTERVENTIONAL Shapley w.r.t. a shared background (KernelSHAP is interventional,
so the heads/meta SHAP feeding the chain-rule use interventional TreeExplainer
with the same background). This aligns baselines: both (A) and (B) sum to
m(x) - E_bg[m], so per-feature values are directly comparable.

Gold standard = well-converged KernelSHAP (the principled outcome-level Shapley
both methods target). We first VALIDATE that interventional TreeSHAP on a head ==
interventional KernelSHAP on that head (confirms setup + convergence), then
compare chain-rule and a cheap KernelSHAP to the converged reference.

Bonus: also reports the path-dependent (pred_contribs) chain-rule -- the variant
that matches the SAVED production npz -- against the same reference.
"""
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import xgboost as xgb
import shap
from config import X_TEST_CONTROL, FEATURE_LIST_811, COMPONENTS
from model import load_models, predict_components, head_shap, meta_shap

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)


def margin_meta(X, b):
    comps = predict_components(pd.DataFrame(X, columns=FEATURE_LIST_811), b).astype(np.float32)
    d = xgb.DMatrix(comps, feature_names=b.meta.get_booster().feature_names)
    return b.meta.get_booster().predict(d, output_margin=True)


def margin_head(X, b, c):
    bo = b.heads[c].get_booster()
    return bo.predict(xgb.DMatrix(np.asarray(X, np.float32), feature_names=bo.feature_names),
                      output_margin=True)


def interventional_tree_shap(model, bg, X):
    """Interventional TreeSHAP in raw-margin space, background = bg."""
    expl = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional",
                              model_output="raw")
    return np.asarray(expl.shap_values(X, check_additivity=False))


def chain_rule_from(psi_by_c, phi_meta, mode="signed"):
    """phi^CR_j = sum_c phi_c * w_{c,j}, both modes additive (sum_j w = 1 per c).

    mode="signed": w = psi/sum(psi)  -- correct fractional contribution, but blows
        up when a head's SHAP nearly cancels (sum~0 while individual psi large).
    mode="mag":    w = |psi|/sum|psi| -- bounded (no blow-up) but sign-blind: every
        biomarker in a component inherits sign(phi_c).
    """
    n = phi_meta.shape[0]
    cr = np.zeros((n, 811))
    for ci, c in enumerate(COMPONENTS):
        psi = psi_by_c[c]
        if mode == "mag":
            num = np.abs(psi); denom = num.sum(1, keepdims=True)
        else:
            num = psi; denom = psi.sum(1, keepdims=True)
        safe = np.abs(denom[:, 0]) > 1e-9
        w = np.zeros_like(psi)
        w[safe] = num[safe] / denom[safe]
        cr += phi_meta[:, ci:ci+1] * w
    return cr


def kernel_shap(f, bg, X, nsamples):
    # l1_reg=0.0 is essential: shap's default "auto" LASSO-sparsifies the Shapley
    # regression when nsamples<2^M (always, here), biasing 811-feature estimates.
    return np.asarray(shap.KernelExplainer(f, bg, silent=True)
                      .shap_values(X, nsamples=nsamples, l1_reg=0.0, silent=True))


def metrics(ref, m):
    """ref, m: (n,811). Focus on meaningful (top-by-ref) features, not 811-wide noise."""
    n = ref.shape[0]
    out = {"pearson_all": [], "pearson_top50": [], "top20_jac": [], "sign_top20": [], "rmse_top50": []}
    for i in range(n):
        topR = np.argsort(np.abs(ref[i]))[::-1][:50]
        t20r = set(np.argsort(np.abs(ref[i]))[::-1][:20])
        t20m = set(np.argsort(np.abs(m[i]))[::-1][:20])
        out["pearson_all"].append(np.corrcoef(ref[i], m[i])[0, 1])
        out["pearson_top50"].append(np.corrcoef(ref[i, topR], m[i, topR])[0, 1])
        out["top20_jac"].append(len(t20r & t20m) / len(t20r | t20m))
        out["sign_top20"].append(np.mean(np.sign(ref[i, list(t20r)]) == np.sign(m[i, list(t20r)])))
        denom = np.sqrt(np.mean(ref[i, topR] ** 2))
        out["rmse_top50"].append(np.sqrt(np.mean((ref[i, topR] - m[i, topR]) ** 2)) / (denom + 1e-12))
    return {k: np.array(v, dtype=float) for k, v in out.items()}   # per-instance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--bg", type=int, default=50)
    ap.add_argument("--ref", type=int, default=16000)   # converged (see convergence test)
    ap.add_argument("--cheap", type=int, default=2000)
    args = ap.parse_args()

    X = pd.read_csv(X_TEST_CONTROL, index_col=0)[FEATURE_LIST_811]
    b = load_models()
    sel = RNG.choice(len(X), size=args.n + args.bg, replace=False)
    Xn = X.iloc[sel[:args.n]].to_numpy(np.float32)
    bg = X.iloc[sel[args.n:].tolist()].to_numpy(np.float32)
    comps_bg = predict_components(pd.DataFrame(bg, columns=FEATURE_LIST_811), b).astype(np.float32)
    comps_n = predict_components(pd.DataFrame(Xn, columns=FEATURE_LIST_811), b).astype(np.float32)

    # ---- 1) validation: interventional TreeSHAP vs interventional KernelSHAP on head EEG ----
    print("== validation: interventional TreeSHAP vs KernelSHAP on head EEG ==")
    tree_eeg = interventional_tree_shap(b.heads["EEG"], bg, Xn)
    ks_eeg = kernel_shap(lambda z: margin_head(z, b, "EEG"), bg, Xn, args.ref)
    vm = metrics(tree_eeg, ks_eeg)
    print(f"   pearson_top50={vm['pearson_top50'].mean():.3f}  top20_jac={vm['top20_jac'].mean():.3f}  "
          f"sign_top20={vm['sign_top20'].mean():.3f}   (high => setup/convergence OK)")

    # ---- 2) chain-rule variants (interventional) ----
    psi_iv = {c: interventional_tree_shap(b.heads[c], bg, Xn) for c in COMPONENTS}
    phi_meta_iv = interventional_tree_shap(b.meta, comps_bg, comps_n)        # (n,3)
    cr_iv = chain_rule_from(psi_iv, phi_meta_iv, mode="signed")
    cr_mag = chain_rule_from(psi_iv, phi_meta_iv, mode="mag")

    # path-dependent chain-rule (matches the saved production npz)
    hs_pd = head_shap(pd.DataFrame(Xn, columns=FEATURE_LIST_811), b)
    phi_meta_pd = meta_shap(comps_n, b)[:, :3]
    cr_pd = chain_rule_from({c: hs_pd[c][:, :811] for c in COMPONENTS}, phi_meta_pd)

    # ---- 3) KernelSHAP on full outcome ----
    ks_ref = kernel_shap(lambda z: margin_meta(z, b), bg, Xn, args.ref)
    ks_cheap = kernel_shap(lambda z: margin_meta(z, b), bg, Xn, args.cheap)

    # efficiency (all should sum to m(x) - E_bg[m])
    fmarg = margin_meta(Xn, b); base = margin_meta(bg, b).mean()
    eff = lambda M: float(np.max(np.abs(M.sum(1) - (fmarg - base))))
    print("\n== efficiency: max|sum(phi) - (m(x) - E_bg[m])| ==")
    print(f"   chain-rule(interv)={eff(cr_iv):.2e}   KernelSHAP(ref)={eff(ks_ref):.2e}   "
          f"(chain-rule path-dep uses a different baseline, not shown)")

    print(f"\n== closeness to reference KernelSHAP (nsamples={args.ref}, n={args.n}, bg={args.bg}) ==")
    print("   means are outlier-sensitive; medians + KernelSHAP-cheap win-rate shown too")
    rows = [("chain-rule signed-share", cr_iv),
            ("chain-rule magnitude-share", cr_mag),
            ("chain-rule path-dependent", cr_pd),
            (f"KernelSHAP n={args.cheap}", ks_cheap)]
    res = {name: metrics(ks_ref, M) for name, M in rows}
    ksm = res[f"KernelSHAP n={args.cheap}"]
    print(f"   {'method':<28}{'pear_top50 (mean/med)':>24}{'top20_jac':>11}{'nRMSE (mean/med)':>20}")
    for name, _ in rows:
        mm = res[name]
        p, r = mm["pearson_top50"], mm["rmse_top50"]
        print(f"   {name:<28}{p.mean():>11.3f} /{np.median(p):>9.3f}{mm['top20_jac'].mean():>11.3f}"
              f"{r.mean():>11.2f} /{np.median(r):>7.2f}")
    print("\n   per-instance win-rate of cheap KernelSHAP vs each chain-rule (fraction where KS is closer):")
    for name, _ in rows[:3]:
        mm = res[name]
        wp = float(np.mean(ksm["pearson_top50"] > mm["pearson_top50"]))
        wr = float(np.mean(ksm["rmse_top50"] < mm["rmse_top50"]))
        wj = float(np.mean(ksm["top20_jac"] > mm["top20_jac"]))
        print(f"     vs {name:<28} pearson:{wp:>4.0%}  nRMSE:{wr:>4.0%}  top20_jac:{wj:>4.0%}")

    np.savez_compressed("artifacts/shap_compare.npz",
                        chain_rule_interv=cr_iv, chain_rule_mag=cr_mag, chain_rule_pathdep=cr_pd,
                        ks_ref=ks_ref, ks_cheap=ks_cheap,
                        instance_index=sel[:args.n], feature_names=np.array(FEATURE_LIST_811))
    print("\n   saved -> artifacts/shap_compare.npz")


if __name__ == "__main__":
    main()
