# TODO — Causal Effect Estimation (DML / Doubly-Robust ML + Causal Trees)

**Status:** Not started — planned separate analysis.
**Created:** 2026-06-11

## Why this is its own analysis (not the stacking ensemble)

The stacking ensemble in `../Predictive Modeling/Ensemble Model/` is **predictive**:
biomarkers → per-component scores → `Outcome`. Doubly-robust ML (DML) is a
different goal — **estimating the causal effect of a treatment on an outcome**,
adjusting for confounders. DML needs a *treatment* variable and a *propensity*
model; it is consistent if **either** the outcome model or the propensity model
is correct ("double robustness"), and uses **cross-fitting** for valid inference
(Chernozhukov et al., 2018). That structure does not map onto a stacking
meta-learner, so it lives here as a standalone analysis. (The one shared idea —
cross-fitting — is already used in the ensemble's out-of-fold stacking.)

This also aligns more naturally with Aim 1's causal-discovery theme: use the
learned DAGs to justify adjustment sets, then estimate effects.

## Goal

Estimate the causal effect (ATE and CATE / heterogeneous effects) of clinically
actionable **treatments/exposures** on the neuro-morbidity outcome in critically
ill children.

## Key design decisions to make first

- [ ] **Choose treatment(s).** Must be a genuine intervention/exposure, NOT a
  component that *defines* the outcome (EEG/CT/MRI/BH-meds are mechanically part
  of `Outcome` → effect is circular/degenerate). Candidates: a specific drug
  exposure (e.g., Dexmedetomidine vs not), ventilation, CRRT, an exposure
  threshold on a biomarker. Decide binary vs continuous treatment.
- [ ] **Define the outcome for the causal question.** The composite `Outcome`
  may be a poor causal target if the treatment is correlated with its components
  by indication. Consider a cleaner downstream outcome (e.g., mortality, length
  of stay, a chart-adjudicated neuro-morbidity label) to avoid outcome-definition
  leakage.
- [ ] **Adjustment set from the DAGs.** Pull confounders from the causal graphs in
  `../DAGs/` and `../Causal Search/` (back-door / valid adjustment set). Document
  the identification assumption. Check **positivity/overlap** and unconfoundedness.
- [ ] **Cohort & time alignment.** Reuse the real cohort
  (`../../Data Pre-processing/Preprocessing/data/real_cohort/*.parquet`). Define
  treatment timing vs. outcome window to avoid reverse causation (treatment must
  precede outcome).

## Methods to implement (EconML)

- [ ] **DML for ATE/CATE:** `econml.dml.LinearDML`, `econml.dml.CausalForestDML`,
  `econml.dml.NonParamDML`. Nuisance models = gradient boosting / XGBoost
  (consistent with the rest of the pipeline) for both outcome `E[Y|X,W]` and
  treatment `E[T|X,W]`; cross-fitting folds.
- [ ] **Doubly-robust learners:** `econml.dr.DRLearner`,
  `econml.dr.LinearDRLearner`, `econml.dr.ForestDRLearner` — compare against DML.
- [ ] **Causal trees / forests (heterogeneity):** `econml.grf.CausalForest`,
  honest causal trees (Athey & Imbens, 2016; Wager & Athey, 2018). Use
  `CausalForestDML` for honest CATE estimation; inspect tree splits for
  interpretable effect-modifier subgroups (which patients benefit/are harmed).
- [ ] **Policy learning (optional):** `econml.policy` to derive a treatment rule
  from the estimated CATE.

## Validation & robustness

- [ ] Overlap/propensity diagnostics; trim or weight if positivity is weak.
- [ ] Refutation/sensitivity analyses (placebo treatment, random common cause,
  subset refuter — DoWhy integrates with EconML for this).
- [ ] Confidence intervals via DML's asymptotic normality; bootstrap for forests.
- [ ] Sensitivity to unmeasured confounding (e.g., E-value, Rosenbaum bounds).

## Environment (ready ✅)

Isolated **pixi** env in this folder (`pixi.toml`) — kept separate from `causal_inf`
because econml 0.16 pins `scikit-learn <1.7`. Installed & smoke-tested:
econml 0.16.0 · shap 0.48.0 · scikit-learn 1.6.1 · xgboost 3.2.0 · numpy 2.4.6.

```bash
pixi run --manifest-path "Causal Effect Estimation/pixi.toml" smoke   # CausalForest -> SHAP check
pixi run --manifest-path "Causal Effect Estimation/pixi.toml" lab     # jupyter lab
```

`smoke_test.py` is the working **CausalForest → SHAP** template: it fits a
`CausalForestDML` on synthetic data with a known heterogeneous effect (recovered at
corr 0.99) and pulls `est.shap_values(X)`, which correctly isolates the true effect
modifiers. Swap in the real cohort + chosen treatment/adjustment set to make it real.

**SHAP note (don't conflate the two):** `est.shap_values(X)` on a CATE model explains
*effect heterogeneity* — who benefits from the treatment and why — NOT a risk
prediction. That is distinct from TreeSHAP on the predictive XGBoost/ensemble, which
explains `P(Outcome|X)` (associational). Same algorithm, different estimand.

- [ ] `dowhy` (optional, for refutation) — add to `pixi.toml` if/when needed.
- [ ] Scaffold: `causal_effects.py` (data + adjustment set), `dml_estimation.py`,
  `causal_forest.py`, `refutation.py`, a results notebook.

## References

- Chernozhukov et al. (2018), *Double/Debiased Machine Learning* — Econometrics J.
- Athey & Imbens (2016), *Recursive partitioning for heterogeneous causal effects* — PNAS.
- Wager & Athey (2018), *Estimation and inference of heterogeneous treatment effects using random forests* — JASA.
- Kennedy (2023), *Towards optimal doubly robust estimation of heterogeneous causal effects* (DR-learner).
- EconML docs: https://econml.azurewebsites.net/  ·  DoWhy: https://www.pywhy.org/
