# Causal Structure Learning for Acquired Acute Brain Dysfunction in the PICU

This repository contains the analysis artifacts for a study evaluating whether clinician expertise and causal structure learning can generate clinically grounded hypotheses about relationships relevant to acquired acute brain dysfunction (ABD) in critically ill children and support more parsimonious predictive models. Hypothesized relationships derived from clinician consensus, GOLEM, and PC-MB are represented as directed acyclic graphs (DAGs), then evaluated through DAG-guided feature selection using Linear Gaussian Bayesian Networks and XGBoost.

The project addresses two questions:

1. Can expert knowledge and data-driven structure learning generate clinically plausible hypotheses about relationships relevant to acquired ABD in the pediatric intensive care unit (PICU)?
2. Can feature sets derived from those hypotheses support models that retain useful predictive performance with fewer biomarkers?

The learned graphs encode causal hypotheses rather than established causal relationships. An edge proposes a relationship for further study; it is not evidence that one variable causes another. Clinician agreement and predictive performance do not establish causal effects.

## Study workflow

1. **Define the cohort and outcome.** The study uses retrospective PICU electronic health record data from 2010–2022 and an Acute Brain Dysfunction computable phenotype. The modeling design uses a 48-hour observation window followed by a 12-hour censor period before an ABD event. Encounters through 2019 are used for training and encounters from 2020–2022 are used for testing.
2. **Prepare longitudinal biomarkers.** Forty-five routinely collected vitals, laboratory results, medications, procedures, and neurologic assessments are imputed with SAITS and summarized as Catch22 time-series features. After variance filtering, the main modeling set contains 811 features.
3. **Elicit expert knowledge.** Four PICU clinicians iteratively specified candidate relationships. Edges supported by at least three clinicians form the consensus DAG and background knowledge for causal search.
4. **Learn and harmonize graphs.** GOLEM and PC-MB provide data-driven structures. The learned graphs are cleaned, reduced to candidate target-related feature sets, and represented at both feature and biomarker levels.
5. **Evaluate predictive utility.** DAG-selected features are used to train Linear Gaussian Bayesian Networks and XGBoost models. Analyses include bootstrap confidence intervals, calibration, SHAP feature importance, removal of medications/interventions, and temporal sensitivity across the 2020–2022 test years. Because ABD is imbalanced, area under the precision-recall curve (AUPRC) is the primary performance measure.

The headline result comes from the vitals-and-laboratory-only experiment, which excludes medications and other interventions. The XGBoost model using the Markov blanket derived from the union of the GOLEM and PC-MB DAGs used 12 biomarkers, that is 16 (57%) fewer than the control and achieved an AUPRC of 0.77 (95% CI 0.73–0.80), compared with 0.78 (95% CI 0.75–0.82) for the 28-biomarker control. The AUPRC permutation test did not detect a statistically significant difference (`p = 0.096`). See the [vitals-and-laboratory-only results](./Predictive%20Modeling/No%20Drugs%20or%20Interventions.csv).

## Repository structure

| Path | Contents |
| --- | --- |
| [`Background Knowledge/`](./Background%20Knowledge/) | Clinician-provided edges, consensus/background-knowledge artifacts, inter-rater agreement results, and the expert-knowledge notebook. |
| [`Causal Search/Tetrad/`](./Causal%20Search/Tetrad/) | Tetrad Causal CMD configuration. |
| [`DAGs/`](./DAGs/) | Cleaned Clinician Consensus, GOLEM, and PC-MB graphs; full and simplified adjacency matrices; union/intersection variants; graph figures; and cleanup utilities. |
| [`Predictive Modeling/`](./Predictive%20Modeling/) | Executed model-parameterization and year-sensitivity notebooks, performance tables, DeLong and permutation tests, calibration curves, and SHAP feature-importance figures. |
| [`Synthetic Data/`](./Synthetic%20Data/) | A generated 100-patient example cohort |
| [`pixi/`](./pixi/) | Python environment definition and lockfile for the main graph and predictive-modeling stack. |

### Graph-file conventions

- `Simplified` graphs pool Catch22 feature nodes to their source biomarker names.
- `+` in a filename denotes a graph union; `x` denotes an intersection. The notebooks display these as $\cup$ and $\cap$, respectively.
- Files ending in `_adjacency.csv` are labeled adjacency matrices. A nonzero value at row `u`, column `v` represents the directed edge `u -> v`.
- `Outcome` is the acquired ABD target node.

## Getting started

The Pixi environment uses Python 3.12 and a single cross-platform lockfile covering 64-bit Intel/AMD Linux (`linux-64`), 64-bit ARM Linux (`linux-aarch64`), Intel macOS (`osx-64`), Apple-silicon macOS (`osx-arm64`), and 64-bit Intel/AMD Windows (`win-64`). On these platforms, Pixi automatically selects the locked environment for the current machine. From the repository root:

```bash
cd pixi
pixi install
pixi run jupyter lab ..
```

This opens the repository in JupyterLab with the main analysis environment. Java 21 or newer is required separately to use the tracked Tetrad command-line JAR. The lockfile is solved against Linux kernel 4.18 with glibc 2.28, macOS 13, and Windows 10. The environment does not yet capture every dependency used by older notebooks. Other targets—including native Windows on ARM (`win-arm64`), 32-bit systems, and mobile operating systems—are not included in the current lockfile.

## Reproducibility boundaries

A fresh clone supports inspection of the executed notebooks, adjacency matrices, result tables, and figures, as well as development against the synthetic data. It does **not** contain everything required to reproduce the clinical results end to end.

## Data governance and responsible use

The files under `Synthetic Data/` are generated examples for understanding the expected schema and developing workflow code. They do not reproduce the reported estimates and must not be used to draw clinical conclusions.

The causal hypotheses require further validation before they can support intervention or causal-effect claims.

## Related manuscript (Pre-print)

[Pérez Claudio E, Horvat CM, Taylor WM, et al. **Leveraging Expert Knowledge and Causal Structure Learning to Build Parsimonious Models of Acute Brain Dysfunction in the Pediatric Intensive Care Unit (PICU).** _medRxiv_. Posted February 18, 2026.](https://www.medrxiv.org/content/10.64898/2026.02.17.26345661v1) [doi:10.64898/2026.02.17.26345661](https://doi.org/10.64898/2026.02.17.26345661)
