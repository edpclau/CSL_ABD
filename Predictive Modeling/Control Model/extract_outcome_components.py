#!/usr/bin/env python
"""
Extract the per-patient OUTCOME COMPONENT labels (BH, EEG, CT, MRI, BHMeds,
and the individual antipsychotics) for the exact individuals used by the
control model, aligned by `uid` to its train/test splits.

The composite `Outcome` the control model predicts is the OR of these
components (CLAUDE.md: CT-head, MRI-brain, EEG, and BH-consult AND antipsychotic
med within 72h). These component columns already live in the same label files
the control model drew `Outcome` from; this script simply selects them and
aligns them to the control cohort.

Outputs (next to this script):
  - outcome_components_train.csv   per-uid component labels, control train rows
  - outcome_components_test.csv    per-uid component labels, control test rows
  - outcome_components_summary.csv positive counts/prevalence per split
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[2] / "Data Pre-processing"

COMPONENTS = [
    "BH", "EEG", "CT", "MRI", "Meds", "BHMeds",
    "Haloperidol", "Olanzapine", "Dexmedetomidine", "Outcome",
]

# Component labels (same files the control model's Outcome came from).
ytr = pd.read_csv(DATA_DIR / "y_train_c12_w48_imp.csv").groupby("uid")[COMPONENTS].max()
yte = pd.read_csv(DATA_DIR / "y_test_c12_w48_imp.csv").groupby("uid")[COMPONENTS].max()

# Align to the control model's exact individuals (preserves row order).
ctrl_tr = pd.read_csv(SCRIPT_DIR / "y_train_control.csv", index_col=0).index
ctrl_te = pd.read_csv(SCRIPT_DIR / "y_test_control.csv", index_col=0).index
assert ctrl_tr.isin(ytr.index).all() and ctrl_te.isin(yte.index).all(), "uid mismatch"

train = ytr.reindex(ctrl_tr)
test = yte.reindex(ctrl_te)
train.to_csv(SCRIPT_DIR / "outcome_components_train.csv")
test.to_csv(SCRIPT_DIR / "outcome_components_test.csv")

# Summary of positives/prevalence per split.
rows = []
for split, df in [("train", train), ("test", test)]:
    for c in COMPONENTS:
        rows.append({
            "split": split, "component": c,
            "n_positive": int(df[c].sum()), "n": len(df),
            "prevalence": round(float(df[c].mean()), 4),
        })
summary = pd.DataFrame(rows)
summary.to_csv(SCRIPT_DIR / "outcome_components_summary.csv", index=False)

print(summary.to_string(index=False))
print(f"\nSaved component labels for {len(train)} train + {len(test)} test individuals to:\n  {SCRIPT_DIR}")
