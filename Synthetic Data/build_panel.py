"""Build synthetic_data.csv: a wide panel from the long observation tables.

Rows are one per (uid, hour) on each patient's own contiguous 72h grid; columns
are one per biomarker. Missing cells are left as NaN -- SAITS consumes the gaps,
and the Catch22 path two-stage-fills them first (see Synthetic Patient
Generation.ipynb).

Each patient's window sits at a different absolute norm_timestamp (10 distinct
starts, spanning 141 distinct timestamps overall), so `hour` is indexed 0-71
relative to that patient's own window start. That is what makes the panel
reshape cleanly to [n_uid, 72, n_feature]; norm_timestamp is retained alongside
it for traceability back to the source tables.
"""

import pandas as pd

N_STEPS = 72
OUT = "synthetic_data.csv"

# Taxonomy order, so related biomarkers sit together in the CSV.
TYPE_ORDER = ["vital", "lab", "score", "demographic", "intervention", "outcome"]


def build() -> pd.DataFrame:
    num = pd.read_csv(
        "synthetic_observations_numeric.csv", parse_dates=["norm_timestamp"]
    )
    cat = pd.read_csv(
        "synthetic_observations_categorical.csv", parse_dates=["norm_timestamp"]
    )
    tax = pd.read_csv("synthetic_column_taxonomy.csv")

    num["value"] = pd.to_numeric(num["value"], errors="coerce")
    long = pd.concat([num, cat], ignore_index=True)

    # Hour relative to each patient's own window start.
    start = long.groupby("uid")["norm_timestamp"].transform("min")
    long["hour"] = (
        (long["norm_timestamp"] - start).dt.total_seconds() // 3600
    ).astype(int)

    panel = long.pivot(index=["uid", "hour"], columns="column_name", values="value")

    # Reindex onto the full uid x 72h grid so every patient is equal length.
    full = pd.MultiIndex.from_product(
        [sorted(long["uid"].unique()), range(N_STEPS)], names=["uid", "hour"]
    )
    panel = panel.reindex(full)

    # Order columns by clinical type, then name.
    rank = {c: (TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER))
            for c, t in zip(tax["column_name"], tax["clinical_type"])}
    cols = sorted(panel.columns, key=lambda c: (rank.get(c, len(TYPE_ORDER)), c))
    panel = panel[cols]

    # Carry norm_timestamp so rows trace back to the source tables.
    ts = long.groupby(["uid", "hour"])["norm_timestamp"].first()
    panel.insert(0, "norm_timestamp", ts.reindex(full))

    return panel.reset_index()


if __name__ == "__main__":
    panel = build()
    panel.to_csv(OUT, index=False)
    n_uid = panel["uid"].nunique()
    feats = [c for c in panel.columns if c not in ("uid", "hour", "norm_timestamp")]
    filled = panel[feats].notna().to_numpy().mean() * 100
    print(f"wrote {OUT}: {len(panel):,} rows = {n_uid} uid x {N_STEPS}h")
    print(f"  {len(feats)} feature columns, {filled:.1f}% observed")
